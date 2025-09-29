import os
import imaplib
import email
import base64
import logging
import requests
from email.header import decode_header
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List

# --- Cargar .env ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMAIL_USER = os.getenv("EMAIL_USER")  # tu correo Gmail
EMAIL_PASS = os.getenv("EMAIL_PASS")  # contraseña de aplicación
IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.gmail.com")
DOWNLOAD_PATH = "./attachments"

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY no está configurada en el .env")
if not EMAIL_USER or not EMAIL_PASS:
    raise RuntimeError("EMAIL_USER o EMAIL_PASS no están configurados en el .env")

# --- LangChain moderno compatible ---
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

app = FastAPI(title="Servidor MCP LangChain")

# --- LLM global ---
llm = ChatOpenAI(model="gpt-4", temperature=0.5, api_key=OPENAI_API_KEY)
prompt = PromptTemplate.from_template("Resume el siguiente texto: {texto}")

def resumir_texto(texto):
    full_prompt = prompt.format(texto=texto)
    response = llm.invoke(full_prompt)
    return response.content

# --- NUEVA FUNCIÓN: Obtener correos y adjuntos ---
# Configuración
DOWNLOAD_PATH = "./attachments"
MAX_EMAILS_TO_CHECK = 50  # últimos N correos
KEYWORDS = ["REQ", "REQUISICION"]  # palabras clave para filtrar asunto antes de LLM

def get_emailpdf_outlook_imap(schema_filter: str) -> Dict[str, any]:
    if not os.path.exists(DOWNLOAD_PATH):
        os.makedirs(DOWNLOAD_PATH)

    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL_USER, EMAIL_PASS)
    mail.select("inbox")

    # Tomar todos los correos no leídos
    status, messages = mail.search(None, '(UNSEEN)')
    email_ids = messages[0].split()

    # Solo últimos N correos para acelerar
    email_ids = email_ids[-MAX_EMAILS_TO_CHECK:]

    valid_emails = []
    attachments_saved: List[str] = []

    for eid in email_ids:
        _, msg_data = mail.fetch(eid, "(RFC822)")
        raw_email = msg_data[0][1]
        msg = email.message_from_bytes(raw_email)

        subject, encoding = decode_header(msg["Subject"])[0]
        if isinstance(subject, bytes):
            subject = subject.decode(encoding or "utf-8")

        # Filtrar rápido por palabras clave antes de llamar al LLM
        if not any(k in subject.upper() for k in KEYWORDS):
            continue

        # Clasificar asunto con LLM
        schema_prompt = f"""
        Tarea: Validar si el siguiente asunto corresponde al schema {schema_filter}.
        Responde SOLO 'VALIDO' o 'NO VALIDO'.
        Asunto: {subject}
        """
        response = llm.invoke([HumanMessage(content=schema_prompt)])
        if "VALIDO" in response.content.upper():
            valid_emails.append(msg)

    # Descargar attachments de todos los correos válidos
    for email_msg in valid_emails:
        for part in email_msg.walk():
            if part.get_content_disposition() == "attachment":
                filename = part.get_filename()
                if filename:
                    filepath = os.path.join(DOWNLOAD_PATH, filename)
                    with open(filepath, "wb") as f:
                        f.write(part.get_payload(decode=True))
                    attachments_saved.append(filepath)

    status = "SUCCESS" if attachments_saved else "NO_CORREOS"
    return {
        "status": status,
        "message": f"Se encontraron {len(valid_emails)} correos válidos y {len(attachments_saved)} adjuntos",
        "attachments": attachments_saved
    }



# Constantes para la conexión con Gemini y Google Sheets.

URL_GEMINI = os.getenv("URL_GEMINI") 
# Texto del prompt a enviar a Gemini.
TEXT_PROMPT = '''{
"Analiza la imagen del **anverso (frente)** de la identificación oficial proporcionada. La identificación puede ser un INSTITUTO NACIONAL ELECTORAL (INE), INSTITUTO FEDERAL ELECTORAL (IFE), pasaporte o licencia de conducir.\n\nTu tarea es extraer **únicamente** la siguiente información del anverso:\n\n- **nombre:**\n  - **Campo:** Busca el campo \"NOMBRE\".\n  - **Requisitos:** Extrae el texto completo tal como aparece, incluyendo apellidos y nombre(s). Reemplaza cualquier salto de línea por un espacio. Si no se encuentra, devuelve null.\n\n- **domicilio:**\n  - **Campo:** Busca el campo \"DOMICILIO\".\n  - **Requisitos:** Extrae la dirección completa. Reemplaza cualquier salto de línea por un espacio. Si no se encuentra, devuelve null.\n\n- **clave_elector:**\n  - **Campo:** Busca el campo \"CLAVE DE ELECTOR\".\n  - **Requisitos:** Extrae la cadena de 18 caracteres. Si el documento no es una 'INE' o 'IFE', este campo debe ser null. Si no se encuentra, devuelve null.\n\n- **curp:**\n  - **Campo:** Busca el campo \"CURP\".\n  - **Requisitos:** Extrae la cadena de 18 caracteres. Si no se encuentra, devuelve null.\n\n- **sexo:**\n  - **Campo:** Busca el campo \"SEXO\".\n  - **Requisitos:** Devuelve 'H' para hombre o 'M' para mujer. Normaliza la salida: si encuentras 'M' (masculino en pasaporte) debe ser 'H', y si encuentras 'F' (femenino en pasaporte) debe ser 'M'. Si no se encuentra, devuelve null.\n\n- **fecha_nacimiento:**\n  - **Campo:** Busca el campo \"FECHA DE NACIMIENTO\".\n  - **Requisitos:** Extrae la fecha en formato DD/MM/AAAA. Si no se encuentra, devuelve null.\n\n- **vigencia:**\n  - **Campo:** Busca el campo \"VIGENCIA\".\n  - **Requisitos:** Extrae y devuelve **únicamente el último año de vigencia** en formato de 4 dígitos (AAAA). Por ejemplo, si dice \"2014-2024\", devuelve \"2024\". Si solo dice \"2025\", devuelve \"2025\". Si no se encuentra, devuelve null.\n\n**Formato de Respuesta Obligatorio:**\nEntrega la respuesta exclusivamente en un objeto JSON con la siguiente estructura. No agregues texto, explicaciones ni comentarios adicionales.\n\n{\n  \"nombre\": \"Value\",\n  \"domicilio\": \"Value\",\n  \"clave_elector\": \"Value\",\n  \"curp\": \"Value\",\n  \"sexo\": \"Value\",\n  \"fecha_nacimiento\": \"Value\",\n  \"vigencia\": \"Value\"\n}"
}'''

ATTACHMENTS_DIR = "attachments"  # Carpeta donde están los archivos

def read_pdffile_ai_api_from_folder():
    """
    Procesa todos los archivos PDF e imágenes en la carpeta `attachments`,
    los convierte en base64 y los envía a la API de Gemini para extraer su texto.
    
    Returns:
        list: Lista con las respuestas de Gemini (o errores).
    """
    logging.info(f"Processing files in folder: {ATTACHMENTS_DIR}")
    responses = []

    # Listar todos los archivos en la carpeta
    for filename in os.listdir(ATTACHMENTS_DIR):
        pdf_file_path = os.path.join(ATTACHMENTS_DIR, filename)
        
        # Ignorar subcarpetas
        if not os.path.isfile(pdf_file_path):
            continue

        logging.info(f"Processing file: {pdf_file_path}")
        try:
            # Obtener extensión
            _, extension = os.path.splitext(pdf_file_path)
            extension = extension.lower()

            # Leer archivo y convertir a base64
            with open(pdf_file_path, "rb") as pdf_file:
                content = base64.b64encode(pdf_file.read()).decode("utf-8")

            # Determinar tipo de archivo
            if extension == ".pdf":
                mimetype = "application/pdf"
                data_key = "pdf"
            elif extension in [".jpg", ".jpeg"]:
                mimetype = "image/jpeg"
                data_key = "image"
            elif extension == ".png":
                mimetype = "image/png"
                data_key = "image"
            else:
                logging.warning(f"Unsupported file type: {extension}. Skipping file.")
                responses.append({"error": f"Unsupported file type: {extension}"})
                continue

            # Preparar payload
            payload = {
                data_key: content,
                "prompt": TEXT_PROMPT,
                "mimetype": mimetype
            }

            # Enviar a Gemini
            logging.info(f"Sending file to Gemini API: {pdf_file_path}")
            try:
                response = requests.post(URL_GEMINI, json=payload)
                response.raise_for_status()
                responses.append(response.json())
                logging.info("Response received from Gemini API successfully.")
            except requests.exceptions.RequestException as e:
                logging.error(f"Error sending data to Gemini: {e}")
                responses.append({"error": f"Error sending data to Gemini: {e}"})

        except Exception as e:
            logging.error(f"Error processing file {pdf_file_path}: {e}")
            responses.append({"error": str(e)})

    return responses

# --- Modelo de request MCP ---
class MCPRequest(BaseModel):
    tool: str  # <-- AHORA ES "tool"
    parameters: Dict[str, Any]

@app.post("/mcp")
def handle_mcp(request: MCPRequest):
    tool = request.tool
    params = request.parameters

    try:
        if tool == "get_emailpdf_outlook_imap":
            schema = params.get("schema", "")
            resultado = get_emailpdf_outlook_imap(schema)
            return resultado
        elif tool == "read_pdffile_ai_api":
            # Llamamos directamente a la función que procesa la carpeta attachments
            resultado = read_pdffile_ai_api_from_folder()  # Esta es la versión que lee la carpeta
            return {"result": resultado}

        else:
            return {"error": f"Herramienta no soportada: {tool}"}

    except Exception as e:
        return {"error": f"Ocurrió un error interno: {str(e)}"}
