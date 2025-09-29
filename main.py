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
    "prompt":"Analiza el siguiente texto {{}}, es de una requisicion, necesito que extraigas informacion de este documento. Primero necesito la date: El campo se llama: \"Fecha de:\", la información está debajo (Ejempos: 11/06/2020, 11/06/202). Luego extrae la purchase_order_number: El campo se llama: \"No.\", la información está debajo (Ejemplo: REQ-002455). Luego extrae general_description: El campo se llama: \"Descripción\", la información está justo debajo del nombre del campo. Si no se encuentra, devuelve \"null\" (Ejemplo: EPP personal nuevo ingreso) es la descripción de la orden de compra, no puede ser la misma que la descripción de los artículos en la tabla, la encontraras después del numero de requisición y antes del campo Usuario. Después extrae soliciting_user_code: El campo se llama: \"Usuario\", es un código alfa numérico, lo encontraras en el campo usuario. Luego extrae el campo soliciting_user_name: Este campo se encuentra inmediatamente junto al texto \"solicitante\", si no lo encuentras déjalo como null y no uses ningún otro nombre del texto debe estar únicamente junto al texto 'solicitante' y no puede ser el mismo que contact_name. Si este campo es el mismo que el soliciting_user_code, devuelve este campo como null. Luego extrae el provider_number: es un código alfanumérico que empieza con una letra seguido de 5 números encontrado justo antes del texto' Proveedor No.' (ejemplo: P00001). Después extrae el contact_name: El campo se llama: \"Contacto\", este dato unicamente se encuentra después de este texto: 'No.\nCorreo \nelectrónico\nContacto\n' y antes del correo electrónico ejemplo: '\nCorreo \nelectrónico\nContacto\nSUBETE A LA NUBE S DE RL DE CV\nDiana@octapus.io' el contacto es: 'SUBETE A LA NUBE S DE RL DE CV' el contacto puede no estar relacionado al correo de contacto. Luego extrae el campo contact_email: este campo es una dirección o conjunto de direcciones de correo electrónico separados por ';' (Ejemplo: example@email.com) este campo debe cumplir con esa nomenclatura porque si no lo hace entonces debe ser 'null'. Luego extrae el campo tax_code: El campo es el código alfanumerico de 12 caracteres que se encuentra antes de la palabra 'Código Fiscal.' (Ejemplo: CULM6401301F3). Luego extrae total_cost: Este campo es el costo total de la orden de compra. Está ubicado al final de la tabla Para line_items Extrae los siguientes campos. Extrae el product_type: unicamente puede ser 'Producto' este es el dato que debes de extraer en este campo y no usar otro campo o texto adicional nunca puede ser nada diferente a 'Producto'. Luego extrae product_number: El encabezado de la columna se llama: \"No.\"  algunos nombres pueden aparecer cortados, ejemplo: 'UNIFORME S, EQ DE SEG' corregido debe quedar 'UNIFORMES, EQ DE SEG' o 'SERVICIO AMBIENTA L' corregido debe quedar 'SERVICIO AMBIENTAL', aquí tienes ejemplos de valores para este campo ya corregidos (Ejemplos: HERRAM Y ARTIC MTTO, UNIFORMES, EQ DE SEG, MANTENIMIENTO, C515-008-000, ART DE LIMPIEZA, SERVICIO AMBIENTAL, SERV DE TRANSPORTE, SERV RECOL DE BASURA, MTTO A LOS SITEMAS, MAT AUX DE PRODUCC, NITROGENO) es un texto de máximo 4 palabras alfanumericas y Asegúrate de que no esté concatenado con la descripción del producto ni que sea el mismo que cost_center o charge ni que sea una unidad de medida de measure_unit. Luego extrae la product_description: este campo es el texto antes de la unit_quantity sin incluir el numero que corresponde al campo de unit_quantity separalos correctamente. Luego extrae el campo unit_quantity: este campo representa cuántas unidades se ordenan de ese artículo, este campo es el numero completo antes del measure unit. Después extrae measure_unit: es la unidad de medida para cada producto, pueden ser piezas, litros, kilos etc. (Ejemplos: PAR, PIEZA, LT, SERV, METR),  esta información es única y se encuentra justo después a la cantidad de ítems así que sepáralos adecuadamente sabiendo que el measure unit va después de la cantidad y la cantidad antes que el measure unit. Luego extrae el campo unit_price: El encabezado de la columna se llama: \"Precio\", Este campo representa el precio unitario para cada producto. Luego extrae discount: El encabezado de la columna se llama: \"Dto.\", este campo representa el descuento aplicado al precio unitario. Después extrae el campo total_price: El encabezado de la columna se llama: \"Cantidad\", este campo representa el precio total del artículo. Luego extrae el cost_center: El encabezado de la columna se llama: \"C-cto\", este campo representa el código del centro de costos, es un código alfanumérico de 9 caracteres separados por '-', la estructura es 3 letras/numeros-3 letras/numeros-3 letras/números, no lleva espacio entre los caracteres ni los guiones aquí tienes un ejemplo: '3HT-MM1-MPO'. Después extrae el campo charge: El encabezado de la columna se llama: \"Cobro\" (Ejemplo: CLIENTE). es una palabra. ** instrucciones importantes** Reemplaza todos los saltos de línea (\\n) por espacios pero deja un solo espacio entre palabras, además si algún campo termina con un espacio al final elimina ese espacio. Si encuentras comillas dobles (\") o dos comillas simples juntas en cualquier campo (\") reemplaza por asterisco (*). No corrijas ningún error de palabras o sintaxis en la extracción regresa los textos tal cual se extraen. Si no cumple algún campo con su nomenclatura establecida en las instrucciones devuélvelo como 'null'.  Usando el siguiente diccionario {{OfensiveWords}} cambia las palabras que coincidan por las que se encuentran en su equivalente clave del diccionario. No agregues ni inventes información adicional o inexistente. Formato de Respuesta: Entrega la respuesta en un JSON respetando la siguiente estructura: JSON {\"date\":\"\",\"purchase_order_number\":\"\",\"general_description\":\"\",\"soliciting_user_name\":\"\",\"soliciting_user_code\":\"\",\"provider_number\":\"\",\"contact_name\":\"\",\"contact_email\":\"\",\"tax_code\":\"\",\"total_cost\":\"\",\"line_items\":[{\"product_type\":\"\",\"product_number\":\"\",\"product_description\":\"\",\"unit_quantity\":\"\",\"measure_unit\":\"\",\"unit_price\":\"\",\"discount\":\"\",\"total_price\":\"\",\"cost_center\":\"\",\"charge\":\"\"}]}"
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
