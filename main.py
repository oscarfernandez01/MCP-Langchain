import os
import imaplib
import email
from email.header import decode_header
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, List

# --- Cargar .env ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
IMAP_SERVER = os.getenv("IMAP_SERVER", "outlook.office365.com")
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
def get_emailpdf_outlook_imap(schema_filter: str) -> Dict[str, Any]:
    if not os.path.exists(DOWNLOAD_PATH):
        os.makedirs(DOWNLOAD_PATH)

    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL_USER, EMAIL_PASS)
    mail.select("inbox")

    status, messages = mail.search(None, '(UNSEEN)')
    email_ids = messages[0].split()

    valid_emails = []
    attachments_saved: List[str] = []

    for eid in email_ids:
        _, msg_data = mail.fetch(eid, "(RFC822)")
        raw_email = msg_data[0][1]
        msg = email.message_from_bytes(raw_email)

        subject, encoding = decode_header(msg["Subject"])[0]
        if isinstance(subject, bytes):
            subject = subject.decode(encoding or "utf-8")

        # Clasificar asunto con LLM
        schema_prompt = f"""
        Tarea: Validar si el siguiente asunto corresponde al schema {schema_filter}.
        Responde SOLO 'VALIDO' o 'NO VALIDO'.
        Asunto: {subject}
        """
        response = llm.invoke([HumanMessage(content=schema_prompt)])
        if "VALIDO" in response.content.upper():
            valid_emails.append(msg)

    if len(valid_emails) == 1:
        email_msg = valid_emails[0]
        for part in email_msg.walk():
            if part.get_content_disposition() == "attachment":
                filename = part.get_filename()
                if filename:
                    filepath = os.path.join(DOWNLOAD_PATH, filename)
                    with open(filepath, "wb") as f:
                        f.write(part.get_payload(decode=True))
                    attachments_saved.append(filepath)

    status = "SUCCESS" if len(attachments_saved) == 1 else "PENDING"
    return {
        "status": status,
        "message": f"Se encontraron {len(valid_emails)} correos válidos y {len(attachments_saved)} adjuntos",
        "attachments": attachments_saved
    }

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

        else:
            return {"error": f"Herramienta no soportada: {tool}"}

    except Exception as e:
        return {"error": f"Ocurrió un error interno: {str(e)}"}
