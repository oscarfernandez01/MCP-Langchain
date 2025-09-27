import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict

# Cargar .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY no está configurada en el .env")

# LangChain moderno compatible
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

app = FastAPI(title="Servidor MCP LangChain")

# LLM y prompt
llm = ChatOpenAI(model="gpt-4", temperature=0.5, api_key=OPENAI_API_KEY)
prompt = PromptTemplate.from_template("Resume el siguiente texto: {texto}")

def resumir_texto(texto):
    full_prompt = prompt.format(texto=texto)
    response = llm.invoke(full_prompt)
    return response.content

# Modelo de request MCP
class MCPRequest(BaseModel):
    action: str
    parameters: Dict[str, Any]

@app.post("/mcp")
def handle_mcp(request: MCPRequest):
    action = request.action
    params = request.parameters

    try:
        if action == "sumar":
            resultado = params.get("a", 0) + params.get("b", 0)
            return {"result": resultado}

        elif action == "resumir_texto":
            texto = params.get("texto", "")
            resumen = resumir_texto(texto)
            return {"resumen": resumen}

        else:
            return {"error": "Acción no soportada"}

    except Exception as e:
        return {"error": f"Ocurrió un error interno: {str(e)}"}
