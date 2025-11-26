import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
from dotenv import load_dotenv

from mistralai import Mistral

# --- CONFIGURACIÓN INICIAL (Sin cambios) ---
load_dotenv()
api_key = os.getenv('MISTRAL_API_KEY')

if not api_key:
    raise ValueError("No se encontró MISTRAL_API_KEY. Asegúrate de que tu archivo .env está configurado.")

client = Mistral(api_key=api_key)

app = FastAPI(
    title="API de OCR con Mistral AI - MODO DIAGNÓSTICO",
    description="Sube un archivo PDF para descubrir la estructura de la respuesta de Mistral OCR."
)


# --- ENDPOINT DE LA API (MODO DIAGNÓSTICO) ---
@app.post("/extraer-texto-pdf/")
async def extraer_texto_de_pdf(
    archivo: UploadFile = File(..., description="El archivo PDF que deseas procesar.")
):
    if archivo.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Error: El archivo proporcionado no es un PDF.")

    try:
        contenido_bytes = await archivo.read()
        nombre_base = Path(archivo.filename).stem

        # 1. Subir archivo (Sin cambios)
        print("Subiendo el archivo...")
        uploaded_file = client.files.upload(
            file={"file_name": nombre_base, "content": contenido_bytes},
            purpose="ocr"
        )

        # 2. Obtener URL (Sin cambios)
        print("Obteniendo URL firmada...")
        signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)

        # 3. Llamar al OCR (Sin cambios)
        documento_para_ocr = {"document_url": signed_url.url}
        print("Enviando a procesar con OCR...")
        response = client.ocr.process(
            document=documento_para_ocr,
            model='mistral-ocr-latest'
        )
        print("¡Respuesta recibida de Mistral!")

        # --- CAMBIO IMPORTANTE: MODO DIAGNÓSTICO ---
        # En lugar de intentar procesar la respuesta, vamos a investigarla.
        
        # 1. Imprimimos el TIPO de objeto que es 'response' en la terminal.
        print(f"DEBUG: El tipo del objeto 'response' es: {type(response)}")
        
        # 2. Imprimimos el contenido CRUDO de 'response' en la terminal.
        print(f"DEBUG: El contenido crudo de 'response' es: {response}")

        # 3. Devolvemos la respuesta como texto para verla en el navegador.
        #    Esto nos mostrará su estructura directamente en la pantalla.
        return {
            "mensaje": "Modo de diagnóstico. Revisa la terminal y la estructura de la respuesta a continuación.",
            "tipo_de_respuesta": str(type(response)),
            "contenido_crudo_respuesta": str(response)
        }

    except Exception as e:
        print(f"Ha ocurrido un error: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor al procesar el archivo: {e}")