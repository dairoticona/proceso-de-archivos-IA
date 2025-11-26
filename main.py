import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Form # <--- AÑADIDO: Form
from pathlib import Path
from dotenv import load_dotenv
from mistralai import Mistral
import openai 
import io
from fastapi.responses import StreamingResponse

load_dotenv()
api_key = os.getenv('MISTRAL_API_KEY')

if not api_key:
    raise ValueError("No se encontró MISTRAL_API_KEY. Asegúrate de que tu archivo .env está configurado.")

client = Mistral(api_key=api_key)

# --- NUEVA CONFIGURACIÓN: CLIENTE OPENAI ---
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("No se encontró OPENAI_API_KEY. Asegúrate de que tu archivo .env está configurado.")

# Inicializa el cliente de OpenAI
openai_client = openai.OpenAI(api_key=openai_api_key)
# Define el modelo a utilizar. Reemplaza "gpt-4o" si prefieres otro como "gpt-4-turbo".
MODELO_OPENAI = "gpt-5-nano"


app = FastAPI(
    title="API de OCR con Mistral y análisis con OpenAI",
    description="Sube un PDF para diagnóstico, fusiona PDF con JSON, o analiza ambos con IA."
)


# --- ENDPOINT 1 (MODO DIAGNÓSTICO - SIN CAMBIOS) ---
@app.post("/mistral-ocr/")
async def extraer_texto_de_pdf(
    archivo: UploadFile = File(..., description="El archivo PDF que deseas procesar.")
):
    if archivo.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Error: El archivo proporcionado no es un PDF.")
    try:
        contenido_bytes = await archivo.read()
        nombre_base = Path(archivo.filename).stem
        print("Subiendo el archivo...")
        uploaded_file = client.files.upload(
            file={"file_name": nombre_base, "content": contenido_bytes}, purpose="ocr"
        )
        print("Obteniendo URL firmada...")
        signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
        documento_para_ocr = {"document_url": signed_url.url}
        print("Enviando a procesar con OCR...")
        response = client.ocr.process(
            document=documento_para_ocr, model='mistral-ocr-latest'
        )
        print("¡Respuesta recibida de Mistral!")
        print(f"DEBUG: El tipo del objeto 'response' es: {type(response)}")
        print(f"DEBUG: El contenido crudo de 'response' es: {response}")
        return {
            "mensaje": "Modo de diagnóstico.",
            "tipo_de_respuesta": str(type(response)),
            "contenido_crudo_respuesta": str(response)
        }
    except Exception as e:
        print(f"Ha ocurrido un error: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {e}")





# --- NUEVO ENDPOINT: ANÁLISIS CON IA (OPENAI) ---
@app.post("/probar-prompt-extractor/")
async def analizar_con_ia(
    pdf_file: UploadFile = File(..., description="El archivo PDF para extraer texto."),
    json_file: UploadFile = File(..., description="El archivo JSON base."),
    prompt: str = Form(..., description="Instrucciones para la IA sobre qué hacer con el texto y el JSON.")
):
    """
    Extrae texto de un PDF, lo combina con un JSON, lo procesa con OpenAI
    y devuelve el resultado como un nuevo archivo .json para descargar.
    """
    # 1. Validación de archivos (similar a los otros endpoints)
    if pdf_file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Error: El primer archivo debe ser un PDF.")
    if json_file.content_type != "application/json":
        raise HTTPException(status_code=400, detail="Error: El segundo archivo debe ser un JSON.")

    try:
        # 2. Cargar el JSON del usuario
        json_contenido_bytes = await json_file.read()
        datos_json_originales = json.loads(json_contenido_bytes)
        
        # 3. Extraer texto del PDF usando Mistral OCR (lógica ya existente)
        contenido_bytes_pdf = await pdf_file.read()
        nombre_base_pdf = Path(pdf_file.filename).stem
        
        print("Iniciando proceso de OCR con Mistral...")
        uploaded_file = client.files.upload(
            file={"file_name": nombre_base_pdf, "content": contenido_bytes_pdf}, purpose="ocr"
        )
        signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
        response_ocr = client.ocr.process(
            document={"document_url": signed_url.url}, model='mistral-ocr-latest'
        )
        
        paginas_texto = [page.markdown for page in response_ocr.pages]
        texto_completo_extraido = "\n\n--- Nueva Página ---\n\n".join(paginas_texto)
        print("OCR completado. Texto extraído.")

        # 4. Preparar la llamada a la API de OpenAI
        print(f"Enviando datos al modelo {MODELO_OPENAI}...")
        
        # Combinamos toda la información en un solo prompt para el modelo
        prompt_completo_usuario = prompt.replace(
            '__TEXT_TO_PROCESS__', texto_completo_extraido
        ).replace(
            '__JSON_SCHEMA__', json.dumps(datos_json_originales, indent=2)
        )

        # 5. Realizar la llamada a OpenAI
        response_openai = openai_client.chat.completions.create(
            model=MODELO_OPENAI,
            messages=[
                # El 'system' message sigue siendo útil para establecer el tono general
                {"role": "system", "content": "Eres un asistente de IA diseñado para seguir instrucciones y devolver datos en formatos estructurados. Tu única salida debe ser el código JSON solicitado."},
                {"role": "user", "content": prompt_completo_usuario}
            ],
            response_format={"type": "json_object"}
        )
        print("Respuesta recibida de OpenAI.")
        
        # 6. Parsear el resultado de OpenAI
        resultado_json_str = response_openai.choices[0].message.content

        print("--- INICIO DE LA RESPUESTA CRUDA DE OPENAI ---")
        print(resultado_json_str)
        print("--- FIN DE LA RESPUESTA CRUDA DE OPENAI ---")

        resultado_final = json.loads(resultado_json_str)
        # 7. Convertir el diccionario final a bytes
        json_bytes = json.dumps(resultado_final, indent=4, ensure_ascii=False).encode('utf-8')
        # 8. Crear un "archivo en memoria"
        stream = io.BytesIO(json_bytes)
        # 9. Definir el nombre del archivo de descarga
        pdf_stem_safe = Path(pdf_file.filename).stem.encode('ascii', 'ignore').decode('ascii')
        json_stem_safe = Path(json_file.filename).stem.encode('ascii', 'ignore').decode('ascii')
        download_filename = f"analisisIA_{json_stem_safe}_con_{pdf_stem_safe}.json"
        # 10. Crear las cabeceras para forzar la descarga
        headers = {
            'Content-Disposition': f'attachment; filename="{download_filename}"'
        }
        # 11. Devolver la StreamingResponse
        return StreamingResponse(content=stream, media_type="application/json", headers=headers)

    except Exception as e:
        print(f"Ha ocurrido un error: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {e}")