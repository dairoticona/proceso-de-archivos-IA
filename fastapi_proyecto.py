import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Form # <--- AÑADIDO: Form
from pathlib import Path
from dotenv import load_dotenv
from mistralai import Mistral
import openai 


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
@app.post("/extraer-texto-pdf/")
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


# --- ENDPOINT 2 (FUSIÓN DE PDF Y JSON - SIN CAMBIOS) ---
@app.post("/fusionar-pdf-con-json/")
async def fusionar_pdf_con_json(
    pdf_file: UploadFile = File(..., description="El archivo PDF para extraer texto."),
    json_file: UploadFile = File(..., description="El archivo JSON base al que se añadirá el texto.")
):
    if pdf_file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Error: El primer archivo debe ser un PDF.")
    if json_file.content_type != "application/json":
        raise HTTPException(status_code=400, detail="Error: El segundo archivo debe ser un JSON.")
    try:
        json_contenido_bytes = await json_file.read()
        datos_json = json.loads(json_contenido_bytes)
        if not isinstance(datos_json, dict):
            raise HTTPException(status_code=400, detail="Error: El contenido del JSON debe ser un objeto.")
        
        contenido_bytes_pdf = await pdf_file.read()
        nombre_base_pdf = Path(pdf_file.filename).stem
        print("Subiendo el archivo PDF...")
        uploaded_file = client.files.upload(
            file={"file_name": nombre_base_pdf, "content": contenido_bytes_pdf}, purpose="ocr"
        )
        print("Obteniendo URL firmada del PDF...")
        signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
        documento_para_ocr = {"document_url": signed_url.url}
        print("Enviando PDF a procesar con OCR...")
        response_ocr = client.ocr.process(
            document=documento_para_ocr, model='mistral-ocr-latest'
        )
        print("¡Respuesta de OCR recibida!")
        
        paginas_texto = [page.markdown for page in response_ocr.pages]
        texto_completo_extraido = "\n\n--- Nueva Página ---\n\n".join(paginas_texto)
        
        datos_json["texto_extraido_del_pdf"] = texto_completo_extraido
        return datos_json
    except Exception as e:
        print(f"Ha ocurrido un error: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {e}")


# --- NUEVO ENDPOINT: ANÁLISIS CON IA (OPENAI) ---
@app.post("/analizar-con-ia/")
async def analizar_con_ia(
    pdf_file: UploadFile = File(..., description="El archivo PDF para extraer texto."),
    json_file: UploadFile = File(..., description="El archivo JSON base."),
    prompt: str = Form(..., description="Instrucciones para la IA sobre qué hacer con el texto y el JSON.")
):
    """
    Extrae texto de un PDF, lo combina con un JSON y usa un modelo de OpenAI para
    procesar ambos según las instrucciones del prompt. Devuelve un nuevo JSON.
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
        prompt_completo_usuario = f"""
        Aquí están los datos que necesito que proceses:

        --- INICIO DEL TEXTO EXTRAÍDO DEL PDF ---
        {texto_completo_extraido}
        --- FIN DEL TEXTO EXTRAÍDO DEL PDF ---

        --- INICIO DEL DOCUMENTO JSON ORIGINAL ---
        {json.dumps(datos_json_originales, indent=2)}
        --- FIN DEL DOCUMENTO JSON ORIGINAL ---

        --- INSTRUCCIONES DEL USUARIO ---
        {prompt}
        --- FIN DE LAS INSTRUCCIONES ---
        """

        # 5. Realizar la llamada a OpenAI
        response_openai = openai_client.chat.completions.create(
            model=MODELO_OPENAI,
            messages=[
                {"role": "system", "content": "Eres un asistente experto en procesar documentos. Tu tarea es seguir las instrucciones del usuario para combinar, analizar o transformar los datos proporcionados (un texto de PDF y un JSON). Debes devolver SIEMPRE tu respuesta final en un formato JSON válido."},
                {"role": "user", "content": prompt_completo_usuario}
            ],
            # Forzamos al modelo a que la salida sea un objeto JSON válido
            response_format={"type": "json_object"}
        )

        print("Respuesta recibida de OpenAI.")
        
        # 6. Devolver el resultado
        # El contenido de la respuesta es un string JSON, lo parseamos a un diccionario
        resultado_json_str = response_openai.choices[0].message.content
        resultado_final = json.loads(resultado_json_str)
        
        return resultado_final

    except Exception as e:
        print(f"Ha ocurrido un error: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {e}")