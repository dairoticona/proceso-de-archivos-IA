import os
import json
from openai import OpenAI
from dotenv import load_dotenv

def procesar_texto_a_json(ruta_texto_entrada, ruta_json_salida):
    """
    Lee texto de un archivo, lo envía a la API de OpenAI para estructurarlo
    y guarda el resultado en un archivo JSON.
    """
    print("Iniciando el proceso con el modelo gpt-5-nano...")

    # 1. Cargar la clave de API desde el archivo .env
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: La variable de entorno OPENAI_API_KEY no se encontró.")
        print("Asegúrate de tener un archivo .env con tu clave.")
        return

    # Instanciar el cliente de OpenAI
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        print(f"Error al inicializar el cliente de OpenAI: {e}")
        return

    # 2. Leer el contenido del archivo de texto plano
    try:
        with open(ruta_texto_entrada, 'r', encoding='utf-8') as archivo_texto:
            texto_plano = archivo_texto.read()
        print(f"Texto leído con éxito de '{ruta_texto_entrada}'.")
    except FileNotFoundError:
        print(f"Error: El archivo de entrada '{ruta_texto_entrada}' no fue encontrado.")
        return

    # 3. Crear el prompt para la IA
    prompt_del_sistema = """
    Eres un asistente experto en análisis y estructuración de datos.
    Tu tarea es convertir el texto que te proporciono en un objeto JSON bien formado.
    Analiza el contenido del texto e identifica sus componentes lógicos
    (título, subtítulos, fechas, puntos clave, etc.) para crear una
    estructura JSON coherente.
    """
    
    prompt_del_usuario = f"""
    Por favor, convierte el siguiente texto a formato JSON.
    Texto a convertir:
    ---
    {texto_plano}
    ---
    """

    # 4. Llamar a la API de OpenAI
    print("Enviando petición a la API de OpenAI...")
    try:
        respuesta = client.chat.completions.create(
      
            # Aquí especificamos el modelo gpt-5-nano.
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": prompt_del_sistema},
                {"role": "user", "content": prompt_del_usuario}
            ],
            # Este parámetro es crucial para asegurar que la salida sea un JSON válido
            response_format={"type": "json_object"}
        )
        
        # Extraer el contenido de la respuesta, que será un string en formato JSON
        json_string = respuesta.choices[0].message.content
        print("Respuesta recibida de la API.")

    except Exception as e:
        print(f"Ocurrió un error al contactar la API de OpenAI: {e}")
        return

    # 5. Sobrescribir el archivo JSON con la nueva estructura
    try:
        # Convertir el string JSON a un objeto Python para poder guardarlo con formato
        datos_json = json.loads(json_string)

        with open(ruta_json_salida, 'w', encoding='utf-8') as archivo_json:
            # Usamos json.dump para escribir el objeto Python en el archivo JSON
            json.dump(datos_json, archivo_json, ensure_ascii=False, indent=4)
        
        print(f"¡Éxito! El archivo '{ruta_json_salida}' ha sido sobrescrito con el texto estructurado.")

    except json.JSONDecodeError:
        print("Error: La API no devolvió un JSON válido. Guardando la respuesta en bruto.")
        with open(ruta_json_salida, 'w', encoding='utf-8') as archivo_json:
            archivo_json.write(json_string)
    except Exception as e:
        print(f"Ocurrió un error al escribir el archivo JSON: {e}")


# --- Ejecutar el programa ---
if __name__ == "__main__":
    archivo_de_texto = "entrada.txt"
    archivo_de_json = "datos.json"
    
    procesar_texto_a_json(archivo_de_texto, archivo_de_json)