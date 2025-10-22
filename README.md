# OcrApi

API REST para reconocimiento óptico de caracteres (OCR) basada en **FastAPI** y **PaddleOCR**.
Proporciona un servicio robusto para extraer texto de imágenes almacenadas localmente,
con pipeline de preprocesamiento avanzado, control de concurrencia y logging con rotación diaria.

## Características principales

- **Endpoints REST**
  - `GET /health`: comprueba el estado del servicio e incluye métricas opcionales de CPU/RAM.
  - `POST /ocr`: recibe un `image_path` y devuelve el texto reconocido junto con el tiempo invertido.
- **Preprocesamiento de imágenes** con OpenCV: deskew, normalización de contraste, reducción de ruido y binarización adaptativa.
- **Motor OCR modular** implementado con PaddleOCR siguiendo el patrón estrategia.
- **Control de concurrencia** mediante semáforo asíncrono que limita las peticiones OCR simultáneas.
- **Logging centralizado** con rotación diaria y niveles configurables (INFO/WARNING/ERROR).
- **Despliegue amigable en Windows**, incluyendo scripts para crear la tarea programada que levanta la API al iniciar el sistema.

## Requisitos

- Python 3.10 o superior.
- Dependencias listadas en `requirements.txt`.
- Modelos de PaddleOCR descargados automáticamente en el primer arranque.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copia `.env.example` a `.env` y ajusta los parámetros necesarios (rutas permitidas, nivel de logs, etc.).

## Ejecución en desarrollo

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

La API estará disponible en `http://localhost:8000`. Puedes probar los endpoints con herramientas como curl o
la documentación interactiva en `http://localhost:8000/docs`.

## Configuración destacada (`.env`)

| Variable | Descripción | Valor por defecto |
| --- | --- | --- |
| `MAX_CONCURRENT_REQUESTS` | Límite de peticiones OCR simultáneas. | `3` |
| `MAX_IMAGE_SIZE_MB` | Tamaño máximo permitido por imagen. | `10` |
| `ALLOWED_IMAGE_DIRS` | Lista separada por `;` de directorios autorizados para leer imágenes. | *(sin restricción)* |
| `MAX_IMAGE_DIMENSION` | Dimensión máxima (ancho/alto) antes de escalar la imagen. | `2500` |
| `LOG_LEVEL` | Nivel de logging (`DEBUG`, `INFO`, `WARNING`, `ERROR`). | `INFO` |
| `OCR_LANGUAGE` | Idioma del modelo PaddleOCR (`es`, `en`, `latin`, etc.). | `es` |

## Scripts para Windows

- `scripts/start_api.bat`: inicia la API usando Uvicorn. Acepta como argumento opcional la ruta al ejecutable de Python.
- `scripts/create_task.bat`: registra una tarea programada que ejecuta `start_api.bat` al iniciar Windows.

Ejemplo:

```bat
scripts\create_task.bat "C:\\Python311\\python.exe" "C:\\ruta\\al\\proyecto"
```

La tarea creada corre con privilegios de `SYSTEM`, se ejecuta al iniciar el equipo y se reinicia automáticamente en caso de falla.

## Flujo del endpoint `/ocr`

1. Validación del `image_path`: existencia, tamaño máximo y pertenencia a directorios autorizados.
2. Pipeline de preprocesamiento (deskew, normalización, denoise, binarización).
3. Ejecución del motor PaddleOCR reutilizando el modelo en memoria.
4. Respuesta JSON con el texto extraído y la métrica de tiempo.

Los errores controlados se devuelven como HTTP 400 (problemas de entrada) u 500 (fallos internos),
siempre registrados en el log para su trazabilidad.

## Manejo de logs

Los logs se escriben en consola y en `logs/ocr_api.log`, rotando diariamente y conservando
el número de archivos configurado en `LOG_BACKUP_COUNT`. Cada petición registra:
- Inicio y fin del procesamiento OCR.
- Duración total y longitud aproximada del texto devuelto.
- Errores o advertencias durante validación, preprocesamiento u OCR.

## Salud y monitoreo

El endpoint `GET /health` responde con `status`, `timestamp`, `version` y, si `INCLUDE_RESOURCE_METRICS=true`,
incorpora el porcentaje de uso de CPU y memoria usando `psutil`.

## Tests

Actualmente no se incluyen pruebas automáticas debido a la dependencia directa de PaddleOCR y procesamiento de imágenes.
Se recomienda preparar un set de imágenes de ejemplo y ejecutar solicitudes manuales para validar el pipeline completo.
