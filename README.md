# PawPatrol IA API (`ia-api`)

API de inteligencia artificial para **reconocimiento visual de mascotas perdidas** por similitud de imágenes.

Consumida por la aplicación Next.js **PawPatrol**. No incluye frontend ni Firebase: solo analiza imágenes (URL o archivo) y persiste embeddings en **Neon PostgreSQL**.

## Stack

| Componente | Uso |
|------------|-----|
| **FastAPI** | API HTTP |
| **YOLOv8** | Detección de perros y gatos |
| **OpenAI CLIP** (open_clip) | Embeddings visuales (pelaje, forma, raza, cara, manchas) |
| **FAISS** | Búsqueda vectorial en memoria |
| **Neon PostgreSQL** | Metadata + vectores (JSONB) |
| **Pillow / Torch** | Procesamiento de imágenes |

## Estructura

```
ia-api/
├── app/
│   ├── main.py           # FastAPI + CORS + lifespan
│   ├── config.py         # Variables de entorno
│   ├── api/
│   │   ├── routes.py     # GET /, POST /register, POST /search
│   │   ├── schemas.py
│   │   └── errors.py
│   ├── db/
│   │   ├── connection.py
│   │   └── repository.py
│   ├── services/
│   │   ├── detection.py  # YOLOv8
│   │   ├── embeddings.py # CLIP
│   │   ├── search.py     # FAISS
│   │   ├── pipeline.py
│   │   └── registration.py
│   └── utils/
│       ├── image_loader.py
│       └── logging_config.py
├── requirements.txt
├── Dockerfile
├── render.yaml
└── run.py
```

## Variables de entorno

Copia `.env.example` a `.env`:

```env
DATABASE_URL=postgresql://user:pass@ep-xxx.neon.tech/neondb?sslmode=require
CORS_ORIGINS=http://localhost:3000,https://tu-dominio.com
LOG_LEVEL=INFO
# API_KEY=opcional-para-produccion
```

## Instalación local

```bash
cd ia-api
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env
# Editar DATABASE_URL

python run.py
```

API en `http://localhost:8000` — documentación interactiva en `/docs`.

> **Nota:** La primera petición descarga YOLOv8 y CLIP (puede tardar varios minutos).

## Endpoints

### `GET /`

Health check.

```json
{
  "service": "PawPatrol IA API",
  "status": "ok",
  "registered_pets": 12,
  "faiss_vectors": 12
}
```

### `POST /register`

Registra una mascota de PawPatrol en el índice visual.

**JSON** (`Content-Type: application/json`):

```json
{
  "pet_id": "id-mascota-pawpatrol",
  "pet_name": "Max",
  "location": "Juliaca",
  "image_url": "https://tu-cdn.com/fotos/max.jpg"
}
```

O con base64:

```json
{
  "pet_id": "id-mascota-pawpatroll",
  "pet_name": "Max",
  "location": "Juliaca",
  "image_base64": "..."
}
```

**Multipart:** `pet_id`, `pet_name`, `location` (opcional), `image_url` o archivo `image`.

### `POST /search`

Busca las **5 mascotas más similares** (configurable con `SEARCH_TOP_K`).

**JSON** (`image_url` **o** `image_base64`; recomendado desde Next.js servidor):

```json
{
  "image_base64": "<base64 sin prefijo data: o data:image/jpeg;base64,...>",
  "exclude_pet_id": "opcional-id-a-excluir"
}
```

Alternativa con URL pública:

```json
{
  "image_url": "https://ejemplo.com/mascota-encontrada.jpg",
  "exclude_pet_id": "opcional-id-a-excluir"
}
```

**Multipart:** `image_url` y/o archivo `image`.

**Respuesta** (detección de la consulta + ranking por similitud):

```json
{
  "detection": {
    "detected_class": "dog",
    "confidence": 91.5
  },
  "matches": [
    {
      "pet_id": "uuid-mascota",
      "pet_name": "Max",
      "similarity": 94,
      "location": "Juliaca",
      "image_url": "https://..."
    }
  ]
}
```

## Integración con Next.js (PawPatrol)

Al crear o actualizar una ficha con foto principal:

```ts
await fetch(`${process.env.IA_API_URL}/register`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": process.env.IA_API_KEY ?? "",
  },
  body: JSON.stringify({
    pet_id: mascota.id,
    pet_name: mascota.nombre,
    location: usuario.ciudad ?? "",
    image_base64: bufferBase64,
  }),
});
```

Al reportar una mascota encontrada:

```ts
const matches = await fetch(`${process.env.IA_API_URL}/search`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": process.env.IA_API_KEY ?? "",
  },
  body: JSON.stringify({ image_base64: bufferBase64 }),
}).then((r) => r.json());
```

## Despliegue en Render

1. Crea un **Web Service** con Docker o Python.
2. Configura `DATABASE_URL` (misma Neon o base dedicada para embeddings).
3. Añade `CORS_ORIGINS` con la URL de PawPatrol.
4. (Opcional) Define `API_KEY` en Render y la misma en PawPatrol como `IA_API_KEY`. Si no la configuras, la API acepta peticiones sin clave.
5. Usa plan con suficiente RAM (≥ 2 GB recomendado por Torch + modelos).

El archivo `render.yaml` incluye una plantilla básica.

## Base de datos

Tabla creada automáticamente al iniciar:

- `pet_embeddings`: `id`, `pet_name`, `location`, `image_url`, `embedding` (JSONB), `detected_class`, `bbox`, timestamps.

El índice FAISS se reconstruye desde Neon al arrancar y se actualiza en cada registro.

## Errores habituales

| Código | Significado |
|--------|-------------|
| 422 | No se detectó perro/gato en la imagen |
| 400 | URL inválida o imagen corrupta |
| 401 | API key incorrecta (si `API_KEY` está configurada) |
