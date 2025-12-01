FROM python:3.10-slim

# Dependencias base para compilacion ligera y OpenMP
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libgomp1 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8050

WORKDIR /app

# Instala dependencias primero para aprovechar capa de cache
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
      torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
 && pip install --no-cache-dir -r requirements.txt

# Copia el resto del proyecto
COPY . .

EXPOSE 8050

# Usa la variable PORT si la plataforma la inyecta (Render, Heroku, etc.)
CMD ["python", "app.py"]
