# Base Python estável
FROM python:3.13-slim

# Diretório de trabalho
WORKDIR /app

# Instalação de dependências de sistema para o XGBoost (OpenMP)
RUN apt-get update && apt-get install -y \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Instala as dependências do Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o projeto
COPY . .

# Define o PYTHONPATH para que o Python encontre o pacote 'src' na raiz
ENV PYTHONPATH="${PYTHONPATH}:/app"
# ────────────────────────────────────────────────────────────────────────

# Expõe a porta padrão do Streamlit
EXPOSE 8501

# Comando para rodar o app
CMD ["streamlit", "run", "production_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]