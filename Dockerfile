# Base Python estável 2026
FROM python:3.13-slim

# Evita a geração de arquivos .pyc e garante log em tempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="${PYTHONPATH}:/app"

WORKDIR /app

# Instalação de dependências de sistema e limpeza de cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Criação de usuário não-privilegiado para segurança
RUN useradd -m appuser
USER appuser

# Instala dependências (usando cache de camada)
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copia o projeto com as permissões corretas
COPY --chown=appuser:appuser . .

# Expõe a porta do Streamlit
EXPOSE 8501

# Verificação de saúde da aplicação
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["python", "-m", "streamlit", "run", "production_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]