FROM python:3.13-slim
WORKDIR /app
COPY requirements_serve.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements_serve.txt
COPY src ./src
COPY models ./models
COPY data ./data
ENV LLM_BASE_URL=http://host.docker.internal:11434/v1
EXPOSE 8000
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]