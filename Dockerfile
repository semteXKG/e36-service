FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir flask rank-bm25 openai python-dotenv
COPY app.py query.py index.json tags.json ./
EXPOSE 5000
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "5000"]
