FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install requests && python -c "import urllib.request; urllib.request.urlretrieve(\"https://raw.githubusercontent.com/tabler/tabler-icons/v3.19.0/packages/icons-webfont/dist/fonts/tabler-icons.woff2\", \"static/fonts/tabler-icons.woff2\"); urllib.request.urlretrieve(\"https://raw.githubusercontent.com/tabler/tabler-icons/v3.19.0/packages/icons-webfont/dist/fonts/tabler-icons.woff\", \"static/fonts/tabler-icons.woff\"); urllib.request.urlretrieve(\"https://raw.githubusercontent.com/tabler/tabler-icons/v3.19.0/packages/icons-webfont/dist/fonts/tabler-icons.ttf\", \"static/fonts/tabler-icons.ttf\")"
RUN mkdir -p model
RUN mkdir -p /data
EXPOSE 7860
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "300", "--workers", "1"]
