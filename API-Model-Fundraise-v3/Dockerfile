# Gunakan image Python
FROM python:3.10-slim

# Set working directory di dalam container
WORKDIR /code

# Copy file requirements dan install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy seluruh file ML dan app.py ke dalam container
COPY . /code

# Port untuk hugging face
EXPOSE 7860

# Perintah untuk menjalankan Uvicorn di port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]