FROM python:3.10-slim
WORKDIR /app
ENV PORT 8080

# 1. Salin hanya file requirements terlebih dahulu
COPY requirements.txt .

# 2. Instal dependensi (layer ini akan di-cache jika requirements.txt tidak berubah)
#    Tambahkan --no-cache-dir untuk mengurangi ukuran image
#    dan bersihkan cache pip setelah instalasi.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# 3. Salin sisa kode aplikasi
COPY . .

EXPOSE 8080
CMD ["python", "manage.py", "runserver", "0.0.0.0:8080"]