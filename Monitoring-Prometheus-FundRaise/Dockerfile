FROM prom/prometheus:latest

# Menyalin berkas konfigurasi asli untuk digunakan sebagai templat
COPY prometheus.yml /etc/prometheus/prometheus_template.yml

# Mengganti titik masuk bawaan agar dapat mengeksekusi perintah shell
ENTRYPOINT [ "/bin/sh", "-c" ]

# Mengganti variabel dengan utilitas sed, menyimpan hasilnya ke direktori sementara, lalu menjalankan Prometheus
CMD [ "sed -e \"s|\\${GRAFANA_USERNAME}|$GRAFANA_USERNAME|g\" -e \"s|\\${GRAFANA_PASSWORD}|$GRAFANA_PASSWORD|g\" /etc/prometheus/prometheus_template.yml > /tmp/prometheus.yml && /bin/prometheus --config.file=/tmp/prometheus.yml --web.listen-address=0.0.0.0:7860" ]