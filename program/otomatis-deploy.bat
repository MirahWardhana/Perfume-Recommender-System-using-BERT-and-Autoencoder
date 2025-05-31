@echo off

set GOOGLE_CLOUD_PROJECT=ta-mirah
set PROJECT_NAME=django-skripsi-mirah

CALL gcloud config^
    set project %GOOGLE_CLOUD_PROJECT%
CALL gcloud builds submit^
    --tag asia-southeast2-docker.pkg.dev/ta-mirah/mirah-gcr-io/%PROJECT_NAME%
CALL gcloud run deploy^
    %PROJECT_NAME%^
    --image asia-southeast2-docker.pkg.dev/ta-mirah/mirah-gcr-io/%PROJECT_NAME%^
    --platform=managed^
    --region=asia-southeast2^
    --allow-unauthenticated^
    --max-instances=1^
    --cpu-boost^
    --cpu=8^
    --memory=8Gi