FROM python:3.11-slim

WORKDIR /app
COPY MLProject/ .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install mlflow

ENV MLFLOW_TRACKING_URI=https://dagshub.com/Saepulloh07/Eksperimen_SML_Saepulloh

ENTRYPOINT ["mlflow", "run", "."]