# dags/embed_pipeline.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import requests, os, datetime as dt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader


# ---------- CONFIG ----------
EMBED_ENDPOINT = "https://nvidia-nv-embedqa-e5-v5-predictor-eviatar-hpe-com-a98a1262.ingress.pcai.hpelabs.co.il/v1"
EMBED_TOKEN = "eyJhbGciOiJSUzI1NiIsImtpZCI6Inl0YnRpQXBOM3E1enkwRnNHUF82bUZBclFRTWx6T1RSZ0xpNkJzUExPWmcifQ.eyJhdWQiOlsiYXBpIiwiaXN0aW8tY2EiXSwiZXhwIjoxNzgzODY3MzkwLCJpYXQiOjE3NTc5NDczOTAsImlzcyI6Imh0dHBzOi8va3ViZXJuZXRlcy5kZWZhdWx0LnN2Yy5jbHVzdGVyLmxvY2FsIiwianRpIjoiMGM3NTBiZTAtMzZiZC00NzY1LWI1OGMtMjNmNmNlN2EyODM2Iiwia3ViZXJuZXRlcy5pbyI6eyJuYW1lc3BhY2UiOiJ1aSIsInNlcnZpY2VhY2NvdW50Ijp7Im5hbWUiOiJpc3ZjLWVwLTE3NTc5NDczOTAyODciLCJ1aWQiOiI2ODAzOTk5ZS0xMDY3LTRjZTktOWJhZC0yNDhkNjk2ZTMwYzIifX0sIm5iZiI6MTc1Nzk0NzM5MCwic3ViIjoic3lzdGVtOnNlcnZpY2VhY2NvdW50OnVpOmlzdmMtZXAtMTc1Nzk0NzM5MDI4NyJ9.BNjKIP1BM7r0UvqLDM_qV_o8jWrDg7S8u_Y6gVIcMYc1gznDqQrSaJndQMt1NED9UBGZk-0w8ZbGgicyrrTpqAHnUtThF4KNpYqKKWnDLZHrNZLVY7Z1ZxgiOUCvCV-lPwBsXKZpNoefxiISZGTDj6ec1HnX1AxCytPWsN9lf4kjIWrNHkRwTLIoeLtysGc0hQrwxbm-F0Z-SnZ3ZMfjcvUchuMyWObhiP2gM02PXBvbu5ttWje1J_WOEU5ov1UJRsWMNH3hPdAgn2kSR85P5s32bWm1fKyW43-Fkmy-V-ssiUYXjPZocaVlMIBeUIssI2espJtKVWDKtClOJ00i2g"
EMBD_MODEL = "nvidia/nv-embedqa-e5-v5"
DATA_PATH = "data"

# ---------- HELPERS ----------
def load_documents(path=DATA_PATH):
    return DirectoryLoader(
        path,
        glob="**/*.*",
        loader_cls=UnstructuredFileLoader,
        loader_kwargs={"languages": ["eng", "heb"]},
        silent_errors=True
    ).load()

def chunk_text(docs):
    splitter = RecursiveCharacterTextSplitter()
    return splitter.split_documents(docs)

def embed(texts):
    """Batch call to embedding endpoint"""
    url = EMBED_ENDPOINT.rstrip("/") + "/embeddings"
    headers = {"Authorization": f"Bearer {EMBED_TOKEN}"}
    r = requests.post(
        url,
        headers=headers,
        json={"model": EMBD_MODEL, "input": texts, "input_type": "passage"},
        verify=False,
    )
    r.raise_for_status()
    return [d["embedding"] for d in r.json()["data"]]

def run_pipeline():
    docs = load_documents()
    chunks = chunk_text(docs)
    print(f"Loaded {len(docs)} docs → {len(chunks)} chunks")

    # embed in batches
    batch_size = 50
    vectors = []
    for i in range(0, len(chunks), batch_size):
        batch = [c.page_content for c in chunks[i:i+batch_size]]
        vectors.extend(embed(batch))
        print(f"✅ Embedded chunks {i+1}-{min(i+batch_size, len(chunks))}")

    print("Done, total vectors:", len(vectors))

# ---------- DAG ----------
with DAG(
    dag_id="embedding_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
) as dag:

    run_task = PythonOperator(
        task_id="run_embedding_pipeline",
        python_callable=run_pipeline,
    )
