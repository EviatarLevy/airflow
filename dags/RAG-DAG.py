# dags/RAG_DAG.py
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonVirtualenvOperator

def _run_pipeline():
    # Everything below runs INSIDE a temporary virtualenv
    import os, requests
    from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    DATA_PATH = os.environ.get("DATA_PATH", "/home/eviatar-hpe.com/shared/leumi")
    EMBED_ENDPOINT = "https://nvidia-nv-embedqa-e5-v5-predictor-eviatar-hpe-com-a98a1262.ingress.pcai.hpelabs.co.il/v1"
    EMBED_TOKEN = "eyJhbGciOiJSUzI1NiIsImtpZCI6Inl0YnRpQXBOM3E1enkwRnNHUF82bUZBclFRTWx6T1RSZ0xpNkJzUExPWmcifQ.eyJhdWQiOlsiYXBpIiwiaXN0aW8tY2EiXSwiZXhwIjoxNzgzODY3MzkwLCJpYXQiOjE3NTc5NDczOTAsImlzcyI6Imh0dHBzOi8va3ViZXJuZXRlcy5kZWZhdWx0LnN2Yy5jbHVzdGVyLmxvY2FsIiwianRpIjoiMGM3NTBiZTAtMzZiZC00NzY1LWI1OGMtMjNmNmNlN2EyODM2Iiwia3ViZXJuZXRlcy5pbyI6eyJuYW1lc3BhY2UiOiJ1aSIsInNlcnZpY2VhY2NvdW50Ijp7Im5hbWUiOiJpc3ZjLWVwLTE3NTc5NDczOTAyODciLCJ1aWQiOiI2ODAzOTk5ZS0xMDY3LTRjZTktOWJhZC0yNDhkNjk2ZTMwYzIifX0sIm5iZiI6MTc1Nzk0NzM5MCwic3ViIjoic3lzdGVtOnNlcnZpY2VhY2NvdW50OnVpOmlzdmMtZXAtMTc1Nzk0NzM5MDI4NyJ9.BNjKIP1BM7r0UvqLDM_qV_o8jWrDg7S8u_Y6gVIcMYc1gznDqQrSaJndQMt1NED9UBGZk-0w8ZbGgicyrrTpqAHnUtThF4KNpYqKKWnDLZHrNZLVY7Z1ZxgiOUCvCV-lPwBsXKZpNoefxiISZGTDj6ec1HnX1AxCytPWsN9lf4kjIWrNHkRwTLIoeLtysGc0hQrwxbm-F0Z-SnZ3ZMfjcvUchuMyWObhiP2gM02PXBvbu5ttWje1J_WOEU5ov1UJRsWMNH3hPdAgn2kSR85P5s32bWm1fKyW43-Fkmy-V-ssiUYXjPZocaVlMIBeUIssI2espJtKVWDKtClOJ00i2g"
    MODEL = "nvidia/nv-embedqa-e5-v5"

    def load_documents(path):
        return DirectoryLoader(
            path,
            glob="**/*.*",
            loader_cls=UnstructuredFileLoader,
            loader_kwargs={"languages": ["eng", "heb"]},
            silent_errors=True,
        ).load()

    def chunk_text(docs, chunk_size=800, chunk_overlap=100):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_documents(docs)

    def embed(texts, batch_size=50):
        if isinstance(texts, str):
            texts = [texts]
        url = EMBED_ENDPOINT.rstrip("/") + "/embeddings"
        headers = {"Authorization": f"Bearer {EMBED_TOKEN}"}
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            r = requests.post(
                url,
                headers=headers,
                json={"model": MODEL, "input": batch, "input_type": "passage"},
                verify=False,  # internal CA in your env
                timeout=120,
            )
            r.raise_for_status()
            all_vecs.extend([d["embedding"] for d in r.json()["data"]])
            print(f"✅ Embedded {i+1}-{i+len(batch)}")
        return all_vecs

    docs = load_documents(DATA_PATH)
    chunks = chunk_text(docs)
    vecs = embed([c.page_content for c in chunks])
    print(f"Done: {len(docs)} docs → {len(chunks)} chunks → {len(vecs)} vectors")

with DAG(
    dag_id="RAG_DAG",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    run = PythonVirtualenvOperator(
        task_id="run_embedding_pipeline",
        python_callable=_run_pipeline,
        requirements=[
            "langchain",
            "langchain-community",
            "unstructured",
            "requests",
        ],
        system_site_packages=False,
    )
