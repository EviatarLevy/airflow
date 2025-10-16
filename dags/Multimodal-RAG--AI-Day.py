# dags/RAG-DAG.py
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonVirtualenvOperator
import os, requests, base64, mimetypes
from PIL import Image

# === CONFIG ===
EMBED_ENDPOINT = "https://nvidia-nv-embedqa-e5-v5-predictor-eviatar-hpe-com-a98a1262.ingress.pcai.hpelabs.co.il/v1"
EMBED_TOKEN = "<YOUR_TOKEN>"
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"

VISION_ENDPOINT = "https://nvidia-nv-llava-onevision-qwen2-72b-instruct-hf-predictor-eviatar-hpe-com-a98a1262.ingress.pcai.hpelabs.co.il/v1"
VISION_TOKEN = "<YOUR_TOKEN>"
VISION_MODEL = "nvidia/nv-llava-onevision-qwen2-72b-instruct-hf"

DATA_PATH = "/usr/local/airflow/dags/gitdags/dags/data"

# === PIPELINE ===
def _trigger_models():
    import requests, base64, mimetypes, os
    from PIL import Image

    # --- 1️⃣ Text Embedding ---
    def embed(texts, batch_size=50):
        if isinstance(texts, str):
            texts = [texts]
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            r = requests.post(
                f"{EMBED_ENDPOINT}/embeddings",
                headers={"Authorization": f"Bearer {EMBED_TOKEN}"},
                json={"model": EMBED_MODEL, "input": batch, "input_type": "passage"},
                verify=False,
            )
            r.raise_for_status()
            data = r.json()["data"]
            all_vecs.extend([d["embedding"] for d in data])
            print(f"✅ Embedded {i+1}-{i+len(batch)}")
        return all_vecs[0] if len(all_vecs) == 1 else all_vecs

    text_vec = embed("שלום עולם / Hello world")
    print(f"Embedded text length: {len(text_vec)} | Preview: {text_vec[:8]}")

    # --- 2️⃣ Image → Text ---
    def img2text(image_path, prompt="Describe this image in detail."):
        mime = mimetypes.guess_type(image_path)[0] or "image/png"
        data_url = "data:%s;base64,%s" % (mime, base64.b64encode(open(image_path, "rb").read()).decode("utf-8"))
        payload = {
            "model": VISION_MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
            ]}],
        }
        r = requests.post(
            f"{VISION_ENDPOINT}/chat/completions",
            headers={"Authorization": f"Bearer {VISION_TOKEN}", "Content-Type": "application/json"},
            json=payload, verify=False, timeout=(10, 120)
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    single_image_path = os.path.join(DATA_PATH, "Reports", "IDC_report.png")
    if os.path.exists(single_image_path):
        text = img2text(
            single_image_path,
            "Read and transcribe all visible on-screen text verbatim, preserving order top-to-bottom, right-to-left. Include titles, axis labels, legends, annotations, footers, watermarks, and company names. Mention HPE not HP.",
        )
        print("🖼️ Image transcription:\n", text)
    else:
        print(f"⚠️ Image not found at: {single_image_path}")


# === DAG DEFINITION ===
with DAG(
    dag_id="trigger_models_dag",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    run = PythonVirtualenvOperator(
        task_id="trigger_models",
        python_callable=_trigger_models,
        requirements=["requests", "pillow"],
        system_site_packages=False,
    )
