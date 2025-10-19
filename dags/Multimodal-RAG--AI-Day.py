# dags/test_models_trigger_dag.py
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonVirtualenvOperator

# === YOUR CONFIG (kept exactly as provided) ===
EMBED_ENDPOINT = "https://nvidia-nv-embedqa-e5-v5.eviatar-hpe-com-a98a1262.serving.ingress.pcai.hpelabs.co.il/v1"
EMBED_TOKEN = "eyJhbGciOiJSUzI1NiIsImtpZCI6Inl0YnRpQXBOM3E1enkwRnNHUF82bUZBclFRTWx6T1RSZ0xpNkJzUExPWmcifQ.eyJhdWQiOlsiYXBpIiwiaXN0aW8tY2EiXSwiZXhwIjoxNzkxMDM2MTYwLCJpYXQiOjE3NTk1MDAxNjAsImlzcyI6Imh0dHBzOi8va3ViZXJuZXRlcy5kZWZhdWx0LnN2Yy5jbHVzdGVyLmxvY2FsIiwianRpIjoiMGU3OTFmYTAtNmI3NS00YWRjLTljNDMtNjFlNjg2MjhiZDY0Iiwia3ViZXJuZXRlcy5pbyI6eyJuYW1lc3BhY2UiOiJ1aSIsInNlcnZpY2VhY2NvdW50Ijp7Im5hbWUiOiJpc3ZjLWVwLTE3NTk1MDAxNjA4OTQiLCJ1aWQiOiIyNzQwMGZiOS1jY2VhLTQ4MjYtODdhMi0zN2EyNWEwM2JmNGYifX0sIm5iZiI6MTc1OTUwMDE2MCwic3ViIjoic3lzdGVtOnNlcnZpY2VhY2NvdW50OnVpOmlzdmMtZXAtMTc1OTUwMDE2MDg5NCJ9.c7ECXNzS8K5J50Dt1_sTy-arKjf3bRWjah6T1z7BEWU6m5Gfm4UPs1IW8TLOyHoHIwvwDO89d7HXIv3QLwDSkaSz-sxyNAxU2J8xW-NIBZ92nVvphFafWKAmSnO8irIEa8zvtZYv7RBhEY_72sGDXioB1l0RNPAxfbNPaNlt-zq7iuOgBxPZT_HOPHd5Zz5KYUITB_qCqSdte0KqjBeb9-YcidM303oWCSsm5SMWqxNbrcN-VJAcur87vnEfRuNIjaup8ERwnmXA7C8GtaYZHZ873OjlU8g3Ap0JIGwisdKZRkRVMM5Ul4dmhXKsxXJZSsNttwt7bEtXr111yvT73A"
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"

VISION_ENDPOINT = "https://llava-hf-llava-1-5-7b-hf.eviatar-hpe-com-a98a1262.serving.ingress.pcai.hpelabs.co.il/v1"
VISION_TOKEN = "eyJhbGciOiJSUzI1NiIsImtpZCI6Inl0YnRpQXBOM3E1enkwRnNHUF82bUZBclFRTWx6T1RSZ0xpNkJzUExPWmcifQ.eyJhdWQiOlsiYXBpIiwiaXN0aW8tY2EiXSwiZXhwIjoxNzkxMDQ0MTczLCJpYXQiOjE3NTk1MDgxNzMsImlzcyI6Imh0dHBzOi8va3ViZXJuZXRlcy5kZWZhdWx0LnN2Yy5jbHVzdGVyLmxvY2FsIiwianRpIjoiMjMyODQyOGYtODZiYS00ZmUwLTg4Y2UtNjAyODI3YzAxYjdlIiwia3ViZXJuZXRlcy5pbyI6eyJuYW1lc3BhY2UiOiJ1aSIsInNlcnZpY2VhY2NvdW50Ijp7Im5hbWUiOiJpc3ZjLWVwLTE3NTk1MDgxNzMyMDEiLCJ1aWQiOiJhOGQ2OWRhYS1hOWVlLTQ5NmUtYTg3Zi1mNzY0NDI2MWJhZjgifX0sIm5iZiI6MTc1OTUwODE3Mywic3ViIjoic3lzdGVtOnNlcnZpY2VhY2NvdW50OnVpOmlzdmMtZXAtMTc1OTUwODE3MzIwMSJ9.YxD0QXC7Abtc8v-C2Zaeoo4jZBZGf0ZvKk-yw4yOoCl2jUeSAGkRB1_IXkzmpg7w4jTT0cKHe7hJBChlVFZXya57mhQBsMLVJgRW42RhtjWKHelrXBvPlmU6aLARGGsen1I8WxmOAmbIOXwyovg8zi3qL3xq2bjQyJxW3Im93ULukcS_SE2Dj9Q3S2zvN7uWItlXh38VYGIqgkLPwIMwnvdO_cPXCGIEracLYcL9MdKSVem3ShjfDKIqxi0_F10pT2ymQi7_Xs1gNAEDC4blOjJdodoGH7mIO3I4CqP6OBXFmACjyveuE0G4dceVxaxuDMiUT9-d3AJ4IZXL4uC_cg"
VISION_MODEL = "llava-hf/llava-1.5-7b-hf"  # text-only attempt

# Minimal Jinja chat template (Transformers >= 4.44 requires explicit template)
CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m['role']=='user' %}USER: "
    "{% if m['content'] is string %}{{ m['content'] }}"
    "{% else %}{% for p in m['content'] %}"
    "{% if p['type']=='text' %}{{ p['text'] }}{% endif %}"
    "{% endfor %}{% endif %}\n"
    "{% elif m['role']=='assistant' %}ASSISTANT: {{ m['content'] }}\n{% endif %}"
    "{% endfor %}ASSISTANT:"
)

def _run_models():
    # Import third-party libs *inside* the callable so Airflow can parse the DAG
    import requests, json, urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def post_or_print(url, headers, payload, timeout=120):
        r = requests.post(url, headers=headers, json=payload, verify=False, timeout=timeout)
        if not r.ok:
            print(f"---- ERROR {r.status_code} for {url} ----")
            try:
                print(json.dumps(r.json(), indent=2, ensure_ascii=False))
            except Exception:
                print(r.text)
            print("---- END ERROR BODY ----")
            r.raise_for_status()
        return r.json()

    # --- 1) Embeddings test ---
    embed_url = EMBED_ENDPOINT.rstrip("/") + "/embeddings"
    embed_headers = {
        "Authorization": f"Bearer {EMBED_TOKEN}",
        "Content-Type": "application/json",
    }
    embed_payload = {
        "model": EMBED_MODEL,
        "input": ["שלום עולם", "Hello world"],
        "input_type": "passage",
    }
    print(f"→ Embeddings POST {embed_url}")
    embed_resp = post_or_print(embed_url, embed_headers, embed_payload, timeout=60)
    vecs = [d["embedding"] for d in embed_resp["data"]]
    print(f"✅ Embeddings OK (dim={len(vecs[0])})")

    # --- 2) Vision (text-only) test with explicit chat_template ---
    vision_url = VISION_ENDPOINT.rstrip("/") + "/chat/completions"
    vision_headers = {
        "Authorization": f"Bearer {VISION_TOKEN}",
        "Content-Type": "application/json",
    }
    vision_payload = {
        "model": VISION_MODEL,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Explain multimodal RAG in one short sentence."}]}
        ],
        "max_tokens": 64,
        "temperature": 0.2,
        "chat_template": CHAT_TEMPLATE,
    }
    print(f"→ Vision (text-only) POST {vision_url}")
    vision_resp = post_or_print(vision_url, vision_headers, vision_payload, timeout=120)
    answer = vision_resp["choices"][0]["message"]["content"]
    print("🧠 Vision reply:\n", answer)

with DAG(
    dag_id="test_models_trigger_dag",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    trigger_models = PythonVirtualenvOperator(
        task_id="trigger_models",
        python_callable=_run_models,
        requirements=[
            "requests",   # HTTP client
            "urllib3",    # to mute SSL warnings (pulled by requests, but explicit for clarity)
        ],
        system_site_packages=False,
    )
