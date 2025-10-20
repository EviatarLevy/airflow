# dags/Multimodal-RAG--AI-Day.py
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonVirtualenvOperator

with DAG(
    dag_id="Multimodal-RAG--AI-Day",
    start_date=datetime(2025, 1, 1),          # leave as-is; no backfill since catchup=False
    schedule="0 2 * * *",                     # every day at 02:00 (cron)
    catchup=False,                            # no historical runs
    tags=["Daily data ingestion on new files"]# << your requested tag
    # ,timezone=TZ                            # ← uncomment to run at 02:00 Israel time
) as dag:

    # ---- Task A: warm VLM (LLaVA) ----
    def _warm_vlm_no_fail():
        import json, requests, urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        VISION_ENDPOINT = "https://llava-hf-llava-1-5-7b-hf.eviatar-hpe-com-a98a1262.serving.ingress.pcai.hpelabs.co.il/v1"
        VISION_TOKEN    = "eyJhbGciOiJSUzI1NiIsImtpZCI6Inl0YnRpQXBOM3E1enkwRnNHUF82bUZBclFRTWx6T1RSZ0xpNkJzUExPWmcifQ.eyJhdWQiOlsiYXBpIiwiaXN0aW8tY2EiXSwiZXhwIjoxNzkxMDQ0MTczLCJpYXQiOjE3NTk1MDgxNzMsImlzcyI6Imh0dHBzOi8va3ViZXJuZXRlcy5kZWZhdWx0LnN2Yy5jbHVzdGVyLmxvY2FsIiwianRpIjoiMjMyODQyOGYtODZiYS00ZmUwLTg4Y2UtNjAyODI3YzAxYjdlIiwia3ViZXJuZXRlcy5pbyI6eyJuYW1lc3BhY2UiOiJ1aSIsInNlcnZpY2VhY2NvdW50Ijp7Im5hbWUiOiJpc3ZjLWVwLTE3NTk1MDgxNzMyMDEiLCJ1aWQiOiJhOGQ2OWRhYS1hOWVlLTQ5NmUtYTg3Zi1mNzY0NDI2MWJhZjgifX0sIm5iZiI6MTc1OTUwODE3Mywic3ViIjoic3lzdGVtOnNlcnZpY2VhY2NvdW50OnVpOmlzdmMtZXAtMTc1OTUwODE3MzIwMSJ9.YxD0QXC7Abtc8v-C2Zaeoo4jZBZGf0ZvKk-yw4yOoCl2jUeSAGkRB1_IXkzmpg7w4jTT0cKHe7hJBChlVFZXya57mhQBsMLVJgRW42RhtjWKHelrXBvPlmU6aLARGGsen1I8WxmOAmbIOXwyovg8zi3qL3xq2bjQyJxW3Im93ULukcS_SE2Dj9Q3S2zvN7uWItlXh38VYGIqgkLPwIMwnvdO_cPXCGIEracLYcL9MdKSVem3ShjfDKIqxi0_F10pT2ymQi7_Xs1gNAEDC4blOjJdodoGH7mIO3I4CqP6OBXFmACjyveuE0G4dceVxaxuDMiUT9-d3AJ4IZXL4uC_cg"
        VISION_MODEL    = "llava-hf/llava-1.5-7b-hf"

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

        url = VISION_ENDPOINT.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {VISION_TOKEN}", "Content-Type": "application/json"}
        payload = {
            "model": VISION_MODEL,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "ping"}]}],
            "max_tokens": 16,
            "temperature": 0.0,
            "chat_template": CHAT_TEMPLATE,
        }
        try:
            r = requests.post(url, headers=headers, json=payload, verify=False, timeout=30)
            if r.ok:
                print("[vlm] warm ping sent (OK)")
            else:
                print(f"[vlm] warm ping sent (HTTP {r.status_code})")
                try: print(json.dumps(r.json(), indent=2, ensure_ascii=False))
                except: print(r.text)
        except Exception as e:
            print(f"[vlm] exception during warm ping: {e}")
        print("[vlm] done (no-fail).")

    warm_vlm = PythonVirtualenvOperator(
        task_id="warm_vlm",
        python_callable=_warm_vlm_no_fail,
        requirements=["requests", "urllib3"],
        system_site_packages=False,
    )

    # ---- Task B: warm Embeddings ----
    def _warm_embed_no_fail():
        import json, requests, urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        EMBED_ENDPOINT = "https://nvidia-nv-embedqa-e5-v5.eviatar-hpe-com-a98a1262.serving.ingress.pcai.hpelabs.co.il/v1"
        EMBED_TOKEN    = "eyJhbGciOiJSUzI1NiIsImtpZCI6Inl0YnRpQXBOM3E1enkwRnNHUF82bUZBclFRTWx6T1RSZ0xpNkJzUExPWmcifQ.eyJhdWQiOlsiYXBpIiwiaXN0aW8tY2EiXSwiZXhwIjoxNzkxMDM2MTYwLCJpYXQiOjE3NTk1MDAxNjAsImlzcyI6Imh0dHBzOi8va3ViZXJuZXRlcy5kZWZhdWx0LnN2Yy5jbHVzdGVyLmxvY2FsIiwianRpIjoiMGU3OTFmYTAtNmI3NS00YWRjLTljNDMtNjFlNjg2MjhiZDY0Iiwia3ViZXJuZXRlcy5pbyI6eyJuYW1lc3BhY2UiOiJ1aSIsInNlcnZpY2VhY2NvdW50Ijp7Im5hbWUiOiJpc3ZjLWVwLTE3NTk1MDAxNjA4OTQiLCJ1aWQiOiIyNzQwMGZiOS1jY2VhLTQ4MjYtODdhMi0zN2EyNWEwM2JmNGYifX0sIm5iZiI6MTc1OTUwMDE2MCwic3ViIjoic3lzdGVtOnNlcnZpY2VhY2NvdW50OnVpOmlzdmMtZXAtMTc1OTUwMDE2MDg5NCJ9.c7ECXNzS8K5J50Dt1_sTy-arKjf3bRWjah6T1z7BEWU6m5Gfm4UPs1IW8TLOyHoHIwvwDO89d7HXIv3QLwDSkaSz-sxyNAxU2J8xW-NIBZ92nVvphFafWKAmSnO8irIEa8zvtZYv7RBhEY_72sGDXioB1l0RNPAxfbNPaNlt-zq7iuOgBxPZT_HOPHd5Zz5KYUITB_qCqSdte0KqjBeb9-YcidM303oWCSsm5SMWqxNbrcN-VJAcur87vnEfRuNIjaup8ERwnmXA7C8GtaYZHZ873OjlU8g3Ap0JIGwisdKZRkRVMM5Ul4dmhXKsxXJZSsNttwt7bEtXr111yvT73A"
        EMBED_MODEL    = "nvidia/nv-embedqa-e5-v5"

        url = EMBED_ENDPOINT.rstrip("/") + "/embeddings"
        headers = {"Authorization": f"Bearer {EMBED_TOKEN}", "Content-Type": "application/json"}
        payload = {"model": EMBED_MODEL, "input": ["שלום עולם", "Hello world"], "input_type": "passage"}

        try:
            r = requests.post(url, headers=headers, json=payload, verify=False, timeout=30)
            if r.ok:
                try:
                    dim = len(r.json()["data"][0]["embedding"])
                    print(f"[embed] warm ping OK (dim={dim})")
                except Exception:
                    print("[embed] warm ping OK (no parse)")
            else:
                print(f"[embed] warm ping sent (HTTP {r.status_code})")
                try: print(json.dumps(r.json(), indent=2, ensure_ascii=False))
                except: print(r.text)
        except Exception as e:
            print(f"[embed] exception during warm ping: {e}")
        print("[embed] done (no-fail).")

    warm_embed = PythonVirtualenvOperator(
        task_id="warm_embed",
        python_callable=_warm_embed_no_fail,
        requirements=["requests", "urllib3"],
        system_site_packages=False,
    )

    # Run BOTH in parallel (no dependencies)
    # If you want a final “done” task, you can add an EmptyOperator and set [warm_vlm, warm_embed] >> done
