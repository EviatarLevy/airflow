from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="hello_demo",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["demo"],
) as dag:
    t1 = BashOperator(task_id="say_hi", bash_command='echo "Hello from PCAI Airflow!"')
