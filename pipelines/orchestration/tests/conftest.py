import os
import tempfile
from pathlib import Path

# Airflow reads its home and settings at import, so isolate them before any test imports it:
# the tests must never touch a contributor's own Airflow home or metadata database.
os.environ["AIRFLOW_HOME"] = tempfile.mkdtemp(prefix="agentwatch-airflow-test-")
os.environ["AIRFLOW__CORE__DAGS_FOLDER"] = str(Path(__file__).resolve().parents[1] / "dags")
os.environ["AIRFLOW__CORE__LOAD_EXAMPLES"] = "False"
