import json
from pathlib import Path
import mlflow


def log_json_artifact(obj: dict[str, object], artifact_name: str) -> None:
    tmp = Path(f"/tmp/{artifact_name}")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    mlflow.log_artifact(str(tmp))
