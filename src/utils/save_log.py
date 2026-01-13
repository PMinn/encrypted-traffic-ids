from pathlib import Path
import mlflow
import numpy.typing as npt
import numpy as np
from utils.json import JSONObject


def log_json_artifact(obj: JSONObject, artifact_name: str) -> None:
    import json

    tmp = Path(f"/tmp/{artifact_name}")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    mlflow.log_artifact(str(tmp))


def log_npz_artifact(
    artifact_name: str, **kwargs: npt.NDArray[np.float64 | np.float32]
) -> None:
    import numpy as np

    tmp = Path(f"/tmp/{artifact_name}")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(tmp, allow_pickle=True, **kwargs)
    mlflow.log_artifact(str(tmp))
