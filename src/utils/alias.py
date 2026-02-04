from pathlib import Path, PurePath
import json

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent

project_root: Path = get_project_root()

def read_config() -> dict:
    with open(project_root / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)
    
config = read_config()

def a2p(path: str) -> Path:
    """
        alias to path 將帶有別名的路徑轉換為實際路徑
        Args:
            path (str): 帶有別名的路徑
        Returns:
            Path: 轉換後的實際路徑
    """
    alias_config = config.get("alias", {})
    for alias, real_path in alias_config.items():
        if path.startswith(f"{alias}/"):
            return Path(real_path).joinpath(PurePath(path[len(alias)+1:]))
    raise ValueError(f"Path '{path}' does not start with any known alias.")