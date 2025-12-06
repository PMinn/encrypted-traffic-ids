from pathlib import Path, PurePath
import json

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent

project_root: Path = get_project_root()
def get_alias() -> dict:
    """
        定義路徑別名
        Args:
            None
        Returns:
            dict: 路徑別名字典
    """
    with open(project_root / "config" / "alias.json", "r") as f:
        alias = json.load(f)
    for key, value in alias.items():
        if value.startswith("/"):
            value = value[1:]
        alias[key] = project_root.joinpath(PurePath(value))
    return alias

alias: dict = get_alias()

def a2p(path: str) -> Path:
    """
        alias to path 將帶有別名的路徑轉換為實際路徑
        Args:
            path (str): 帶有別名的路徑
        Returns:
            Path: 轉換後的實際路徑
    """
    for key, value in alias.items():
        if path.startswith(key):
            return value.joinpath(PurePath(path[len(key):]))
    raise ValueError(f"Path '{path}' does not start with any known alias.")