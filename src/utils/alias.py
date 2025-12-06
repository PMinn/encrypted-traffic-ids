from pathlib import Path, PurePath

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent

project_root: Path = get_project_root()

def a2p(path: str) -> Path:
    """
        alias to path 將帶有別名的路徑轉換為實際路徑
        Args:
            path (str): 帶有別名的路徑
        Returns:
            Path: 轉換後的實際路徑
    """
    if path.startswith("@/"):
        return project_root.joinpath(PurePath(path[2:]))
    raise ValueError(f"Path '{path}' does not start with any known alias.")