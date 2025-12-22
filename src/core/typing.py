from typing import TypedDict


class ConfigDict(TypedDict):
    model_name: str
    seed: int | None
    batch_size: int
    gamma: float
    lr: float
    epochs: int
    patch_size: int
    rl_step: int
    seq_len: int
    task: str
