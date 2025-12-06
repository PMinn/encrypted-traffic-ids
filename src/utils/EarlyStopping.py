import torch
import logging

class EarlyStopping:
    """
        EarlyStopping 可監控 validation accuracy，當連續 patience 次無改善時停止訓練。
        同時支援保存最佳模型權重。
    """
    def __init__(self, patience: int = 5, delta: float = 0.0001, path: str = "checkpoint.pth", verbose: bool = True):
        """
            Args:
                patience (int): 容忍無改善的次數
                delta (float): 改善的最小變化量
                path (str): 保存最佳模型權重的路徑
                verbose (bool): 是否輸出日誌訊息
            Returns:
                None
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        
        self.best_acc = float('-inf')
        self.no_improvement = 0
        
        self.logger = logging.getLogger("EarlyStopping")

    def check(self, val_acc: float, model: torch.nn.Module = None):
        """
            傳入 val_acc，並選擇性傳入 model 以保存最佳權重。
            Args:
                val_acc (float): 當前的 validation accuracy
                model (torch.nn.Module, optional): 當前模型，用於保存最佳權重
            Return:
                stop (bool)：是否應該停止訓練
                improved (bool)：此次是否改善
        """
        improved = False
        
        if val_acc > self.best_acc + self.delta:
            # 更新最佳紀錄
            self.best_acc = val_acc
            self.no_improvement = 0
            improved = True
            
            # 保存 checkpoint
            if model is not None and self.path:
                torch.save(model, self.path)

            if self.verbose:
                self.logger.info(f"Validation accuracy improved to {val_acc:.4f}.")
        
        else:
            self.no_improvement += 1
            if self.verbose:
                self.logger.info(
                    f"No improvement ({self.no_improvement}/{self.patience}). "
                    f"Current accuracy: {val_acc:.4f}, Best accuracy: {self.best_acc:.4f}"
                )
        
        # 若持續無改善 → 停止
        stop = self.no_improvement >= self.patience
        if stop and self.verbose:
            self.logger.info("Early stopping triggered.")
            if model is not None and self.path:
                self.logger.info(f"Best model saved at {self.path} with accuracy {self.best_acc}.")
        
        return stop, improved