import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrainMLPClassifier:
    def __init__(self, model, data_path, label_path, sub_dir, hyper_parameter):
        self.epoch = hyper_parameter['epoch']
        self.LR = hyper_parameter['learningRate']
        self.gamma = hyper_parameter['gamma']
        self.batchSize = hyper_parameter['batchSize']
        self.stepLR = hyper_parameter['stepLR']

        self.model = model.to(device)
        self.sub_dir = sub_dir
        self.data_path = data_path
        self.label_path = label_path

        if not os.path.exists(f'{self.sub_dir}/pth'):
            os.makedirs(f'{self.sub_dir}/pth')

        self.train_loader, self.valid_loader = self.load_data(self.data_path, self.label_path, self.batchSize)
        self.train()

    def load_data(self, data_path, label_path, batch_size):
        x = np.load(data_path)
        y = np.load(label_path)

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=42)

        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        x_val = torch.tensor(x_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)

        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        return train_loader, val_loader

    def train(self):
        model = nn.DataParallel(self.model).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.LR)
        scheduler = StepLR(optimizer, step_size=self.stepLR, gamma=self.gamma)

        best_val_loss = float('inf')
        print(f"\n{'-'*20} Start Training {'-'*20}\n")

        for epoch in range(self.epoch):
            model.train()
            total_loss = 0
            all_preds = []
            all_labels = []

            scheduler.step()
            print(f"Epoch {epoch+1}/{self.epoch}, LR: {scheduler.get_last_lr()}")

            for x_batch, y_batch in self.train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                output = model(x_batch)
                loss = criterion(output, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

            avg_train_loss = total_loss / len(self.train_loader)
            train_acc = accuracy_score(all_labels, all_preds)
            train_f1 = f1_score(all_labels, all_preds, average='macro')

            avg_val_loss, val_acc, val_f1 = self.validate(model, criterion)

            print(f"[Train] Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
            print(f"[Valid] Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f'{self.sub_dir}/pth/mlp_best_model.pth')
                print(f">>> Model saved: {self.sub_dir}/pth/mlp_best_model.pth")

        print("\nTraining finished.\n")

    def validate(self, model, criterion):
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x_batch, y_batch in self.valid_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                output = model(x_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item()

                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        avg_loss = val_loss / len(self.valid_loader)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')

        return avg_loss, acc, f1
