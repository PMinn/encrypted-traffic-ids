import os
import time
# third-party library
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.cuda
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, precision_score, recall_score
import logging
logger = logging.getLogger()


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
########Hyperparameter############
# gamma = 0.001
seed = 42
# init_lr = 0.01
# EPOCH = 100
# BATCH_SIZE = 2048
# PATCH_SIZE = 3
# IMAGE_SIZE = 15
# RL_STEP = 5
##################################
class TypeException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class Train_1D:
    def __init__(self,model,data_path,label_path,subdir,hyper_parameter,classes) :#,class_type = 'binary'

        self.model = model
        self.data_path = data_path
        self.label_path = label_path
        self.BATCH_SIZE = hyper_parameter['batch_size']
        self.PATCH_SIZE = hyper_parameter['patch_size']
        self.EPOCH = hyper_parameter['epoch']
        self.gamma = hyper_parameter['gamma']
        self.RL_STEP = hyper_parameter['learning_rate_step']
        self.init_lr = hyper_parameter['learning_rate']
        self.seq_size = hyper_parameter['seq_size']
        self.classes = classes
        self.valL_loss = list()
        self.best_val_acc = 0

        self.subdir = subdir
        if not os.path.exists(self.subdir+'/pth'):
            os.makedirs(self.subdir+'/pth')

        with open(self.subdir+'/Hyperparameter.txt','w') as file:
            file.write('gamma: '+str(self.gamma)+'\n')
            file.write('seed: '+str(seed)+'\n')
            file.write('init_lr: '+str(self.init_lr)+'\n')
            file.write('learning_rate_step: '+str(self.RL_STEP)+'\n')
            file.write('epoch number: '+str(self.EPOCH)+'\n')
            file.write('batch size: '+str(self.BATCH_SIZE)+'\n')
            file.write('patch size: '+str(self.PATCH_SIZE)+'\n')
            file.write('seq size: '+str(self.seq_size)+'\n')
            file.write('train data path: '+self.data_path+'\n')

        # 紀錄 loss 用
        self.train_loss_curve = []
        self.val_loss_curve = []


        logger.info('process dataset....')
        self.train_loader,self.valid_loader,self.train_num,self.valid_num = self.input_train_data(self.data_path,self.label_path,self.BATCH_SIZE) #(15,12)=180

        self.train()

    def get_dataLoader(self,data,label,batch_size):
        data = torch.tensor(data, dtype=torch.float)
        data_set = TensorDataset(data, torch.tensor(label, dtype=torch.float))
        data_loader = DataLoader(data_set, batch_size = batch_size, shuffle = True)
        return data_loader

    def input_train_data(self,data_path,label_path,batch_size):
        data = np.load(data_path)
        label = np.load(label_path)

        data = data/255

        train_data, val_data, train_label, val_label = train_test_split(data,label, test_size=0.1, stratify=label, random_state=42)#分訓練/驗證
        logger.info('-' * 25 +'train data'+'-' * 25 )
        for i in set(train_label):
            logger.info(f'{self.classes[i]}: ',np.sum(train_label == i ))

        logger.info('-' * 25 +'valid data'+'-' * 25 )
        for i in set(val_label):
            logger.info(f'{self.classes[i]}: ',np.sum(val_label == i ))

        train_loader = self.get_dataLoader(train_data,train_label,batch_size)
        val_loader = self.get_dataLoader(val_data,val_label,batch_size)

        return train_loader,val_loader,len(train_data),len(val_data)

    def adjust_lr(self,optimizer, epoch):

        # 1/10 learning rate every 5 epochs
        lr = self.init_lr * (0.001 ** (epoch // 10))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        logger.info('learning rate:',lr )

#######################################################################  binary  #############################################################################

    def validation(self,model,epoch,loss):
        val_total_correct = 0
        model = model.cuda(device=device)
        model.eval()
        #logger.info(torch.cuda.get_device_name())
        criterion = nn.CrossEntropyLoss()

        class_correct = list(0. for i in enumerate(self.classes))
        class_total = list(0. for i in enumerate(self.classes))

        all_labels = []
        all_predictions = []

        with torch.no_grad():#val
            val_acc_add = 0.0
            val_loss=0.0
            for inputs, labels in self.valid_loader:
                inputs = inputs.cuda(device=device)
                labels = labels.cuda(device=device)
                #logger.info(torch.cuda.get_device_name())
                inputs = inputs.unsqueeze(1)
                outputs, cls_token_features = model(inputs)

                # val_loss = criterion(outputs, labels.long())
                loss = criterion(outputs, labels.long())
                # logger.info(loss.item())
                # val_loss += float(loss.item() * inputs.size(0))
                val_loss += float(loss * inputs.size(0))
                _, predicted = torch.max(outputs.data, 1)
                # logger.info('predicted:',predicted)

                val_total_correct += (predicted == labels).sum().item()
                c = (predicted == labels).squeeze()

                # batch size
                for i in range(labels.size(0)):
                    label = int(labels[i])
                    class_correct[label] += c[i].item()
                    class_total[label] += 1


                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
            # Calculate F1 score
            f1 = f1_score(all_labels, all_predictions, average='weighted')

            with open(self.subdir+'/valid_Acc_Loss.txt', 'a') as file:
                file.write('Epoch :'+str(epoch+1)+'/'+str(self.EPOCH)+'\n')
                for i, c in enumerate(self.classes):
                    try:
                        val_acc_add += class_correct[i] / class_total[i]
                    except ZeroDivisionError:
                        val_acc_add += 0
                    try:
                        acc = 100 * class_correct[i] / class_total[i]
                    except:
                        acc = 0
                    file.write(f'valid Accuracy of {c}: {str(acc)}% \n')
                    logger.info('valid Accuracy of %5s : %8.4f %%' % (c, acc))

                val_acc = val_acc_add/len(self.classes)
                # val_loss = val_loss.item() / self.valid_num
                val_loss = val_loss / self.valid_num
                file.write(f'Valid Accuracy:{str(val_acc)} | loss: {str(val_loss)} \n')
                logger.info(f'Valid Accuracy:{str(val_acc)}| loss:{str(val_loss)}| F1 Score: {f1:.4f}')

        # 紀錄 val_loss
        self.val_loss_curve.append(val_loss)
        self.valL_loss.append(val_loss)
        if epoch % 5 == 0:
            torch.save(model, '{}/pth/model-{:.2f}-val_acc-{}-epoch.pth'.format(self.subdir,val_acc,epoch))

        if val_acc >= self.best_val_acc:
            self.best_val_acc = val_acc
            # best_val_model_params = copy.deepcopy(model.state_dict())
            with open(self.subdir+'/Best_valid_Acc.txt', 'w') as file:
                file.write('Best Epoch :'+str(epoch+1)+f'\n Best acc: {self.best_val_acc}')

            torch.save(model, f'{self.subdir}/pth/model-best_val_acc.pth')
#######################################################################  binary  #############################################################################




#######################################################################  multi  #############################################################################
    # def validation(self, model, epoch, loss):
    #     val_total_correct = 0
    #     model = model.cuda(device=device)
    #     model.eval()
    #     criterion = nn.CrossEntropyLoss()

    #     class_correct = list(0. for i in range(len(self.classes)))
    #     class_total = list(0. for i in range(len(self.classes)))

    #     all_labels = []
    #     all_predictions = []

    #     with torch.no_grad():  # val
    #         val_acc_add = 0.0
    #         val_loss = 0.0
    #         for inputs, labels in self.valid_loader:
    #             inputs = inputs.cuda(device=device)
    #             labels = labels.cuda(device=device)
    #             inputs = inputs.unsqueeze(1)
    #             outputs, cls_token_features = model(inputs)

    #             loss = criterion(outputs, labels.long())
    #             val_loss += float(loss * inputs.size(0))
    #             _, predicted = torch.max(outputs.data, 1)

    #             val_total_correct += (predicted == labels).sum().item()
    #             c = (predicted == labels).squeeze()

    #             for i in range(labels.size(0)):
    #                 label = int(labels[i])
    #                 class_correct[label] += c[i].item()
    #                 class_total[label] += 1

    #             all_labels.extend(labels.cpu().numpy())
    #             all_predictions.extend(predicted.cpu().numpy())

    #         # Calculate metrics
    #         f1_per_class = f1_score(all_labels, all_predictions, average=None)
    #         precision_per_class = precision_score(all_labels, all_predictions, average=None)
    #         recall_per_class = recall_score(all_labels, all_predictions, average=None)

    #         overall_f1 = f1_score(all_labels, all_predictions, average='micro')
    #         overall_precision = precision_score(all_labels, all_predictions, average='micro')
    #         overall_recall = recall_score(all_labels, all_predictions, average='micro')

    #         with open(self.subdir + '/valid_Acc_Loss.txt', 'a') as file:
    #             file.write('Epoch :' + str(epoch + 1) + '/' + str(self.EPOCH) + '\n')
    #             for i, c in enumerate(self.classes):
    #                 try:
    #                     val_acc_add += class_correct[i] / class_total[i]
    #                 except ZeroDivisionError:
    #                     val_acc_add += 0
    #                 try:
    #                     acc = 100 * class_correct[i] / class_total[i]
    #                 except ZeroDivisionError:
    #                     acc = 0
    #                 file.write(f'valid Accuracy of {c}: {acc:.2f}% \n')
    #                 file.write(f'Precision: {precision_per_class[i]:.4f} | Recall: {recall_per_class[i]:.4f} | F1 Score: {f1_per_class[i]:.4f}\n')
    #                 # logger.info(f'valid Accuracy of {c} : {acc:.2f}% | Precision: {precision_per_class[i]:.4f} | Recall: {recall_per_class[i]:.4f} | F1 Score: {f1_per_class[i]:.4f}')
    #                 logger.info(f'valid Accuracy of {c} : {acc:.2f}%')
    #             val_acc = val_acc_add / len(self.classes)
    #             val_loss = val_loss / self.valid_num
    #             file.write(f'\nOverall Metrics: \n')
    #             file.write(f'Valid Accuracy: {val_acc:.4f} | Loss: {val_loss:.4f}\n')
    #             file.write(f'F1 Score: {overall_f1:.4f} | Precision: {overall_precision:.4f} | Recall: {overall_recall:.4f}\n')
    #             logger.info(f'Overall Metrics: Valid Accuracy: {val_acc:.4f} | Loss: {val_loss:.4f} | F1 Score: {overall_f1:.4f} | Precision: {overall_precision:.4f} | Recall: {overall_recall:.4f}')

    #     self.valL_loss.append(val_loss)
    #     if epoch % 5 == 0:
    #         torch.save(model, './{}/pth/model-{:.2f}-val_acc-{}-epoch.pth'.format(self.subdir, val_acc, epoch))

    #     if val_acc >= self.best_val_acc:
    #         self.best_val_acc = val_acc
    #         with open(self.subdir + '/Best_valid_Acc.txt', 'w') as file:
    #             file.write('Best Epoch :' + str(epoch + 1) + f'\n Best acc: {self.best_val_acc}')
    #         torch.save(model, f'./{self.subdir}/pth/model-best_val_acc.pth')

#######################################################################  multi  #############################################################################


    def train(self,):
        train_loss = list()

        model = self.model.cuda(device=device)
        logger.info(torch.cuda.get_device_name())
        model = nn.DataParallel(model)
        # loss function
        criterion = nn.CrossEntropyLoss()
        # criterion = FocalLoss()
        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.init_lr)
        # optimizer = torch.optim.SGD(params=model.parameters(), lr=init_lr, momentum=0.5)
        # scheduler
        scheduler = StepLR(optimizer, step_size=self.RL_STEP, gamma=self.gamma)

        logger.info('start training ...')
        for epoch in range(self.EPOCH):

            localtime = time.asctime( time.localtime(time.time()) )
            training_loss = 0.0
            training_corrects = 0
            train_acc_add = 0.0
            train_class_correct = list(0. for i in enumerate(self.classes))
            train_class_total = list(0. for i in enumerate(self.classes))
            # parameter_total = 0


            # 在每个批次的迭代中，更新学习率
            # adjust_lr(optimizer, epoch)
            scheduler.step()# 更新学习率
            logger.info("learning rate:", scheduler.get_last_lr())

            logger.info('Epoch: {}/{} --- < Starting Time : {} >'.format(epoch + 1,self.EPOCH,localtime))
            logger.info('-' * len('Epoch: {}/{} --- < Starting Time : {} >'.format(epoch + 1,self.EPOCH,localtime)))

            for i, (inputs, labels) in enumerate(self.train_loader):#train

                optimizer.zero_grad()  # clear gradients for this training step
                inputs = inputs.cuda(device=device)
                labels = labels.cuda(device=device)
                #logger.info(torch.cuda.get_device_name())
                inputs = inputs.unsqueeze(1)  # 增加一個新的維度來表示通道
                # logger.info(inputs.shape)
                outputs, cls_token_features = model(inputs) #結果出來了
                # logger.info(outputs)

                _, preds = torch.max(outputs.data, 1) #preds表示最大得分類別
                t = (preds == labels).squeeze()
                for i in range(labels.size(0)):
                    label = int(labels[i])
                    train_class_correct[label] += t[i].item()
                    train_class_total[label] += 1

                loss = criterion(outputs, labels.long())
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

                training_loss += float(loss.item() * inputs.size(0))#單一批次總損失
                training_corrects += torch.sum(preds == labels.data)#預測正確數量

            training_loss = training_loss / self.train_num
            training_acc = training_corrects.double() /self.train_num
            train_loss.append(training_loss)
            # 紀錄 train_loss
            self.train_loss_curve.append(training_loss)
            with open(self.subdir+'/train_Acc_Loss.txt', 'a') as file:
                file.write('Epoch :'+str(epoch+1)+'/'+str(self.EPOCH)+'\n')
                for i, c in enumerate(self.classes):
                    try:
                        train_acc_add += train_class_correct[i] / train_class_total[i] #累加準確率
                    except ZeroDivisionError:
                        train_acc_add += 0
                    try:
                        acc = 100 * train_class_correct[i] / train_class_total[i]
                    except ZeroDivisionError:
                        acc = 0

                    file.write(f'Train Accuracy of {c}: {str(acc)}% \n')
                    logger.info('Train Accuracy of %5s : %8.4f %%' % (c, acc))
                    logger.info(c+':',train_class_total[i])

                train_acc = train_acc_add #/ len(self.classes)
                file.write(f'Training Accuracy: {str(train_acc)}')
                file.write(f' | Training loss: {str(training_loss)}\n')
                logger.info('Training Accuracy: ',train_acc)

            logger.info('Training loss: {:.4f}\taccuracy: {:.4f}\n'.format(training_loss,training_acc))

            # if training_acc > best_train_acc:
            #     best_train_acc = training_acc
            #     best_model_params = copy.deepcopy(model.state_dict())
            self.validation(model,epoch,loss)

        plt.plot(range(1, self.EPOCH+1), self.train_loss_curve, label="Training Loss", color="blue")
        plt.plot(range(1, self.EPOCH+1), self.val_loss_curve, label="Validation Loss", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("ViT Overfitting Check")
        plt.legend()
        plt.savefig(f"{self.subdir}/loss_curve.png")
        plt.show()
        logger.info(f"Loss curve saved to {self.subdir}/loss_curve.png")

        logger.info(f'Best model save to: {self.subdir}/pth/model-best_val_acc.pth')
        parameter_total = sum([param.nelement() for param in model.parameters()])
        logger.info("Number of parameter: %.2fM" % (parameter_total/1e6))
        # model.load_state_dict(best_model_params)
        # best_model_name = './{}/pth/multi-ViT-model-best_train_acc.pth'.format(subdir,best_val_acc)
        # torch.save(model, best_model_name)
        # logger.info("Best model name : " + best_model_name)
        # draw('Training Loss Curve', 'epoch', 'loss', epochnum, train_loss, valL_loss, 'Training Loss Curve.png')


class Train_1D_Swin(Train_1D):
    def validation(self, model, epoch):
        """验证：只返回 logits，不解包 cls_token_features"""
        model = model.cuda(device=device)
        model.eval()
        criterion = nn.CrossEntropyLoss()

        class_correct = [0 for _ in self.classes]
        class_total   = [0 for _ in self.classes]
        all_labels = []
        all_preds  = []

        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.valid_loader:
                inputs = inputs.cuda(device=device).unsqueeze(1)  # (B,1,L)
                labels = labels.cuda(device=device).long()

                outputs = model(inputs)                          # (B, num_classes)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                preds = outputs.argmax(dim=1)
                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())

                # per-class stats
                correct = preds.eq(labels)
                for i, lbl in enumerate(labels.cpu().tolist()):
                    class_total[lbl] += 1
                    if correct[i]:
                        class_correct[lbl] += 1

        # 计算平均 loss & accuracy
        val_loss = val_loss / self.valid_num
        val_acc = sum(c / t if t>0 else 0 for c,t in zip(class_correct, class_total)) / len(self.classes)

        # 计算 F1 分数
        f1 = f1_score(all_labels, all_preds, average='weighted')

        # 写日志 & 保存最佳
        with open(f'{self.subdir}/valid_Acc_Loss.txt','a') as f:
            f.write(f'Epoch {epoch+1}/{self.EPOCH}\n')
            for idx, cname in enumerate(self.classes):
                acc = 100 * class_correct[idx] / class_total[idx] if class_total[idx]>0 else 0
                f.write(f'  {cname}: {acc:.2f}%\n')
            f.write(f'  Avg Acc: {val_acc:.4f}  Loss: {val_loss:.4f}  F1: {f1:.4f}\n\n')

        logger.info(f'Validation — Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  F1: {f1:.4f}')

        # 更新曲线 & 保存最佳模型
        self.val_loss_curve.append(val_loss)
        if val_acc >= self.best_val_acc:
            self.best_val_acc = val_acc
            torch.save(model, f'{self.subdir}/pth/model-best_val_acc.pth')

    def train(self):
        model = self.model.cuda(device=device)
        model = nn.DataParallel(model)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.init_lr)
        scheduler = StepLR(optimizer, step_size=self.RL_STEP, gamma=self.gamma)

        for epoch in range(self.EPOCH):
            model.train()
            scheduler.step()
            logger.info(f'\nEpoch {epoch+1}/{self.EPOCH} — LR: {scheduler.get_last_lr()[0]:.2e}')

            train_loss = 0.0
            train_correct = 0
            class_corr = [0 for _ in self.classes]
            class_tot  = [0 for _ in self.classes]

            for inputs, labels in self.train_loader:
                inputs = inputs.cuda(device=device).unsqueeze(1)
                labels = labels.cuda(device=device).long()

                optimizer.zero_grad()
                outputs = model(inputs)             # (B, num_classes)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                train_correct += preds.eq(labels).sum().item()

                corr = preds.eq(labels)
                for i, lbl in enumerate(labels.cpu().tolist()):
                    class_tot[lbl]  += 1
                    class_corr[lbl] += corr.cpu().tolist()[i]

            # 统计并写日志
            train_loss = train_loss / self.train_num
            train_acc = sum(c/t if t>0 else 0 for c,t in zip(class_corr,class_tot)) / len(self.classes)
            with open(f'{self.subdir}/train_Acc_Loss.txt','a') as f:
                f.write(f'Epoch {epoch+1}/{self.EPOCH}\n')
                for idx, cname in enumerate(self.classes):
                    acc = 100 * class_corr[idx] / class_tot[idx] if class_tot[idx]>0 else 0
                    f.write(f'  {cname}: {acc:.2f}%\n')
                f.write(f'  Avg Acc: {train_acc:.4f}  Loss: {train_loss:.4f}\n\n')

            logger.info(f'Train   — Loss: {train_loss:.4f}  Acc: {train_acc:.4f}')

            # 每 epoch 验证
            self.validation(model, epoch)

        # 绘制 & 保存 loss 曲线
        plt.plot(self.train_loss_curve, label='Train Loss')
        plt.plot(self.val_loss_curve,   label='Val   Loss')
        plt.legend()
        plt.savefig(f'{self.subdir}/loss_curve.png')
        logger.info(f'Loss curve → {self.subdir}/loss_curve.png')
        logger.info(f'Best model → {self.subdir}/pth/model-best_val_acc.pth')
