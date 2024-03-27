'''
Train and evaluate BearingPGA-Net via decoupled knowledge distillation
'''

import os
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, recall_score, \
    precision_score, f1_score

import wandb
from torch import nn
from torch.utils.data import DataLoader
from Model.Student import SCNN
from Model.Teacher import TCNN

from utils.DatasetLoader import CustomTensorDataset
from utils.Preprocess import prepro
from utils.loss_fun import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()
print(torch.cuda.is_available())





def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def select_model(chosen_model):
    if chosen_model == 'DKD':
        model = SCNN()
    print(model)
    return model


def train_teacher(config, dataloader):
    print('--------------Train Student Prossing--------------')
    teacher_model = TCNN()
    if use_gpu:
        teacher_model.cuda()

    teacher_model.train()

    wandb.watch(teacher_model, log="all")

    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []
    acc_max = 0
    loss_func = nn.CrossEntropyLoss()

    for e in range(config.epochs):
        for phase in ['train', 'validation']:
            loss = 0
            total = 0
            correct = 0
            loss_total = 0

            if phase == 'train':
                teacher_model.train()
            if phase == 'validation':
                teacher_model.eval()
                torch.no_grad()

            for step, (x, y) in enumerate(dataloader[phase]):

                x = x.type(torch.float)
                y = y.type(torch.long)
                y = y.view(-1)
                if use_gpu:
                    x, y = x.cuda(), y.cuda()

                optimizer = torch.optim.SGD(teacher_model.parameters(), lr=config.tlr, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs,
                                                                       eta_min=1e-8)
                if use_gpu:
                    y_t = teacher_model(x).cuda()

                else:
                    y_t = teacher_model(x)

                hard_loss = loss_func(y_t, y)

                if phase == 'train':
                    optimizer.zero_grad()
                    hard_loss.backward()
                    optimizer.step()

                loss_total += hard_loss.item()
                y_predict = y_t.argmax(dim=1)

                total += y.size(0)
                if use_gpu:
                    correct += (y_predict == y).cpu().squeeze().sum().numpy()
                else:
                    correct += (y_predict == y).squeeze().sum().numpy()

                if step % 20 == 0 and phase == 'train':
                    print('Epoch:%d, Step [%d/%d], Loss: %.4f'
                          % (
                              e + 1, step + 1, len(dataloader[phase]), loss_total / len(dataloader[phase].dataset)))
            loss_total = loss_total / len(dataloader[phase].dataset)

            acc = correct / total
            if phase == 'train':
                train_loss.append(loss_total)
                train_acc.append(acc)
                wandb.log({
                    "T_Train Accuracy": 100. * acc,
                    "T_Train Loss": loss_total})
            if phase == 'validation':
                scheduler.step(loss_total)
                valid_loss.append(loss_total)
                valid_acc.append(acc)
                wandb.log({
                    "T_Validation Accuracy": 100. * acc,
                    "T_Validation Loss": loss_total})
                if acc >= acc_max:
                    acc_max = acc
                    # file = "Pth/" + config.stdut + "_best" + str(acc_max.item()) + ".pth"
                    # torch.save(net.state_dict(), file)
                    tnet_best = teacher_model
            print('%s ACC:%.4f' % (phase, acc))
    if not os.path.exists('Pth'):
        os.mkdir('Pth')
    file = "Pth/" + "DKD11_" + 'Tmodel' + str(config.snr) + '_' + config.chosen_data + '_'  + str(round(acc_max.item(),2)) + ".pth"
    torch.save(teacher_model.state_dict(), file)
    wandb.save('*.pth')
    return tnet_best

def train_student(config, dataloader, tnet_best):
    print('--------------Train Student Prossing--------------')
    teacher_model = tnet_best
    student_model = select_model(config.chosen_model)
    if use_gpu:
        teacher_model.cuda()
        student_model.cuda()

    student_model.train()
    teacher_model.eval()

    wandb.watch(student_model, log="all")


    acc_max = 0
    loss_func = nn.CrossEntropyLoss()

    for e in range(config.epochs):
        for phase in ['train', 'validation']:
            loss = 0
            total = 0
            correct = 0
            loss_total = 0
            if phase == 'train':
                student_model.train()
            if phase == 'validation':
                student_model.eval()
                torch.no_grad()

            for step, (x, y) in enumerate(dataloader[phase]):

                x = x.type(torch.float)
                y = y.type(torch.long)
                y = y.view(-1)
                if use_gpu:
                    x, y = x.cuda(), y.cuda()

                optimizer = torch.optim.SGD(student_model.parameters(), lr=config.slr, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs,
                                                                     eta_min=1e-8)
                if use_gpu:
                    y_s = student_model(x).cuda()
                    y_t = teacher_model(x).cuda()

                else:
                    y_s = student_model(x)
                    y_t = teacher_model(x)
                soft_loss = dkd_loss(y_s, y_t, y, config)
                soft_loss = min(e / config.warmup, 1.0) * soft_loss # DKD
                # soft_loss = min(e / config.warmup, 1.0) * kd_loss(y_s, y_t, config) # KD
                hard_loss = loss_func(y_s, y)
                loss = (1-config.alpha)*hard_loss + config.alpha*soft_loss

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                loss_total += loss.item()
                y_predict = y_s.argmax(dim=1)

                total += y.size(0)
                if use_gpu:
                    correct += (y_predict == y).cpu().squeeze().sum().numpy()
                else:
                    correct += (y_predict == y).squeeze().sum().numpy()

                if step % 20 == 0 and phase == 'train':
                    print('Epoch:%d, Step [%d/%d], Loss: %.4f'
                          % (
                          e + 1, step + 1, len(dataloader[phase]), loss_total / len(dataloader[phase].dataset)))
            loss_total = loss_total / len(dataloader[phase].dataset)

            acc = correct / total
            if phase == 'train':

                wandb.log({
                    "S_Train Accuracy": 100. * acc,
                    "S_Train Loss": loss_total})
            if phase == 'validation':
                scheduler.step(loss_total)

                wandb.log({
                    "S_Validation Accuracy": 100. * acc,
                    "S_Validation Loss": loss_total})
                if acc >= acc_max:
                    acc_max = acc
                    net_best = student_model
            print('%s ACC:%.4f' % (phase, acc))


    return net_best


def inference(dataloader, model):
    net = model
    y_list, y_predict_list = [], []
    if use_gpu:
        net.cuda()
    net.eval()
    # endregion
    with torch.no_grad():
        for step, (x, y) in enumerate(dataloader):
            x = x.type(torch.float)
            y = y.type(torch.long)
            y = y.view(-1)
            if use_gpu:
                x, y = x.cuda(), y.cuda()
            y_hat = net(x)
            y_predict = y_hat.argmax(dim=1)
            y_list.extend(y.detach().cpu().numpy())
            y_predict_list.extend(y_predict.detach().cpu().numpy())

        cnf_matrix = confusion_matrix(y_list, y_predict_list)
        recall = recall_score(y_list, y_predict_list, average="macro")
        precision = precision_score(y_list, y_predict_list, average="macro")

        F1 = f1_score(y_list, y_predict_list, average="macro")
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)

        FP = FP.astype(float)
        TN = TN.astype(float)
        FPR = np.mean(FP / (FP + TN))
        print(F1, FPR, recall, precision)
        return F1,FPR,recall,precision

def inference10(net0,config):
    path = os.path.join('data', config.chosen_data)
    # train set, number denotes each category has 750 samples

    test_f1 = []
    test_FPR = []
    test_recall = []
    test_precision = []
    for seed in range(config.seed - 5, config.seed + 5):

        # if(seed == 40|seed==39):
        #     continue
        random_seed(seed)

        test_X, test_Y = prepro(d_path=path,
                                length=2048,
                                number=250,
                                normal=True,
                                enc=True,
                                enc_step=28,
                                snr=config.snr,
                                property='Test',
                                noise=config.add_noise
                                )
        test_dataset = CustomTensorDataset(torch.tensor(test_X, dtype=torch.float), torch.tensor(test_Y))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
        f1, FPR, recall, precision = inference(test_loader, net0)
        if (seed == 42):
            F1 = f1
        print(seed)
        test_f1.append(f1)
        test_FPR.append(FPR)
        test_recall.append(recall)
        test_precision.append(precision)

    print("test_f1: mean: ", np.mean(test_f1), "var: ", np.var(test_f1))
    print("test_FPR: mean: ", np.mean(test_FPR), "var: ", np.var(test_FPR))
    print("test_recall: mean: ", np.mean(test_recall), "var: ", np.var(test_recall))
    print("test_precision: mean: ", np.mean(test_precision), "var: ", np.var(test_precision))
    # print('model name:', str(file))
    wandb.log({
        "F1": F1,
        "test_f1": np.mean(test_f1),
        "test_FPR": np.mean(test_FPR),
        "test_recall": np.mean(test_recall),
        "test_precision": np.mean(test_precision)})
    file = "Pth/" + "DKD11_" + config.chosen_model + str(config.snr) + '_' + config.chosen_data + '_' + str(np.mean(test_f1).item()) + ".pth"
    torch.save(net0.state_dict(), file)
    wandb.save('*.pth')
    print('model saved')


def main(config):
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    random_seed(config.seed)

    path = os.path.join('data', config.chosen_data)
    # train set, number denotes each category has 750 samples
    train_X, train_Y, valid_X, valid_Y = prepro(d_path=path,
                                                 length=2048,
                                                 number=750,
                                                 normal=True,
                                                 enc=True,
                                                 enc_step=28,
                                                 snr=config.snr,
                                                 property='Train',
                                                 noise=config.add_noise
                                                 )

    train_X, valid_X = train_X[:, np.newaxis, :], valid_X[:, np.newaxis, :]

    train_dataset = CustomTensorDataset(torch.tensor(train_X, dtype=torch.float), torch.tensor(train_Y))
    valid_dataset = CustomTensorDataset(torch.tensor(valid_X, dtype=torch.float), torch.tensor(valid_Y))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    data_loaders = {
        "train": train_loader,
        "validation": valid_loader
    }
    # Stage 1: Training Teacher model
    tnet = train_teacher(config, data_loaders)
    # tnet = TCNN()
    # file = "Pth/" + "DKD11_Teachermodel-60HP0.10016025641025642" + ".pth"
    # ckpt = torch.load(file, map_location=lambda storage, loc: storage.cuda())
    # tnet.load_state_dict(ckpt)
    # Stage 2: Training Student model
    snet = train_student(config, data_loaders, tnet)
    inference10(snet,config)



if __name__ == '__main__':
    # wandb initialization, you need to create a wandb account and enter the username in 'entity'
    wandb.init(project="DKD", entity="jing-xiaoliao")
    # WandB – Config is a variable that holds and saves hypermarkets and inputs
    config = wandb.config  # Initialize config
    config.no_cuda = False  # disables CUDA training
    config.log_interval = 200  # how many batches to wait before logging training status
    config.seed = 42  # random seed (default: 42)

    # Hyperparameters need to fine-tune
    config.batch_size = 64  # input batch size for training (default: 64)
    config.epochs = 75  # number of epochs to train (default: 10)
    config.tlr = 0.1  # teacher learning rate (default: 0.5)
    config.slr = 0.1  # student learning rate (default: 0.5)

    config.alpha = 0.2 # scale factor alpha hard loss and dkd loss
    #DKD
    config.beta = 5 #TCKD  beta
    config.gammar = 1  #NCKD  gammar
    config.warmup = 5

    #KD temperature
    config.temperature = 2.5

    # noisy condition
    config.add_noise = True
    config.snr = -6

    # dataset and model
    config.chosen_data = '0HP'
    config.chosen_model = 'DKD'

    main(config)
