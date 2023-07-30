'''
Train a student network without KD
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





def train(config, dataloader):
    print('--------------Train Prossing--------------')
    model = SCNN()
    if use_gpu:
        model.cuda()

    model.train()

    wandb.watch(model, log="all")

    acc_max = 0
    loss_func = nn.CrossEntropyLoss()

    for e in range(config.epochs):
        for phase in ['train', 'validation']:
            loss = 0
            total = 0
            correct = 0
            loss_total = 0

            if phase == 'train':
                model.train()
            if phase == 'validation':
                model.eval()
                torch.no_grad()

            for step, (x, y) in enumerate(dataloader[phase]):

                x = x.type(torch.float)
                y = y.type(torch.long)
                y = y.view(-1)
                if use_gpu:
                    x, y = x.cuda(), y.cuda()

                optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs,
                                                                       eta_min=1e-8)
                if use_gpu:
                    y_t = model(x).cuda()

                else:
                    y_t = model(x)

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
                wandb.log({
                    "T_Train Accuracy": 100. * acc,
                    "T_Train Loss": loss_total})
            if phase == 'validation':
                scheduler.step(loss_total)
                wandb.log({
                    "T_Validation Accuracy": 100. * acc,
                    "T_Validation Loss": loss_total})
                if acc >= acc_max:
                    acc_max = acc
                    tnet_best = model
            print('%s ACC:%.4f' % (phase, acc))
    if not os.path.exists('Pth'):
        os.mkdir('Pth')
    return tnet_best


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

    test_f1 = []
    test_FPR = []
    test_recall = []
    test_precision = []
    for seed in range(config.seed - 5, config.seed + 5):

        random_seed(seed)
        # test set, number denotes each category has 250 samples
        # for seed in range(42, 52):
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
        test_X =  test_X[:, np.newaxis, :]
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
    file = "Pth/" + "SCNN_" + config.chosen_model + '_' + str(config.snr) + '_' +  config.chosen_data + '_' + str(round(np.mean(test_f1).item(), 2)) + ".pth"
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

    snet = train(config, data_loaders)
    inference10(snet,config)



if __name__ == '__main__':
    # wandb initialization, you need to create a wandb account and enter the username in 'entity'
    wandb.init(project="DKD", entity="jing-xiaoliao")
    # WandB – Config is a variable that holds and saves hypermarkets and inputs
    config = wandb.config  # Initialize config
    config.no_cuda = False  # disables CUDA training
    config.log_interval = 200  # how many batches to wait before logging training status
    config.seed = 42  # random seed (default: 42)

    # Hyperparameters, lr and alpha need to fine-tune
    config.batch_size = 64  # input batch size for training (default: 64)
    config.epochs = 75  # number of epochs to train (default: 10)
    config.lr = 0.5  # learning rate (default: 0.5)

    # noisy condition
    config.add_noise = True
    config.snr = 8

    # dataset and model
    config.chosen_data = '0HP'

    main(config)