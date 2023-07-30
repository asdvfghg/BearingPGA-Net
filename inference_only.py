'''
Test Model
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
        train_X, valid_X, test_X = train_X[:, np.newaxis, :], valid_X[:, np.newaxis, :], test_X[:, np.newaxis, :]

        test_dataset = CustomTensorDataset(torch.tensor(test_X, dtype=torch.float), torch.tensor(test_Y))

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
        f1, FPR, recall, precision = inference(test_loader, net0)
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






if __name__ == '__main__':
    wandb.init(project="DKD", entity="jing-xiaoliao")

    config = wandb.config  # Initialize config
    # noisy condition
    config.add_noise = True
    config.snr = -6
    config.seed = 42
    # dataset and model
    config.chosen_data = '0HP'  #0.1HP-1800 0HP 1HP 2HP 3HP
    config.chosen_model = 'DKD'
    net = SCNN() # SCNN TCNN
    file = "Pth/" + "DKD11_DKD-6_0HP_0.9618907769281796" + ".pth"
    ckpt = torch.load(file, map_location=lambda storage, loc: storage.cuda())
    net.load_state_dict(ckpt)
    inference10(net,config)

