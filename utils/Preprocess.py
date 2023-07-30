
from scipy.io import loadmat
import numpy as np
import os
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
import random
import torch
'''
Preprocess for training and test
Refer to https://github.com/AaronCosmos/wdcnn_bearning_fault_diagnosis
'''
def random_seed(seed = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def prepro(d_path, length=2048, number=1000, normal=True, enc=True, enc_step=28, snr=0, property='Train', noise=True):

    if (property == 'Train') & (noise == True) :
        d_path = d_path +'_TrainNoised_' + str(snr)
    elif (property == 'Train_only') & (noise == True):
        d_path = d_path +'_TrainNoised_' + str(snr)
    elif (property == 'Test') & (noise == True):
        d_path = d_path +'_TestNoised_' + str(snr)
    elif (property == 'Train') & (noise == False):
        d_path = d_path +'_Train'
    elif (property == 'Test') & (noise == False):
        d_path = d_path +'_Test'
    # elif (noise ==False) | (snr == 0):
    #     d_path = d_path
    filenames = os.listdir(d_path)

    def capture_mat():
        files = {}
        for i in filenames:
            # 文件路径
            file_path = os.path.join(d_path, i)
            file = loadmat(file_path)
            file_keys = file.keys()
            for key in file_keys:
                if 'DE' in key:
                    files[i] = file[key].ravel()
            # files[i] = file['DE'].ravel()
        return files

    def slice_enc(data):
        keys = data.keys()
        Train_Samples = {}
        for i in keys:
            slice_data = data[i]
            all_length = len(slice_data)
            end_index = int(all_length)
            samp_train = int(number)
            Train_sample = []
            if enc:
                enc_time = length // enc_step
                samp_step = 0
                for j in range(samp_train):
                    random_start = np.random.randint(low=0, high=(end_index - 2 * length))
                    label = 0
                    for h in range(enc_time):
                        samp_step += 1
                        random_start += enc_step
                        sample = slice_data[random_start: random_start + length]
                        Train_sample.append(sample)
                        if samp_step == samp_train:
                            label = 1
                            break
                    if label:
                        break
            else:
                for j in range(samp_train):  # 抓取训练数据
                    random_start = np.random.randint(low=0, high=(end_index - length))
                    sample = slice_data[random_start:random_start + length]
                    Train_sample.append(sample)
            Train_Samples[i] = Train_sample

        return Train_Samples

    def add_labels(train_test):
        X = []
        Y = []
        label = 0
        for i in filenames:
            x = train_test[i]
            X += x
            lenx = len(x)
            Y += [label] * lenx
            label += 1
        return X, Y

    def one_hot(Train_Y):
        Train_Y = np.array(Train_Y).reshape([-1, 1])
        Encoder = preprocessing.OneHotEncoder()
        Encoder.fit(Train_Y)
        Train_Y = Encoder.transform(Train_Y).toarray()
        Train_Y = Encoder.inverse_transform(Train_Y)
        # Train_Y = np.asarray(Train_Y, dtype=float)
        Train_Y = np.asarray(Train_Y, dtype=np.int32)
        return Train_Y

    def scalar_stand(Train_X):
        scalar = preprocessing.StandardScaler().fit(Train_X)
        Train_X = scalar.transform(Train_X)
        return Train_X

    def valid_test_slice(Test_X, Test_Y):
        test_size = 1 / 3
        ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        for train_index, test_index in ss.split(Test_X, Test_Y):
            X_valid, X_test = Test_X[train_index], Test_X[test_index]
            Y_valid, Y_test = Test_Y[train_index], Test_Y[test_index]
            return X_valid, Y_valid, X_test, Y_test


    data = capture_mat()
    train = slice_enc(data)
    Train_X, Train_Y = add_labels(train)
    Train_Y = one_hot(Train_Y)
    Train_X = np.asarray(Train_X)


    if normal:
        Train_X = scalar_stand(Train_X)

    if property == 'Train':
        Train_X1, Train_Y, Valid_X1, Valid_Y = valid_test_slice(Train_X, Train_Y)
        return Train_X1, Train_Y, Valid_X1, Valid_Y

    if property == 'Train_only':
        return Train_X, Train_Y

    if property == 'Test':
        return Train_X, Train_Y


if __name__ == "__main__":
    path = '../data/0.1HP-1800'
    np.random.seed(42)  # 设置随机种子
    snr = 2
    train_X, train_Y, valid_X, valid_Y = prepro(d_path=path,
                                length=1024,
                                number=750,
                                normal=True,
                                enc=True,
                                enc_step=28,
                                snr=snr,
                                property='Train',
                                noise=False
                                )

    test_X, test_Y = prepro(d_path=path,
                             length=2048,
                             number=10,
                             normal=True,
                             enc=True,
                             enc_step=28,
                             snr=snr,
                             property='Test',
                             noise=True
                            )

    # savemat('../data/0.1HP-1800_mat/data.mat',{'train_X': train_X,
    #                                            'train_Y': train_Y,
    #                                            'test_X': test_X,
    #                                            'test_Y': test_Y})
    # random_seed(42)
    train_X = np.array(test_X)
    # # train_X, valid_X, test_X = train_X[:, np.newaxis, :], valid_X[:, np.newaxis, :], test_X[:, np.newaxis, :]
    #
    # # # print(train_X[1].max,train_X[1].min)
    # # print(train_X.max(),train_X.min())
    # # # print(train_X)
    # # print(train_X.shape)
    # # print(train_Y.shape)
    #
    # for i in range(10):
    #     # if(train_X[i].max()<8 and train_X[i].min()>-8):
    #     filename = f'train/train_X{i}_{train_Y[i]}.txt'  # 文件名
    #     np.savetxt(filename, train_X[i], fmt='%8f')  # 保存train_X的第i个数组
    #     # train_X1 = np.loadtxt(filename)
    #     # filename = f'train_Y_{i}.txt'  # 文件名
    #     # np.savetxt(filename, train_Y[i], fmt='%d')  # 保存train_Y的第i个数组
    #
    # # pass

    test_X = np.array(test_X)


    for i in range(len(test_X)):
        # if(train_X[i].max()<8 and train_X[i].min()>-8):
        #     filename = f'test_0.1HP/test_X{i}_{test_Y[i]}.txt'  # 文件名
        filename = f'test1/test_X{i}_{test_Y[i]}'+ str(snr) +'.txt'  # 文件名
        np.savetxt(filename, test_X[i], fmt='%8f')  # 保存train_X的第i个数组
        # train_X1 = np.loadtxt(filename)
        # filename = f'train_Y_{i}.txt'  # 文件名
        # np.savetxt(filename, train_Y[i], fmt='%d')  # 保存train_Y的第i个数组

    pass

