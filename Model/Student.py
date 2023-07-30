import time
import torch
import torch.nn as nn
import numpy as np
from fvcore.nn import flop_count_str, FlopCountAnalysis



class SCNN(nn.Module):
    """
    Student model builder
    """

    def __init__(self):
        super(SCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=64, stride=8, padding=28)

        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        # 卷积层
        x = abs(torch.fft.fft(x, dim=2, norm="forward"))
        _, x = x.chunk(2, dim=2)
        x1 = self.conv1(x)
        x2 = nn.functional.relu(x1)
        x3 = self.pool1(x2)
        x4 = x3.view(-1, 256)
        x5 = self.fc1(x4)
        return x5

if __name__ == '__main__':
    filename = f'../utils/Ftest/Ftest_X1000_[4].txt'
    N = 2048
    train_X0 = np.loadtxt(filename)
    train_X0 = train_X0.reshape((1, 1, 2048))
    X = torch.from_numpy(train_X0).float()
    m = SCNN()
    file = "../Pth/" + "DKD_fswdcnn0.10.1HP-18000.9645117714054185" + ".pth"
    m.load_state_dict(torch.load(file))
    start = time.time()
    output = m(X)
    stop = time.time()
    print(output)
    print(stop-start)
    print(flop_count_str(FlopCountAnalysis(m, X)))

