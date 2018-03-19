
import numpy as np
import matplotlib
import sys
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

# find gpu
cuda = torch.cuda.is_available()

# hyperparamters
timesteps = 8
vtimesteps=1200
num_clusters = 20
cell_size = 4
nlayers = 2
bsize = 2
init_lr = 1E-3
text_len = 65
K = 10 # num of attention clusters
vocab_len = 60

# prepare training data
train_data = [np.load('train_strokes_800.npy'), np.load('train_masks_800.npy'), np.load('train_onehot_800.npy')]
for _ in range(len(train_data)):
    train_data[_] =torch.from_numpy(train_data[_]).type(torch.FloatTensor)
    if cuda:
        train_data[_] = train_data[_].cuda()
train_data = [(train_data[0][i], train_data[1][i], train_data[2][i]) for i in range(len(train_data[0]))] 
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=bsize, shuffle=True, drop_last=True)
    
# prepare validation data
validation_data = [np.load('validation_strokes_800.npy'), np.load('validation_masks_800.npy'), 
                   np.load('validation_onehot_800.npy')]
for _ in range(len(validation_data)):
    validation_data[_] = torch.from_numpy(validation_data[_]).type(torch.FloatTensor)
    if cuda:
        validation_data[_] = validation_data[_].cuda()
validation_data = [(validation_data[0][i], validation_data[1][i], validation_data[2][i]) 
                   for i in range(len(validation_data[0]))] 
validation_loader = torch.utils.data.DataLoader(
    validation_data, batch_size=bsize, shuffle=True, drop_last=True)
    
class Window(nn.Module):
    def __init__(self):
        super(Window, self).__init__()
        self.linear = nn.Linear(cell_size, 3*K)
        
    def forward(self, x, kappa_old, text_len, onehots):
        params = self.linear(x)
        alpha = params.narrow(-1, 0, K).exp()
        beta = params.narrow(-1, K, K).exp()
        kappa = kappa_old + params.narrow(-1, 2*K, K).exp() 
        
        #phi = alpha*(())
        indices = torch.from_numpy(np.array(range(text_len))).type(torch.FloatTensor)
        if cuda:
            indices = indices.cuda()
        indices = Variable(indices, requires_grad=False)
        gravity = -beta.unsqueeze(2)*(kappa.unsqueeze(2).repeat(1, 1, text_len)-indices)**2
        phi = (alpha.unsqueeze(2) * gravity.exp()).sum(dim=1)
        print(phi.unsqueeze(2).size())
        print(onehots.size())
        weight = (phi.unsqueeze(2) * onehots).sum(dim=1) 
        return weight, kappa
        
class LSTM1(nn.Module):
    def __init__(self):
        super(LSTM1, self).__init__()
        self.lstm = nn.LSTMCell(input_size = 3 + vocab_len, hidden_size = cell_size)
        self.window = Window()
        
    def forward(self, x, onehots, w_old, kappa_old, prev):
        h1s = []
        ws = []
        for _ in range(x.size()[1]):
            cell_input = torch.cat([x.narrow(1,_,1).squeeze(),w_old], dim=-1)
            h1, prev = self.lstm(cell_input, prev)
            w_old, kappa_old = self.window(h1, kappa_old, text_len, onehots)
            h1s.append(h1)
            ws.append(w_old)
        
        return torch.stack(ws, dim=0).permute(2,1,0), torch.stack(h1s, dim=0).permute(2,1,0) 
        
l = LSTM1()
l = l.cuda()

for a, b in enumerate(train_loader):
    data = b[0].narrow(1,0,timesteps)
    onehots = b[2]
    break
    
# debug
w = Variable(torch.zeros(bsize,60).cuda())
x = Variable(onehots)
d = Variable(data)
k_old = Variable(torch.zeros(bsize, K).cuda())
h1_init, c1_init = torch.zeros((bsize,cell_size)).cuda(), torch.zeros((bsize,cell_size)).cuda()
h1_init, c1_init = Variable(h1_init), Variable(c1_init)
print(l(d, x, w, k_old, (h1_init, c1_init)))
