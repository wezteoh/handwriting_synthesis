# import pytorch modules
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# 2-layer lstm with mixture of gaussian parameters as outputs
# with skip connections
class LSTMRandWriter(nn.Module):
    def __init__(self, cell_size=400, num_clusters=20):
        super(LSTMRandWriter, self).__init__()
        
        self.lstm = nn.LSTM(input_size = 3, hidden_size = cell_size,\
                                num_layers = 1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size = cell_size+3, \
                                hidden_size = cell_size,\
                                num_layers = 1, batch_first=True)
        self.linear1 = nn.Linear(cell_size*2,\
                                    1+ num_clusters*6)
        self.tanh = nn.Tanh()
        
    def forward(self, x, prev, prev2):
        h1, (h1_n, c1_n) = self.lstm(x, prev)
        
        x2 = torch.cat([h1, x], dim=-1) # skip connection
        h2, (h2_n, c2_n) = self.lstm2(x2, prev2)
        
        h = torch.cat([h1, h2], dim=-1) # skip connection
        params = self.linear1(h)

        
        mog_params = params.narrow(-1, 0, params.size()[-1]-1)
        pre_weights, mu_1, mu_2, log_sigma_1, log_sigma_2, pre_rho, pre_end = mog_params.chunk(6, dim=-1)
        weights = F.softmax(pre_weights*(1+bias), dim=-1)
        rho = self.tanh(pre_rho)
        end = F.sigmoid(params.narrow(-1, params.size()[-1]-1, 1))
        # 
        # weights = F.softmax(params.narrow(-1, 0, self.num_clusters), dim=-1)
        # 
        # # MoG parameters
        # mu_1 = params.narrow(-1, self.num_clusters, self.num_clusters)
        # mu_2 = params.narrow(-1, 2*self.num_clusters, self.num_clusters)
        # log_sigma_1 = params.narrow(-1, 3*self.num_clusters, self.num_clusters)
        # log_sigma_2 = params.narrow(-1, 4*self.num_clusters, self.num_clusters)
        # rho = self.tanh(params.narrow(-1, 5*self.num_clusters, \
        #                 self.num_clusters))
        # 
        # # Bernoulli parameter
        # end = F.sigmoid(params.narrow(-1, 6*self.num_clusters, 1))
        
        return end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2,\
            rho, (h1_n, c1_n), (h2_n, c2_n)
            
# attention window for handwriting synthesis
class Window(nn.Module):
    def __init__(self, padded_text_len, K):
        super(Window, self).__init__()
        self.linear = nn.Linear(cell_size, 3*K)
        self.padded_text_len = text_len
        
    def forward(self, x, kappa_old, onehots, text_lens):
        params = self.linear(x).exp()
        
        alpha, beta, pre_kappa = params.chunck(3, dim=-1)
        kappa = kappa_old + pre_kappa
        # alpha = params.narrow(-1, 0, K).exp()
        # beta = params.narrow(-1, K, K).exp()
        # kappa = kappa_old + params.narrow(-1, 2*K, K).exp() 
        
        indices = torch.from_numpy(np.array(range(self.padded_text_len + 1))).type(torch.FloatTensor)
        if cuda:
            indices = indices.cuda()
        indices = Variable(indices, requires_grad=False)
        gravity = -beta.unsqueeze(2)*(kappa.unsqueeze(2).repeat(1, 1, self.padded_text_len + 1)-indices)**2
        phi = (alpha.unsqueeze(2) * gravity.exp()).sum(dim=1)*(self.padded_text_len/text_lens)
        
        w = (phi.narrow(-1, 0, self.text_len).unsqueeze(2) * onehots).sum(dim=1) 
        return w, kappa, phi

class LSTM1(nn.Module):
    def __init__(self, padded_text_len, vocab_len, cell_size):
        super(LSTM1, self).__init__()
        self.lstm = nn.LSTMCell(input_size = 3 + vocab_len, hidden_size = cell_size)
        self.window = Window(padded_text_len)
        
    def forward(self, x, onehots, text_lens, w_old, kappa_old, prev):
        h1s = []
        ws = []
        phis = []
        for _ in range(x.size()[1]):

            cell_input = torch.cat([x.narrow(1,_,1).squeeze(1),w_old], dim=-1)
            prev = self.lstm(cell_input, prev)
            
            # attention window parameters
            w_old, kappa_old, old_phi = self.window(prev[0], kappa_old, onehots, text_lens)
            
            # concatenate for single pass through the next layer
            h1s.append(prev[0])
            ws.append(w_old)
        
        return torch.stack(ws, dim=0).permute(1,0,2), torch.stack(h1s, dim=0).permute(1,0,2), \
                prev, w_old, kappa_old, old_phi 
    
class LSTM2(nn.Module):
    def __init__(self, vocab_len, cell_size):
        super(LSTM2, self).__init__()
        self.lstm = nn.LSTM(input_size=3 + vocab_len + cell_size, 
                            hidden_size = cell_size, num_layers = 1, batch_first =True)
        
    def forward(self, x, ws, h1s, prev2):
        lstm_input = torch.cat([x,ws,h1s], -1)
        h2s, prev2 = self.lstm(lstm_input, prev2)
        return h2s, prev2

# 2-layer lstm with mixture of gaussian parameters as outputs
# with skip connections
class LSTMSynthesis(nn.Module):
    def __init__(self, padded_text_len, vocab_len, cell_size=400, num_clusters=20, K=10):
        super(LSTMSynthesis, self).__init__()
        self.lstm1 = LSTM1(padded_text_len, vocab_len, cell_size)
        self.lstm2 = LSTM2(vocab_len, cell_size)
        self.linear = nn.Linear(cell_size*2, 1+ num_clusters*6)
        self.tanh = nn.Tanh()
        
    def forward(self, x, onehots, text_lens, w_old, kappa_old, prev, prev2, bias=0.):
        ws, h1s, prev, w_old, kappa_old, old_phi = self.lstm1(x, onehots, text_lens, w_old, kappa_old, prev)
        h2s, prev2 = self.lstm2(x, ws, h1s, prev2)
        
        params = self.linear(torch.cat([h1s,h2s], dim=-1))
        mog_params = params.narrow(-1, 0, params.size()[-1]-1)
        pre_weights, mu_1, mu_2, log_sigma_1, log_sigma_2, pre_rho, pre_end = mog_params.chunk(6, dim=-1)
        weights = F.softmax(pre_weights*(1+bias), dim=-1)
        rho = self.tanh(pre_rho)
        end = F.sigmoid(params.narrow(-1, params.size()[-1]-1, 1))
        
        # weights = F.softmax(params.narrow(-1, 0, num_clusters)*(1+bias), dim=-1)
        # mu_1 = params.narrow(-1, num_clusters, num_clusters)
        # mu_2 = params.narrow(-1, 2*num_clusters, num_clusters)
        # log_sigma_1 = params.narrow(-1, 3*num_clusters, num_clusters)
        # log_sigma_2 = params.narrow(-1, 4*num_clusters, num_clusters)
        # rho = self.tanh(params.narrow(-1, 5*num_clusters, num_clusters))
        # end = F.sigmoid(params.narrow(-1, 6*num_clusters, 1))
        
        return end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, rho, w_old, kappa_old, prev, prev2, old_phi
