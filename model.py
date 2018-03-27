# import pytorch modules
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# 2-layer lstm with mixture of gaussian parameters as outputs
# with skip connections
class LSTMRandWriter(nn.Module):
    def __init__(self, args):
        super(LSTMRandWriter, self).__init__()
        
        self.num_clusters = args.num_clusters
        self.cell_size = args.cell_size
        
        self.lstm = nn.LSTM(input_size = 3, hidden_size = self.cell_size,\
                                num_layers = 1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size = self.cell_size+3, \
                                hidden_size = self.cell_size,\
                                num_layers = 1, batch_first=True)
        self.linear1 = nn.Linear(self.cell_size*2,\
                                    1+ self.num_clusters*6)
        self.tanh = nn.Tanh()
        
    def forward(self, x, prev, prev2):
        h1, (h1_n, c1_n) = self.lstm(x, prev)
        
        x2 = torch.cat([h1, x], dim=-1) # skip connection
        h2, (h2_n, c2_n) = self.lstm2(x2, prev2)
        
        h = torch.cat([h1, h2], dim=-1) # skip connection
        params = self.linear1(h)
        weights = F.softmax(params.narrow(-1, 0, self.num_clusters), dim=-1)
        
        # MoG parameters
        mu_1 = params.narrow(-1, self.num_clusters, self.num_clusters)
        mu_2 = params.narrow(-1, 2*self.num_clusters, self.num_clusters)
        log_sigma_1 = params.narrow(-1, 3*self.num_clusters, self.num_clusters)
        log_sigma_2 = params.narrow(-1, 4*self.num_clusters, self.num_clusters)
        rho = self.tanh(params.narrow(-1, 5*self.num_clusters, \
                        self.num_clusters))
        
        # Bernoulli parameter
        end = F.sigmoid(params.narrow(-1, 6*self.num_clusters, 1))
        
        return end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2,\
            rho, (h1_n, c1_n), (h2_n, c2_n)
            
    
    
