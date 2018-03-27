import numpy as np
import torch
from torch.autograd import Variable
from utils2 import plot_stroke


def generate_unconditionally(model, cell_size=400, num_clusters=20, steps=800, random_seed=1, state_dict_file='save/rand_write_best.pt'):
    
    model.load_state_dict(torch.load(state_dict_file)['model'])
    
    np.random.seed(random_seed)
    zero_tensor = torch.zeros((1,1,3))
    
    # initialize null hidden states and memory states
    init_states = [torch.zeros((1,1, cell_size))]*4
    if cuda:
        zero_tensor = zero_tensor.cuda()
        init_states  = [state.cuda() for state in init_states]
    x = Variable(zero_tensor)
    init_states  = [Variable(state, requires_grad = False) for state in init_states]
    h1_init, c1_init, h2_init, c2_init = init_states
    prev = (h1_init, c1_init)
    prev2 = (h2_init, c2_init)
    
    record = [np.array([0,0,0])]

    for i in range(steps):        
        end, weights, mu_1, mu_2, log_sigma_1, log_sigma_2, p, prev, prev2 = model(x, prev, prev2)
        
        # sample end stroke indicator
        prob_end = end.data[0][0][0]        
        sample_end = np.random.binomial(1,prob_end)
        sample_index = np.random.choice(range(20),p = weights.data[0][0].cpu().numpy())
        
        # sample new stroke point
        mu = np.array([mu_1.data[0][0][sample_index], mu_2.data[0][0][sample_index]])
        v1 = log_sigma_1.exp().data[0][0][sample_index]**2
        v2 = log_sigma_2.exp().data[0][0][sample_index]**2
        c = p.data[0][0][sample_index]*log_sigma_1.exp().data[0][0][sample_index]\
            *log_sigma_2.exp().data[0][0][sample_index]
        cov = np.array([[v1,c],[c,v2]])
        sample_point = np.random.multivariate_normal(mu, cov)
        
        out = np.insert(sample_point,0,sample_end)
        record.append(out)
        x = torch.from_numpy(out).type(torch.FloatTensor)
        if cuda:
            x = x.cuda()
        x = Variable(x, requires_grad=False)
        x = x.view((1,1,3))
        
    return plot_stroke(np.array(record))