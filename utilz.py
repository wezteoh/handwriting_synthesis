import torch
import os

def decay_learning_rate(optimizer, decay_rate):
    # learning rate annealing
    state_dict = optimizer.state_dict()
    lr = state_dict['param_groups'][0]['lr']
    lr *= decay_rate
    for param_group in state_dict['param_groups']:
        param_group['lr'] = lr
    optimizer.load_state_dict(state_dict)
    return optimizer

def save_checkpoint(epoch, model, validation_loss, optimizer, directory, \
                    filename='best.pt'):
    checkpoint=({'epoch': epoch+1,
    'model': model.state_dict(),
    'validation_loss': validation_loss,
    'optimizer' : optimizer.state_dict()
    })
    try:
        torch.save(checkpoint, os.path.join(directory, filename))
        
    except:
        os.mkdir(directory)
        torch.save(checkpoint, os.path.join(directory, filename))