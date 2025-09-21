import torch.nn as nn

class TrainingAgent(nn.Module):

    def __init__(self, action_model, value_model, device='gpu'):
        super(TrainingAgent, self).__init__()
        self.action_model = action_model
        self.value_model = value_model
        self.device = device
        if device == 'gpu':
            self.action_model = self.action_model.cuda()
            self.value_model = self.value_model.cuda()