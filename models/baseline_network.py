import torch
import torch.nn as nn
from core.base_network import BaseNetwork

class BaselineNetwork(BaseNetwork):
    def __init__(self, init_type='kaiming', gain=0.02, **kwargs):
        super(BaselineNetwork, self).__init__(init_type=init_type, gain=gain)
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def init_weights(self, init_type='normal', gain=0.02):
        pass

    def set_loss(self, loss_fn):
        pass

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        pass

    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8, adjust=False):
        # Just return the conditional image (input) as the result
        # Because the dataloader already upsampled it using Bicubic
        # We return y_cond as the final output, and a list containing it for visuals/intermediates
        return y_cond, [y_cond]

    def forward(self, y_0, y_cond=None, mask=None, noise=None):
        return torch.tensor(0.0, device=y_0.device, requires_grad=True)
