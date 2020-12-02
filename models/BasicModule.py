# coding: utf-8

import torch


class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = self.__class__.__name__

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name):
        torch.save(self.state_dict(), name)
        print('Model ' + name + ' has been saved!')
        return name
