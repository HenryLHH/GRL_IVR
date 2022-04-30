import torch
import yaml
import argparse

class Config:
    DEVICE = torch.device('cpu')

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_argument = self.parser.add_argument

        self.hyperparams = {}

    def load_hyperparams(self):
        args = self.parser.parse_args()
        
        if hasattr(args, 'hyperparams'):
            with open(args.hyperparams, 'r') as f:
                self.hyperparams = yaml.safe_load(f)

        for key in args.__dict__.keys():
            self.hyperparams['misc'][key] = args.__dict__[key]

    def select_device(self):
        cudaid = self.hyperparams['misc']['cudaid']
        if cudaid >= 0:
            Config.DEVICE = torch.device('cuda:%d' % (cudaid))
        else:
            Config.DEVICE = torch.device('cpu')