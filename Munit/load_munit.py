from __future__ import print_function
from ast import Raise
from xml.parsers.expat import model
from .utils import get_config, get_data_loader_folder, pytorch03_to_pytorch04, load_inception
#from trainer import MUNIT_Trainer, UNIT_Trainer
from .trainer import Trainer

import numpy as np
import torchvision.utils as vutils
import sys
import torch
import os


# # car racing
# parser.add_argument('--config', type=str, default='configs/carroad2road_folder.yaml', help='Path to the config file.')
# parser.add_argument('--input_folder',default='./inputs/testA', type=str, help="input image folder")
# parser.add_argument('--output_folder', type=str, default='results', help="output image folder")
# parser.add_argument('--checkpoint', type=str, default='./outputs/carroad2road_folder/checkpoints/gen_00100000.pt',help="checkpoint of autoencoders")


# # rl env pong old: ball noise
# parser.add_argument('--config', type=str, default='configs/rl_env_old.yaml', help='Path to the config file.')
# parser.add_argument('--input_folder',default='./datasets/data_pong_old/testA', type=str, help="input image folder")
# parser.add_argument('--output_folder', type=str, default='results', help="output image folder")
# parser.add_argument('--checkpoint', type=str, default='./outputs/rl_env_old/checkpoints/gen_00050000.pt',help="checkpoint of autoencoders")

'''
# # rl env pong: rain noise
# parser.add_argument('--config', type=str, default='configs/rl_env.yaml', help='Path to the config file.')
# parser.add_argument('--input_folder',default='./datasets/data_pong/testA', type=str, help="input image folder")
# parser.add_argument('--output_folder', type=str, default='results', help="output image folder")
# parser.add_argument('--checkpoint', type=str, default='./outputs/rl_env/checkpoints/gen_00020000.pt',help="checkpoint of autoencoders")'''

# parser.add_argument('--a2b', type=int, help="1 for a2b and 0 for b2a", default=1)
# # parser.add_argument('--seed', type=int, default=1, help="random seed")
# # parser.add_argument('--num_style',type=int, default=10, help="number of styles to sample")
# # parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
# parser.add_argument('--inception_a', type=str, default='.', help="path to the pretrained inception network for domain A")
# parser.add_argument('--inception_b', type=str, default='.', help="path to the pretrained inception network for domain B")

# opts = parser.parse_args()

# # Load experiment setting
# config = get_config(opts.config)
# input_dim = config['input_dim_a'] if opts.a2b else config['input_dim_b']

from torchvision import transforms
from PIL import Image


class MunitInferModel:
    def __init__(self, encoder, decoder, style_dim, new_size, device):
        self.encoder = encoder
        self.decoder = decoder
        self.style_dim = style_dim
        self.new_size = new_size
        self.device = device

        self.transform = transforms.Compose(
            [transforms.Resize(new_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def infer(self, img):
        img = Image.fromarray(img)
        img = self.transform(img).unsqueeze(0).to(self.device)
        content, _ = self.encoder(img)
        # style = torch.randn(1, self.style_dim, 1, 1).cuda()

        outputs = self.decoder(content)
        outputs = (outputs + 1) / 2.
        grid = vutils.make_grid(outputs)
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        return ndarr



def load_munit_model(model_name, device):
    if model_name == 'Pong':
        config = get_config('./Munit/configs/Pong.yaml')
        model_path = './Munit/outputs/Pong/checkpoints/gen_00050000.pt'
    elif model_name == 'CarRacing':
        config = get_config('./Munit/configs/CarRacing.yaml')
        model_path = './Munit/outputs/CarRacing/checkpoints/gen_00100000.pt'
    else:
        raise RuntimeError('No model for.' + model_name)

    trainer = Trainer(config)
    style_dim = config['gen']['style_dim']
    new_size = config['new_size']
    try:
        state_dict = torch.load(model_path)
        trainer.gen_a.load_state_dict(state_dict['a'])
        trainer.gen_b.load_state_dict(state_dict['b'])
    except:
        state_dict = pytorch03_to_pytorch04(torch.load(model_path), 'MUNIT')
        trainer.gen_a.load_state_dict(state_dict['a'])
        trainer.gen_b.load_state_dict(state_dict['b'])
    
    device = torch.device(device)
    trainer.to(device)
    trainer.eval()
    encoder = trainer.gen_a.encode # if opts.a2b else trainer.gen_b.encode
    decoder = trainer.gen_b.decode # if opts.a2b else trainer.gen_a.decode

    return MunitInferModel(encoder, decoder, style_dim, new_size, device)

if __name__ == '__main__':
    model = load_munit_model('Pong')
    img = Image.open('new.jpg').convert('RGB')
    output = model.infer(img)
    print(output.shape)
    print(output)
    im = Image.fromarray(output)
    im.save('old.jpg')