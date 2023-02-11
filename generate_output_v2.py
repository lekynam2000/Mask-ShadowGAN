#!/usr/bin/python3

import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from PIL import Image
import numpy as np

from models_guided import Generator_F2S, Generator_S2F
from utils import mask_generator
from utils import QueueMask

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=400, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='output/netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='output/netG_B2A.pth', help='B2A generator checkpoint file')
parser.add_argument('--generated_dir', type=str, default='../generated', help='B2A generator checkpoint file')
opt = parser.parse_args()

## ISTD
opt.dataroot_A = os.path.join(opt.dataroot,'test','test_A')
opt.dataroot_B = os.path.join(opt.dataroot,'test','test_C')

opt.im_suf_A = '.png'
opt.im_suf_B = '.png'

### SRD
# opt.dataroot_A = '/home/xwhu/dataset/SRD/test_data/shadow'
# opt.dataroot_B = '/home/xwhu/dataset/SRD/test_data/shadow_free'
#
# opt.im_suf_A = '.jpg'
# opt.im_suf_B = '.jpg'

### USR
# opt.dataroot_A = '/home/xwhu/dataset/shadow_USR/shadow_test'
# opt.dataroot_B = '/home/xwhu/dataset/shadow_USR/shadow_free'

# opt.im_suf_A = '.jpg'
# opt.im_suf_B = '.jpg'


if torch.cuda.is_available():
    opt.cuda = True
    torch.cuda.set_device(3)
    print(f"using GPU: {torch.cuda.current_device()}")

print(opt)


###### Definition of variables ######
# Networks
netG_A2B = Generator_S2F(opt.input_nc, opt.output_nc)
# netG_B2A = Generator_F2S(opt.output_nc, opt.input_nc)

if opt.cuda:
    netG_A2B.cuda()
    # netG_B2A.cuda()

# Load state dicts
netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
# netG_B2A.load_state_dict(torch.load(opt.generator_B2A))

# Set model's test mode
netG_A2B.eval()
# netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

# Dataset loader
norm_mean = np.array([0.485, 0.456, 0.406])
norm_std = np.array([0.229, 0.224, 0.225])
img_transform = transforms.Compose([
    transforms.Resize((int(opt.size),int(opt.size)), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean,norm_std)
])
#dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test'),
#                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
###################################
to_pil = transforms.ToPILImage()

###### Testing######

# Create output dirs if they don't exist
shadow_free_dir = os.path.join(opt.generated_dir,'shadow_free')
if not os.path.exists(shadow_free_dir):
    os.makedirs(shadow_free_dir)
# if not os.path.exists('output/recovered_shadow'):
#     os.makedirs('output/recovered_shadow')
# if not os.path.exists('output/same_A'):
#     os.makedirs('output/same_A')
# if not os.path.exists('output/recovered_shadow_free'):
#     os.makedirs('output/recovered_shadow_free')
# if not os.path.exists('output/same_B'):
#     os.makedirs('output/same_B')

##################################### A to B // shadow to shadow-free
gt_list = [os.path.splitext(f)[0] for f in os.listdir(opt.dataroot_A) if f.endswith(opt.im_suf_A)]

# mask_queue = QueueMask(gt_list.__len__())

mask_non_shadow = Variable(Tensor(1, 1, opt.size, opt.size).fill_(-1.0), requires_grad=False)

for idx, img_name in enumerate(gt_list):
    print('predicting: %d / %d' % (idx + 1, len(gt_list)))

    # Set model input
    img = Image.open(os.path.join(opt.dataroot_A, img_name + opt.im_suf_A)).convert('RGB')
    w, h = img.size

    img_var = (img_transform(img).unsqueeze(0)).cuda()

    # Generate output

    temp_B = netG_A2B(img_var)

    fake_B = norm_std*temp_B.data + norm_mean
    fake_B = np.array(transforms.Resize((h, w))(to_pil(fake_B.data.squeeze(0).cpu())))
    Image.fromarray(fake_B).save(os.path.join(shadow_free_dir,img_name + opt.im_suf_A))
    print('Generated images %04d of %04d' % (idx+1, len(gt_list)))


