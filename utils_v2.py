import random
from torch.autograd import Variable
import torch
# from visdom import Visdom
import torchvision.transforms as transforms
import numpy as np
from skimage.filters import threshold_otsu


to_pil = transforms.ToPILImage()
to_gray = transforms.Grayscale(num_output_channels=1)

class QueueMask():
    def __init__(self, length):
        self.max_length = length
        self.queue = []

    def insert(self, mask):
        if self.queue.__len__() >= self.max_length:
            self.queue.pop(0)

        self.queue.append(mask)

    def multi_insert(self,masks):
        for mask in masks:
            self.insert(mask)
            
    def rand_item(self):
        assert self.queue.__len__() > 0, 'Error! Empty queue!'
        return self.queue[np.random.randint(0, self.queue.__len__())]

    def last_item(self):
        assert self.queue.__len__() > 0, 'Error! Empty queue!'
        return self.queue[self.queue.__len__()-1]

    def get_masks(self,batch_size):
        masks = []
        for _ in range(batch_size):
            masks.append(self.rand_item())
        return torch.cat(masks)



class mask_generator_fac:
    def __init__(self, norm_mean = np.array([0.5,0.5,0.5]), norm_std = np.array([0.5,0.5,0.5])):
        self.norm_mean = torch.tensor(np.reshape(norm_mean,(3,1,1))).cuda()
        self.norm_std = torch.tensor(np.reshape(norm_std,(3,1,1))).cuda()
    def __call__(self, shadow, shadow_free):
        norm_mean = self.norm_mean
        norm_std = self.norm_std
        assert shadow.size()[0]==shadow_free.size()[0]
        # ps(shadow,"shadow")
        # ps(shadow_free,"shadow_free")
        masks=[]
        for i in range(shadow.size()[0]):
            sf = shadow_free.data[i]
            s = shadow.data[i]
            im_f = to_gray(to_pil((sf*norm_std + norm_mean).detach().cpu()))
            im_s = to_gray(to_pil((s*norm_std + norm_mean).detach().cpu()))

            diff = (np.asarray(im_f, dtype='float32')- np.asarray(im_s, dtype='float32')) # difference between shadow image and shadow_free image
            L = threshold_otsu(diff)
            mask = torch.tensor((np.float32(diff >= L)-0.5)/0.5).unsqueeze(0).unsqueeze(0) #-1.0:non-shadow, 1.0:shadow
            mask.requires_grad = False
            masks.append(mask)
        return masks    

def ps(t,name):
    print(f"{name}_size: {t.size()}")

def psn(t,name):
    print(f"{name}_size: {t.size}")

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


