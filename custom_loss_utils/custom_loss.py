from functools import reduce
import torch
from custom_loss_utils.loss_utils import blur

class content_loss_factory:
    def __init__(self,model,layer='relu5_4'):
        self.model=model
        self.layer=layer
    def __call__(self,img1,img2):
        img1_feat = getattr(self.model(img1),self.layer)
        img2_feat = getattr(self.model(img2),self.layer)
        #content_size = reduce(lambda a,b: a*b, img1_feat.size())
        return torch.nn.MSELoss(img1_feat,img2_feat)

class color_loss_factory:
    def __init__(self):
        pass
    def __call__(self,img1,img2):
        img1_blur = blur(img1)
        img2_blur = blur(img2)
        return torch.nn.MSELoss(img1_blur,img2_blur)

class style_loss_factory:
    def __init__(self,model):
        self.model = model
    def __call__(self,img1,img2):
        pass