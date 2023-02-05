from functools import reduce
import torch
from custom_loss_utils.loss_utils import blur,gram_matrix

mse_loss = torch.nn.MSELoss()

class content_loss_factory:
    def __init__(self,model,layers=['relu5_4']):
        self.model = model
        self.layers = layers
    def __call__(self,img1,img2):
        loss = 0
        for layer in self.layers:
            img1_feat = getattr(self.model(img1),layer)
            img2_feat = getattr(self.model(img2),layer)
            loss += mse_loss(img1_feat,img2_feat)
        #content_size = reduce(lambda a,b: a*b, img1_feat.size())
        loss /= len(self.layers)
        return loss 

class color_loss_factory:
    def __init__(self):
        pass
    def __call__(self,img1,img2):
        img1_blur = blur(img1)
        img2_blur = blur(img2)
        return mse_loss(img1_blur,img2_blur)

class style_loss_factory:
    def __init__(self,model,layers=['relu5_4']):
        self.model = model
        self.layers = layers
    def __call__(self,img1,img2):
        loss = 0
        for layer in self.layers:
            gram1 = gram_matrix(img1)
            gram2 = gram_matrix(img2)
            loss += mse_loss(gram1,gram2)
        loss /= len(self.layers)
        return loss