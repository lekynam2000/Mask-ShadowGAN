from functools import reduce
import torch
from custom_loss_utils.loss_utils import gram_matrix, gauss_filter

mse_loss = torch.nn.MSELoss()

class content_loss_factory:
    def __init__(self,extractor,layers=[19]):
        self.extractor = extractor
        self.layers = layers
    def __call__(self,img1,img2):
        #Remember to call the extractor in advance
        batch_size = img1.shape[0]
        should_be_input = torch.cat((img1,img2))
        if not torch.equal(should_be_input,self.extractor["input"]):
            print(f"Input mismatch at {self.__name__}:")
            print(f'should be input: {should_be_input}')
            print(f'extractor: {self.extractor["input"]}')
        loss = 0
        for layer in self.layers:
            stacked_feat = self.extractor[layer]
            img1_feat = stacked_feat[:batch_size//2]
            img2_feat = stacked_feat[batch_size//2:]
            loss += mse_loss(img1_feat,img2_feat)

        loss /= len(self.layers)
        return loss 

class color_loss_factory:
    def __init__(self):
        self.filter = gauss_filter(21,3,3).cuda()

    def __call__(self,img1,img2):
        img1_blur = self.filter(img1)
        img2_blur = self.filter(img2)
        return mse_loss(img1_blur,img2_blur)

class style_loss_factory:
    def __init__(self,extractor,layers=[35]):
        self.extractor = extractor
        self.layers = layers
    def __call__(self,img1,img2):
        #Remember to call the extractor in advance
        batch_size = img1.shape[0]
        should_be_input = torch.cat((img1,img2))
        if not torch.equal(should_be_input,self.extractor["input"]):
            print(f"Input mismatch at {self.__name__}:")
            print(f'should be input: {should_be_input}')
            print(f'extractor: {self.extractor["input"]}')
        loss = 0
        for layer in self.layers:
            stacked_feat = self.extractor[layer]
            img1_feat = gram_matrix(stacked_feat[:batch_size//2])
            img2_feat = gram_matrix(stacked_feat[batch_size//2:])
            loss += mse_loss(img1_feat,img2_feat)

        loss /= len(self.layers)
        return loss 