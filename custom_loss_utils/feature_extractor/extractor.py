from torchvision import models
from collections import namedtuple
import torch
import torch.nn as nn
from custom_loss_utils.feature_extractor.hook import hook_fn_fac

class Extractor_VGG19BN:
    def __init__(self, cuda=True):
        self.feature={}
        self.model = models.vgg19_bn(pretrained=True)
        if cuda:
            self.model.cuda()
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad=False
        for layer_id in range(len(self.model.features)):
            self.model.features[layer_id].register_forward_hook(hook_fn_fac(self.feature,layer_id))

    def __getitem__(self,layer_id):
        if layer_id not in self.feature.keys():
            print("cannot find this feature")
            return self.feature[0]
        return self.feature[layer_id] 

    def __call__(self,x):
        self.feature["input"] = x
        self.model(x)

class Extractor_VGG16BN:
    def __init__(self, cuda=True):
        self.feature={}
        self.model = models.vgg16_bn(pretrained=True)
        if cuda:
            self.model.cuda()
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad=False
        for layer_id in range(len(self.model.features)):
            self.model.features[layer_id].register_forward_hook(hook_fn_fac(self.feature,layer_id))

    def __getitem__(self,layer_id):
        if layer_id not in self.feature.keys():
            print(f"cannot find this feature: {layer_id}")
            return self.feature[0].cuda()
        return self.feature[layer_id].cuda()

    def __call__(self,x):
        self.feature["input"] = x
        self.model(x)
        
