from log_utils import timeit

class LossFormat:
    def __init__(self,name,func,weight) -> None:
        self.name = name
        self.func = func
        self.weight = weight
        self.temp = 0
        self.hist = []
    
    #@timeit
    def calc_loss(self,img1,img2):
        self.temp = self.func(img1,img2)*self.weight
        self.hist.append(self.temp)
        return self.temp
        
    def __repr__(self):
        return self.name