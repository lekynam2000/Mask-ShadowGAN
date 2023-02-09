import matplotlib.pyplot as plt
import os
import numpy as np

plt.ioff()

def draw_loss(loss_array, name_array, iter_loss, output_dir, plt_name):
    plt.figure(figsize=(10, 5))
    plt.title(f"{plt_name} During Training")
    for loss,name in zip(loss_array,name_array):
        plt.plot(loss, label=name)
    plt.xlabel("iterations / %d" % (iter_loss))
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir,f"{plt_name}.png"))

class StackDrawer:
    def __init__(self,gamma,num_loss,iter_loss,output_dir,loss_name):
        self.gamma = gamma
        self.num_loss = num_loss
        self.loss_hist = [[] for _ in range(self.num_loss)]
        self.iter_loss = iter_loss
        self.output_dir = output_dir
        self.colors = ['red', 'orange', 'blue', 'black']
        self.loss_name = loss_name
    def update(self,loss_array):
        assert len(loss_array)==self.num_loss
        for i in range(self.num_loss):
            self.loss_hist[i].append(self.gamma[i]*loss_array[i].detach().cpu())
    
    def draw(self):
        assert self.num_loss<=len(self.colors)
        y = np.vstack([hist for hist in self.loss_hist])
        x = np.arange(0,len(self.loss_hist[0]),1)
        plt.figure(figsize=(10, 5))
        plt.title("Loss Component During Training")
        for i in range(self.num_loss):
            plt.plot([],[],color=self.colors[i],label=self.loss_name[i])
        plt.stackplot(x,*self.loss_hist,colors=self.colors)
        plt.xlabel("iterations / %d" % (self.iter_loss))
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir,f"Loss_component.png"))