import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary
import torch
from utils import ps

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator_S2F(nn.Module):
    def __init__(self, input_nc, output_nc, norm_mean,norm_std,n_residual_blocks=9):
        super(Generator_S2F, self).__init__()
        self.a = torch.tensor(-norm_mean/norm_std).cuda()
        temp_b = torch.tensor((1.0-norm_mean)/norm_std).cuda()
        self.b = (temp_b-self.a)/2.0
        self.a = self.a.unsqueeze(1).unsqueeze(1)
        self.b = self.b.unsqueeze(1).unsqueeze(1)
        print(f"self.a: {self.a}")
        print(f"self.b: {self.b}")

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7) ]
                    #nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def normalize(self,x):
        #Normalize (-1,1) to the distribution of input image
        return self.a + (x+1)*self.b

    def forward(self, x):
        y = (self.model(x) + x).tanh()
        # ps(y,"y")
        # ps(self.a,"self.a")
        # ps(self.b,"self.b")
        return self.normalize(y) 


class Generator_F2S(nn.Module):
    def __init__(self, input_nc, output_nc, norm_mean,norm_std, n_residual_blocks=9):
        super(Generator_F2S, self).__init__()

        self.a = torch.tensor(-norm_mean/norm_std).cuda()
        temp_b = torch.tensor((1.0-norm_mean)/norm_std).cuda()
        self.b = (temp_b-self.a)/2.0
        self.a = self.a.unsqueeze(1).unsqueeze(1)
        self.b = self.b.unsqueeze(1).unsqueeze(1)


        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc+1, 64, 7), # + mask
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7) ]
                    #nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def normalize(self,x):
        #Normalize (-1,1) to the distribution of input image
        return self.a + (x+1)*self.b

    def forward(self, x, mask):
        y = (self.model(torch.cat((x, mask.cuda()), 1)) + x).tanh()
        # ps(y,"y")
        # ps(self.a,"self.a")
        # ps(self.b,"self.b")
        return self.normalize(y) 


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1) #global avg pool


#
# net = Generator(3,3).cuda()
# summary(net, input_size=(3, 256, 256))

# dtype = torch.float
# #device = torch.deviec("cpu")
# device = torch.device("cuda:0")
#
# x = torch.randn(5, 3, 256, 256, device = device, dtype = dtype)
# out = net(x)
# print(out.shape)