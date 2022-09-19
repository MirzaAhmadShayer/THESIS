import torch
from torch import nn
from spp import spatial_pyramid_pool

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)
    
class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, repeat_num=2):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.ReflectionPad2d(3))
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, padding=0))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2
            
        layers.append(nn.ReflectionPad2d(3))
        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, padding=0))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        # Replicate spatially and concatenate domain information.
        return self.main(x) 

class D_V(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self,conv_dim=64, repeat_num=4):
        super(D_V, self).__init__()
        self.output_num = [4,2,1]
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=3, stride=1, padding=0))
        layers.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim
        for i in range(0, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=1, padding=1))
          
            layers.append(nn.BatchNorm2d(curr_dim*2))
            layers.append(nn.LeakyReLU(0.2))
            curr_dim = curr_dim * 2

        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=0, bias=False)
      
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
       # print(out_src.shape)
        out_src = spatial_pyramid_pool(out_src, self.output_num)
        return out_src

class D_b(nn.Module):
    '''
    args:
        input:
            GT + Noise: 6 channel img or Generator + Noise 6 channel img
        output:
            patch GAN
    '''
    
    def __init__(self, in_channels=6, use_sigmoid=False):
        super(D_b, self).__init__()
        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
            return layers
        
        layers = []
        in_filters = in_channels
        
        for out_filters, stride, normalize in [ (64, 1, False),
                                                (64, 2, True),
                                                (128, 1, True),
                                                (128, 2, True),
                                                (256, 1, True),
                                                (256, 2, True),
                                                (512, 1, True),
                                                (512, 2, False),]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        # Output layer
        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)
        
    def forward(self, img):
        return self.model(img)

if __name__=="__main__":
    
    x = torch.randn(1,3,300,300)
    b = Generator(conv_dim=64, repeat_num=8)
    y = b(x)
    print(y.shape)   