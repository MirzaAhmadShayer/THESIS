import torch
import math
import torch.nn as nn

def spatial_pyramid_pool(previous_conv, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''    
    h,w = previous_conv.size(2),previous_conv.size(3)
    num_sample = 1 # change 1 if necessary
    for i in range(len(out_pool_size)):
        h_wid = int(math.ceil(h / out_pool_size[i]))
        w_wid = int(math.ceil(w / out_pool_size[i]))
        h_pad = int((h_wid*out_pool_size[i] - h + 1)/2)
        w_pad = int((w_wid*out_pool_size[i] - w + 1)/2)
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        if(i == 0):
            spp = x.view(num_sample,-1)
        else:
            spp = torch.cat((spp,x.view(num_sample,-1)), 1)
    return spp

def spatial_pyramid_pool2(previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''    
    for i in range(len(out_pool_size)):
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_pad = int((h_wid*out_pool_size[i] - previous_conv_size[0] + 1)/2)
        w_pad = int((w_wid*out_pool_size[i] - previous_conv_size[1] + 1)/2)
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        if(i == 0):
            spp = x.view(num_sample,-1)
        else:
            spp = torch.cat((spp,x.view(num_sample,-1)), 1)
    return spp

x=  torch.randn(1,1,8,6)
y = spatial_pyramid_pool2(x,1,[8,6],[4,2,1])