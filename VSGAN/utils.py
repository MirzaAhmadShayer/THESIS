import torch
import numpy as np
import random
import torch.autograd as autograd

IMAGE_Size = 200 #change 256

def calc_gradient_penalty(netD, real_data, fake_data):
    BATCH_SIZE = 1 #change 1
    use_cuda = False #True
    
    alpha = torch.rand(BATCH_SIZE, 1)
    _,_,h,w = real_data.shape
    alpha = alpha.expand(BATCH_SIZE, real_data.nelement()//BATCH_SIZE).contiguous().view(BATCH_SIZE, 3, h, w)
    alpha = alpha.cpu() #.cuda()
    
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def add_noise(img,label):
    a = [0,99,100,255]
    x,y,w,h = [int(i.item()) for i in label]
    for i in range(y,y+h):
        for j in range(x,x+w):
            value = [random.sample(a,1)[0],random.sample(a,1)[0],random.sample(a,1)[0]]
            img[i,j,:] = value
    return img

def noise_img(image, label, idx, manner='not normal'):
    '''
    in: PIL image
    labels: x,y,w,h
    '''  
    img = np.array(image)[...,:3]
    img_noise = img
    if manner == 'normal':
        x,y,w,h = [int(i.item()) for i in label[idx]]
        img_noise[y:y+h,x:x+w,:] = 255.* np.random.random(img_noise[y:y+h,x:x+w,:].shape)
    else:
        label = label[idx]
        img_noise = add_noise(img_noise, label)
    return img_noise

def crop_car(img, idx):
    x = int(idx[0].item())
    y = int(idx[1].item())
    w = int(idx[2].item())
    h = int(idx[3].item())
    cropped_car = img[:,:,y:y+h,x:x+w]
    return cropped_car

def weights_init_normal(m):
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
   
def Gen_img(img, label, fixed_size=IMAGE_Size):
    '''
    img: PIL image
    label: x,y,w,h   Tensor
    return:
        416*416 image and label        
    '''
    img = np.array(img)[...,:3]
    img_w = img.shape[1]
    img_h = img.shape[0]
    x,y,w,h = [i.item() for i in label]
    x_ = random.randint(max(0,x-(fixed_size-w) ),min(x,img_w-fixed_size ))
    y_ = random.randint(max(0,y-(fixed_size-h) ),min(y,img_h-fixed_size ))
    w_, h_ = fixed_size, fixed_size
    label_x = x - x_
    label_y = y - y_
    cropped_label = [x_,y_,w_,h_]
    label_res = torch.Tensor([label_x,label_y,w,h])
    cropped_img = img[y_:y_+h_,x_:x_+w_,:]
    return cropped_img, label_res, cropped_label

def pair_data_gen(image, label, img_size=IMAGE_Size, manner='not normal'):  
    img = np.array(image)[...,:3]
    noise = noise_img(image,label,0,manner)
    cropped_noise, label, cropped_label = Gen_img(noise,label[0],fixed_size=img_size)
    x,y,w,h = cropped_label
    ori_img = img[y:y+h,x:x+w,:]
    img_mask = np.zeros((ori_img.shape[0], ori_img.shape[1]*2,3),dtype=np.uint8)
    img_mask[:,0:ori_img.shape[0],:] = ori_img
    img_mask[:,ori_img.shape[0]:,:] = cropped_noise
    return img_mask, label   