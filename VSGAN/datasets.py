import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

IMAGE_Shape = 200 #change 256

class ListDataset(Dataset):
    '''
        input:im_path x y w h
        return:
            orignal_img
            noise_img
            labels: x,y,w,h
    '''
    def __init__(self,list_path, img_shape=IMAGE_Shape, transform=None, train=False):
        with open(list_path,'r') as file:
            files = file.readlines()
            self.num_samples = len(files)
        self.train = train
        files = [i.strip() for i in files]
        self.img_files = [i.split(' ')[0] for i in files]
        self.label_files = [i.strip().split(' ')[1:] for i in files]
        self.img_shape = img_shape
        self.transform = transforms.Compose(transform)
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        image_path = self.img_files[index % len(self.img_files)]
        #extract images
        image = Image.open(image_path)  # h w
            
        region = (0,0,self.img_shape,self.img_shape)
        noise_region = (self.img_shape, 0, self.img_shape*2, self.img_shape)
        original = image.crop(region)
        noise = image.crop(noise_region)
        
        #Label
        label_path = self.label_files[index % len(self.img_files)]
        label_path = list(map(float, label_path))
        labels = np.array(label_path).reshape(-1,4).astype('float64')
        x,y,w,h = [int(i) for i in labels[0]]
        region = [x,y,x+w,y+h]
        orig_img = self.transform(original)
        noise_img = self.transform(noise)
        return {'A':orig_img, 'B':noise_img, "region":region}
    
    def collate_fn(self, batch):
        orig_img = [x[0] for x in batch]
        noise_img = [x[1] for x in batch]
        region = [x[2] for x in batch]
        return orig_img, noise_img, region