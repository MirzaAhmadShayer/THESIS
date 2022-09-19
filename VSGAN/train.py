import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from datasets import ListDataset
from utils import weights_init_normal, calc_gradient_penalty
from models import Generator, D_b, D_V

Dv_Loss = []
Db_Loss = []
G_Loss = []
Real_scores = []
Fake_scores = []
#Hyperparameters
Number_Of_Epochs = 3
Num_Of_Rows = 8
IMG_Shape = 200 #300, also wont work <=64 works at 128,160,200 higher->more power, time
Learning_Rate = 0.0002
Beta_1 = 0.5
Beta_2 = 0.999
Lambda_L1 = 100
BATCH_Size_DL = 1
BATCH_Size_DLV = 32

# Create samples
os.makedirs('images', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#criteron
criterion_MSE = torch.nn.MSELoss().to(device)
criterion_L1 = torch.nn.L1Loss().to(device)
criterion_BCE = torch.nn.BCELoss().to(device)

#Setting G, D_b, D_v
Generator = Generator(conv_dim=64, repeat_num=8).to(device)
D_vehicle = D_V(conv_dim=64, repeat_num=4).to(device)
D_backgroud = D_b(in_channels=6, use_sigmoid=False).to(device)

#Applying the weights to the G, D_b, D_v
Generator.apply(weights_init_normal)
D_backgroud.apply(weights_init_normal)
D_vehicle.apply(weights_init_normal)

#Adding optimizer to D_b, D_v, G
optimizer_G = torch.optim.Adam(Generator.parameters(),lr=Learning_Rate,betas=(Beta_1, Beta_2))
optimizer_D_b = torch.optim.Adam(D_backgroud.parameters(),lr=Learning_Rate,betas=(Beta_1, Beta_2))
optimizer_D_v = torch.optim.Adam(D_vehicle.parameters(),lr=Learning_Rate,betas=(Beta_1, Beta_2))

# Path of dataset text files
#train,test works only train, train and train test only!!!
train_root = r'D:/Thesis/My_Code_VSGAN/Training.txt'
test_root = r'D:/Thesis/My_Code_VSGAN/Training.txt'
#train_root = r'D:/Thesis/My_Code_VSGAN/Training.txt'
#test_root = r'D:/Thesis/My_Code_VSGAN/Testing.txt'
    
# Image transformations
transforms_ = [transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]

datasets = ListDataset(train_root, img_shape=IMG_Shape, transform=transforms_, train = True)
dataloader = torch.utils.data.DataLoader(datasets,batch_size=BATCH_Size_DL,shuffle=True)
    
datasets_val = ListDataset(test_root, img_shape=IMG_Shape,transform=transforms_)
dataloader_val = torch.utils.data.DataLoader(datasets_val,batch_size=BATCH_Size_DLV,shuffle=False)

def test_G(epoch):
    imgs = next(iter(dataloader_val))
    A = imgs['A'].to(device)
    B = imgs['B'].to(device)
    Generator.eval()
    gen_image = Generator(A)
    end = torch.cat((A,gen_image),0)
    save_image(end, 'images/%s.jpg' % (epoch), nrow=Num_Of_Rows, normalize=True)
    
# Loss weight for gradient penalty
lambda_gp = 1.

for epoch in range(0, Number_Of_Epochs):
    for i, img in enumerate(dataloader):
        A = img['A'].to(device)  
        B = img['B'].to(device)
        region = img['region']
        [x1,y1,x2,y2] = [i.item() for i in region]
        gen_image = Generator(A)
        car_crop_real = A[:,:,y1:y2,x1:x2]
        car_crop_fake = gen_image[:,:,y1:y2,x1:x2]

        #train D_v
        optimizer_D_v.zero_grad()
        real_v = D_vehicle(car_crop_real)        
        fake_v = D_vehicle(car_crop_fake.detach())     
        gradient_penalty = calc_gradient_penalty(D_vehicle, car_crop_real.data, car_crop_fake.data)
        dv_loss = -torch.mean(real_v) + torch.mean(fake_v) + lambda_gp * gradient_penalty
        dv_loss.backward(retain_graph=True)
        optimizer_D_v.step()
        
        # train D_b
        optimizer_D_b.zero_grad()
        real_AB = torch.cat((B, A),1)
        fake_AB = torch.cat((B, gen_image),1)  
        D_b_real = D_backgroud(real_AB)
        D_b_fake = D_backgroud(fake_AB.detach())   
        real_label = torch.ones_like(D_b_real)
        fake_label = torch.zeros_like(D_b_fake)
        loss_D_1 = criterion_MSE( (D_b_real), real_label)
        loss_D_2 = criterion_MSE( (D_b_fake), fake_label)
        loss_D_b = (loss_D_1 + loss_D_2) * 0.5
        loss_D_b.backward()
        optimizer_D_b.step()

        #Generator_vehicle
        optimizer_G.zero_grad()     
        fake_car_ = car_crop_fake
        output_v = D_vehicle(fake_car_)
        loss_1 = -torch.mean(output_v)
        
        #Generator_backgroud
        fake_AB = torch.cat((B , gen_image),1)
        output_b = D_backgroud(fake_AB)
        label_b1 = torch.ones_like(output_b)
        loss_2 = criterion_MSE(output_b,label_b1)
        loss_l1 = criterion_L1(gen_image, A)
        loss_g = loss_1 + loss_2 + Lambda_L1*loss_l1
        loss_g.backward()
        optimizer_G.step()
        
        if(i % 100 == 0):
         print("[Epoch %d/%d] [Batch %d/%d] [D_v loss: %f] [D_b loss: %f] [G loss: %f]" % (epoch, Number_Of_Epochs, i, len(dataloader),
                                                            dv_loss.item(), loss_D_b.item(),loss_g.item() ))
         Dv_Loss.append(dv_loss.item())
         Db_Loss.append(loss_D_b.item())
         G_Loss.append(loss_g.item())
         Real_scores.append(round(torch.mean(real_label).item()))
         Fake_scores.append(round(torch.mean(fake_label).item()))
    test_G(epoch)

plt.figure(figsize=(10,5))
plt.title("Generator and Discremenators Losses during training")
plt.plot(Dv_Loss, label="Dv")
plt.plot(Db_Loss, label="Db")
plt.plot(G_Loss, label="G")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
print('Real scores:',Real_scores)
print('Fake scores:',Fake_scores)