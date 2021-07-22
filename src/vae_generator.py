## generator model
class generator(nn.Module):
    def __init__(self):         # z is input noise
        super(generator,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1,8,kernel_size=3,padding=1,stride=2),
                                   nn.BatchNorm2d(8),
        nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(8,16,kernel_size=5,padding=2,stride=2),
                                   nn.BatchNorm2d(16),
        nn.LeakyReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(16,32,kernel_size=3,padding=1,stride=2),
                                   nn.BatchNorm2d(32),
        nn.LeakyReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(32,64,kernel_size=5,padding=2,stride=2),    #size is 8x8 at this point
        nn.LeakyReLU())
        # BottleNeck
        self.bottleneck = nn.Sequential(nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1),
                                        nn.LeakyReLU())     # size 4x4
        self.deconv7 = nn.Sequential(nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1),
        nn.ReLU())
        self.deconv6 = nn.Sequential(nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1),
        nn.ReLU())
        self.deconv5 = nn.Sequential(nn.ConvTranspose2d(64,64,kernel_size=4,stride=2,padding=1),
        nn.ReLU())
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(64,32,kernel_size=4,stride=2,padding=1),
        nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(32,16,kernel_size=4,stride=2,padding=1),
        nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(16,8,kernel_size=4,stride=2,padding=1),
        nn.ReLU())
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(8,2,kernel_size=4,stride=2,padding=1),
        nn.Tanh())
        self.linear = nn.Linear(128*4*4,512)

    def forward(self,xb,z):
        out1 = self.conv1(xb)
        #print("out1 ",out1.shape)
        out2 = self.conv2(out1)
        #print("out 2",out2.shape)
        out3 = self.conv3(out2)
        #print("out3 ",out3.shape)
        out4 = self.conv4(out3)
        #print("after conv4",out4.shape)
        out5 = self.bottleneck(out4)
        out6 = out5.view(z.shape[0],-1)         # Replaced hardcoded batch_size
        #print("after applying view",out6.shape)
        out6 = self.linear(out6) #we get 256 size vector
        mu = out6[:,:256]
        sig = torch.abs(out6[:,256:])
        #print(out6.shape,mu.shape,sig.shape)
        noise = z*sig + mu
        #print("noise shape",noise.shape)

        out5 = self.deconv7(noise.unsqueeze(2).unsqueeze(2))
        #print("after deconv7",out5.shape)
        out5 = self.deconv6(out5)
        #print("after deconv3",out4.shape)
        out5 = self.deconv5(out5)
        out5 = self.deconv4(out5)
        out5 = self.deconv3(out5)
        out5 = self.deconv2(out5)
        out5 = self.deconv1(out5)
        #print("after deconv1",out5.shape)
        #out5 = self.deconv1(out5)
        #print("after deconv1",out4.shape)
        return torch.cat((xb,out5),1)

