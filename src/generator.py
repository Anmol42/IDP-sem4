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
        self.bottleneck = nn.Sequential(nn.Conv2d(32,32,kernel_size=5,padding=2,stride=2),
        nn.LeakyReLU())
        # BottleNeck
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(34,32,kernel_size=4,stride=2,padding=1),
        nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(32,16,kernel_size=4,stride=2,padding=1),
        nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(16,8,kernel_size=4,stride=2,padding=1),
        nn.ReLU())
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(8,2,kernel_size=4,stride=2,padding=1),
        nn.Tanh())

    def forward(self,xb,z):
        out1 = self.conv1(xb)
        #print("out1 ",out1.shape)
        out2 = self.conv2(out1)
        #print("out 2",out2.shape)
        out3 = self.conv3(out2)
        #print("out3 ",out3.shape)
        out4 = self.bottleneck(out3)
        #print("after bottleneck",out4.shape)
        out4 = torch.cat((z,out4),1)
        out4 = self.deconv4(out4)
        #print("after deconv4",out4.shape)
        out4 = self.deconv3(out4)
        #print("after deconv3",out4.shape)
        out4 = self.deconv2(out4)
        #print("after deconv2",out4.hape)
        out4 = self.deconv1(out4)
        #print("after deconv1",out4.shape)
        return torch.cat((xb,out4),1)
