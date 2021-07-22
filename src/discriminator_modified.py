## discriminator
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator,self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3,8,kernel_size=3,stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(8,16,kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(16,32,kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten()
        )
        self.linear1 = nn.Linear(64*25,128)
        self.linear2 = nn.Linear(128,1)
    def forward(self,x):
        out = self.network(x)
        out = self.linear1(out)
        out = self.linear2(out)
        out = torch.sigmoid(out)
        return out
