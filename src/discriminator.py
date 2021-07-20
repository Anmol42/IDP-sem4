## discriminator
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator,self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=4,stride=4),
            nn.ReLU(),
            nn.Conv2d(16,64,kernel_size=4,stride=4),
            nn.ReLU(),
            nn.Conv2d(64,128,kernel_size=4,stride=4),
            nn.ReLU(),
            nn.Conv2d(128,64,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64,16,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16,4,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(4,1,kernel_size=1),
            nn.Sigmoid(),
            nn.Flatten()
        )
    def forward(self,x):
      return self.network(x)
