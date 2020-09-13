#Lenet5

import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,6,5,1,2), #in_channels=1,out_channels=6,kernel_size=5,stride=1,padding=2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fc1=nn.Sequential(
            nn.Linear(16*5*5,120),
            nn.ReLU()
        )
        self.fc2=nn.Sequential(
            nn.Linear(120,84),
            nn.ReLU()
        )
        self.fc3=nn.Linear(84,10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        #nn.Linear()的输入输出都是维度为1的值,所以要把多维度的tensor展平为一维
        x=x.view(x.size()[0],-1)#-1表示第二个维度是根据第一个维度改变的,如果全部数据为100,第一个维度是2,则第二个维度为100/2=50
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        return x



