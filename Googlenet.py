import torch
import torch.nn as nn

#定义conv-bn-relu函数
def conv_relu(in_channel, out_channel, kernel, stride=1, padding=0):
    conv = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel, stride, padding),
        nn.BatchNorm2d(out_channel, eps=1e-3),
        nn.ReLU(True),
    )
    return conv

#定义incepion结构，见inception图
class inception(nn.Module):
    def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_5,
                 out4_1):
        super(inception, self).__init__()
        self.branch1 = conv_relu(in_channel, out1_1, 1)
        self.branch2 = nn.Sequential(
            conv_relu(in_channel, out2_1, 1),
            conv_relu(out2_1, out2_3, 3, padding=1))
        self.branch3 = nn.Sequential(
            conv_relu(in_channel, out3_1, 1),
            conv_relu(out3_1, out3_5, 5, padding=2))
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            conv_relu(in_channel, out4_1, 1),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        output = torch.cat([b1, b2, b3, b4], dim=1)
        return output

# 堆叠GOOGLENET，见上表所示结构
class GOOGLENET(nn.Module):
    def __init__(self):
        super(GOOGLENET, self).__init__()
        self.features = nn.Sequential(
            conv_relu(3, 64, 7, 2, 3),
            nn.MaxPool2d(3, stride=2, padding=0),
            conv_relu(64, 64, 1),
            conv_relu(64, 192, 3, padding=1),
            nn.MaxPool2d(3, 2),
            inception(192, 64, 96, 128, 16, 32, 32),
            inception(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, stride=2),
            inception(480, 192, 96, 208, 16, 48, 64),
            inception(512, 160, 112, 224, 24, 64, 64),
            inception(512, 128, 128, 256, 24, 64, 64),
            inception(512, 112, 144, 288, 32, 64, 64),
            inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, 2),
            inception(832, 256, 160, 320, 32, 128, 128),
            inception(832, 384, 182, 384, 48, 128, 128),
            nn.AvgPool2d(2))
        self.classifier = nn.Sequential(
            nn.Linear(9216,1024),
            nn.Dropout2d(p=0.4),
            nn.Linear(1024, 10))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out
