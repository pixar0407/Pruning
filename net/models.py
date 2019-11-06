import torch.nn as nn
import torch.nn.functional as F
from .prune import PruningModule, MaskedLinear

# Depth Map Prediction 필요
import torch
import torch.nn as nn
from torchvision.models import vgg16

# 아래 코드는 우리 적용 대상인 Depth Map Prediction
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, batch):
        return batch.view([batch.shape[0], -1])


class Scale1_Linear(PruningModule):
    # input 512x7x10
    # output 64x15x20

    def __init__(self, mask=False):
        super(Scale1_Linear, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        if mask: #mask가 들어가는지 안들어가는지 확인하는 코드
            print(f'Mask is true')
        else:
            print(f'Mask is not true')

        self.block = nn.Sequential(
            Flatten(),
            linear(512 * 7 * 10, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            linear(4096, 64 * 15 * 20)
        )

    def forward(self, x):
        scale_1_op = torch.reshape(self.block(x), (x.shape[0], 64, 15, 20))
        return nn.functional.interpolate(scale_1_op, scale_factor=4, mode='bilinear', align_corners=True)


class Scale2(PruningModule):
    # input 64x60x80, 3x240x320
    # output 1x120x160

    def __init__(self):
        super(Scale2, self).__init__()
        self.input_img_proc = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=64 + 64, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5, padding=2)
        )

    def forward(self, x, input_img):
        proc_img = self.input_img_proc(input_img)
        concatenate_input = torch.cat((x, proc_img), dim=1)
        return nn.functional.interpolate(self.block(concatenate_input), scale_factor=2, mode='bilinear',
                                         align_corners=True)


class Scale3(PruningModule):
    # input 1x120x160, 3x240x320
    # output 1x120x160

    def __init__(self):
        super(Scale3, self).__init__()
        self.input_img_proc = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=65, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5, padding=2)
        )

    def forward(self, x, input_img):
        proc_img = self.input_img_proc(input_img)
        concatenate_input = torch.cat((x, proc_img), dim=1)
        return self.block(concatenate_input)


class Net(PruningModule):
    def __init__(self, mask=False):
        super(Net, self).__init__()
        self.VGG = nn.Sequential(*list(vgg16(pretrained=True).children())[0])
        self.Scale_1 = Scale1_Linear(mask)
        self.Scale_2 = Scale2()
        self.Scale_3 = Scale3()

    def forward(self, x):
        input_img = x.clone()  # 3x240x320
        x = self.VGG(x)  # 512x7x10
        x = self.Scale_1(x)  # 64x60x80
        x = self.Scale_2(x, input_img.clone())  # 1x120x160
        x = self.Scale_3(x, input_img.clone())  # 1x120x160
        return x

############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
# 이하 코드는 기존 song han의 예제인 LeNet
class LeNet(PruningModule):
    def __init__(self, mask=False):
        super(LeNet, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.fc1 = linear(784, 300)
        self.fc2 = linear(300, 100)
        self.fc3 = linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


class LeNet_5(PruningModule):
    def __init__(self, mask=False):
        super(LeNet_5, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.conv3 = nn.Conv2d(16, 120, kernel_size=(5,5))
        self.fc1 = linear(120, 84)
        self.fc2 = linear(84, 10)

    def forward(self, x):
        # Conv1
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # Conv2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # Conv3
        x = self.conv3(x)
        x = F.relu(x)

        # Fully-connected
        x = x.view(-1, 120)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x
