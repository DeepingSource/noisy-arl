import torch
from torch.nn import Module, ModuleList, Sequential, Flatten, Linear, ReLU, Identity, AvgPool2d
from torchvision.models import resnet18
from models.resnet_cifar import ResNet18 as resnet18_cifar


class SplitResNet18(Module):
    def __init__(self, nc, start, end, std=None, is_cifar=False):
        super(SplitResNet18, self).__init__()
        self.nc, self.start, self.end = nc, start, end
        if is_cifar:
            new_net = resnet18_cifar(num_classes=nc)
            self.model = ModuleList([
                new_net.conv1,
                new_net.bn1,
                ReLU(),
                Identity(),  # No maxpool for madry resnet
                new_net.layer1,
                new_net.layer2,
                new_net.layer3,
                new_net.layer4,
                AvgPool2d(4),
                Sequential(Flatten(), new_net.linear)
            ])[start:end+1]
        else:
            new_net = resnet18(num_classes=nc)
            new_net.fc = Sequential(Flatten(), new_net.fc)

            self.model = ModuleList(new_net.children())[start:end+1]

        print(self.model)

        self.freq = None
        self.std = std
        if self.std is not None:
            print(f'NOISE will be used : {self.std}')

    def forward(self, x, end=-1):
        for i, m in enumerate(self.model):
            x = m(x)
            if i == end:
                break
        if self.std is not None:
            x = x + torch.randn_like(x) * self.std
        return x

    def load_state_dict(self, state_dict, strict=True, orig_dict=False):
        if orig_dict:
            new_net = resnet18(num_classes=self.nc)
            new_net.load_state_dict(state_dict, strict)
            new_net.fc = Sequential(Flatten(), new_net.fc)
            self.model = ModuleList(new_net.children())[self.start:self.end]
        else:
            super().load_state_dict(state_dict, strict)
