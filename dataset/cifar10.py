from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torchvision
import torch

DOWNLOAD = False
DATA_ROOT = 'FILL_THIS'

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
])


class CIFAR10Adv(CIFAR10):
    def __getitem__(self, index: int):
        img, target = super().__getitem__(index)
        if self.sup:
            # 0 for non-living, 1 for living
            # 0: airplane, automobile, ship, truck
            # 1: bird, cat, deer, dog, frog, horse
            target = 0 if target in [0, 1, 8, 9] else 1
        elif self.sub:
            pass
        else:
            target = torch.tensor(
                [0 if target in [0, 1, 8, 9] else 1, target])

        return img, target


def gen(sup, sub):
    d_train = CIFAR10Adv(DATA_ROOT,
                         train=True,
                         transform=train_transform,
                         download=DOWNLOAD)
    d_train.sup = sup
    d_train.sub = sub
    d_test = CIFAR10Adv(DATA_ROOT,
                        train=False,
                        transform=train_transform,
                        download=DOWNLOAD)
    d_test.sup = sup
    d_test.sub = sub

    def label_divider(y):
        if sup or sub:
            return (y, None)
        else:
            return (y[:, 0], y[:, 1])

    adv_task_nc = 0
    if sup:
        task_nc = 1
    elif sub:
        task_nc = 10
    else:
        task_nc, adv_task_nc = 1, 10
    return d_train, d_test, label_divider, task_nc, adv_task_nc
