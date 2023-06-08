from torchvision.models import resnet18
from models.resnet_cifar import ResNet18 as ResNet18_CIFAR
from models.resnet_split import SplitResNet18
from models.vardecoder import VarDecoder


def get_classification_model(model_name, num_classes=1000):
    net = None

    if model_name.startswith('resnet18'):
        if model_name.endswith('_cifar'):
            net = ResNet18_CIFAR(num_classes=num_classes)
        else:
            net = resnet18(num_classes=num_classes)

    is_cifar = False
    if model_name.endswith('_cifar'):
        model_name = model_name[:-6]
        is_cifar = True
    if model_name.startswith('splitres18'):
        layer_idx = int(model_name.split('_')[1]) + 3  # block number + 3
        net = SplitResNet18(num_classes, layer_idx+1, 20, is_cifar=is_cifar)

    if net is None:
        raise NotImplementedError

    return net


def get_obfuscator_model(model_name, std):
    net = None

    is_cifar = False
    if model_name.endswith('_cifar'):
        model_name = model_name[:-6]
        is_cifar = True

    if model_name.startswith('splitres18'):
        layer_idx = int(model_name.split('_')[1]) + 3  # block number + 3
        net = SplitResNet18(1, 0, layer_idx, std, is_cifar=is_cifar)

    if net is None:
        raise NotImplementedError

    return net

def get_reconstructor_model(model_name):
    if model_name.startswith('splitres18'):
        layer_idx = int(model_name.split('_')[1]) + 3
        l = [64, 64, 64, 64, 64, 128, 256, 512]
        net = VarDecoder(l[layer_idx])
    else:
        raise NotImplementedError
    return net