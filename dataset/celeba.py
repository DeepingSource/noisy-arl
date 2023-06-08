from torchvision.datasets import CelebA
import torchvision.transforms as transforms


CELEBA_CLASS_NAMES = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
                      'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
                      'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache',
                      'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
                      'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young', '']

DOWNLOAD = False
DATA_ROOT = 'FILL_THIS'


train_transform = transforms.Compose([
    transforms.CenterCrop((178, 178)),
    transforms.ColorJitter(0.1, 0.1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.CenterCrop((178, 178)),
    transforms.ToTensor()
])


def gen(class_names, adv_class_names):
    def label_divider(y): return (y, None)
    data_train = CelebA(DATA_ROOT,
                        'train',
                        transform=train_transform,
                        download=DOWNLOAD)

    data_test = CelebA(DATA_ROOT,
                       'valid',
                       transform=test_transform,
                       download=DOWNLOAD)

    task_nc = len(class_names)
    adv_task_nc = len(adv_class_names)
    if task_nc > 0 and adv_task_nc == 0:
        for cn in class_names:
            assert cn in CELEBA_CLASS_NAMES
        class_idx = sorted([CELEBA_CLASS_NAMES.index(c)
                            for c in class_names])
        data_train.attr = data_train.attr[:, class_idx]
        data_train.attr_names = [
            data_train.attr_names[i] for i in class_idx]
        data_test.attr = data_test.attr[:, class_idx]
        data_test.attr_names = [data_test.attr_names[i] for i in class_idx]

    elif task_nc > 0 and adv_task_nc > 0:
        for cn in class_names:
            assert cn in CELEBA_CLASS_NAMES
        for cn in adv_class_names:
            assert cn in CELEBA_CLASS_NAMES
        class_idx = sorted([CELEBA_CLASS_NAMES.index(c)
                            for c in class_names])
        adv_class_idx = sorted([CELEBA_CLASS_NAMES.index(c)
                                for c in adv_class_names])
        data_train.attr = data_train.attr[:, class_idx + adv_class_idx]
        data_train.attr_names = [data_train.attr_names[i]
                                 for i in class_idx + adv_class_idx]
        data_test.attr = data_test.attr[:, class_idx + adv_class_idx]
        data_test.attr_names = [data_test.attr_names[i]
                                for i in class_idx + adv_class_idx]

        def label_divider(y): return (y[:, :task_nc], y[:, task_nc:])
    return data_train, data_test, label_divider, task_nc, adv_task_nc
