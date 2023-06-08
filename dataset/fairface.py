import torchvision.transforms as transforms
from torch.utils.data import Dataset
from os.path import join
from pandas import read_csv
from PIL import Image
import torch

train_transform = transforms.Compose([
    transforms.Resize((178, 178)),  # 224^2 -> 178^2
    transforms.ColorJitter(0.1, 0.1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((178, 178)),
    transforms.ToTensor()
])

DATA_ROOT = 'FILL_THIS'


class FairFace(Dataset):
    def __init__(self, root, gender, race, age, agerace, raceage, is_train):
        super().__init__()
        self.root = join(root, 'fairface')
        self.gender_map = {'Female': 0, 'Male': 1}
        self.race_map = {'Black': 0,
                         'East Asian': 1,
                         'Indian': 2,
                         'Latino_Hispanic': 3,
                         'Middle Eastern': 4,
                         'Southeast Asian': 5,
                         'White': 6}
        self.age_map = {'0-2': 0,
                        '3-9': 1,
                        '10-19': 2,
                        '20-29': 3,
                        '30-39': 4,
                        '40-49': 5,
                        '50-59': 6,
                        '60-69': 7,
                        'more than 70': 8
                        }
        self.is_gender = gender
        self.is_race = race
        self.is_age = age
        self.is_agerace = agerace
        self.is_raceage = raceage
        self.transform = train_transform if is_train else test_transform
        self.labels = read_csv(
            join(self.root, f'fairface_label_{"train" if is_train else "val"}.csv')).values.tolist()

    def __getitem__(self, index):
        img_path, age, gender, race, _ = self.labels[index]
        img_path = join(self.root, img_path)
        img = self.transform(Image.open(img_path))
        gender = self.gender_map[gender]
        race = self.race_map[race]
        age = self.age_map[age]

        if self.is_gender:
            target = gender
        elif self.is_race:
            target = race
        elif self.is_age:
            target = age
        elif self.is_agerace:
            target = torch.tensor([age, race])
        elif self.is_raceage:
            target = torch.tensor([race, age])
        else:
            target = torch.tensor([gender, race])
        return img, target

    def __len__(self):
        return len(self.labels)


def gen(dataset_name):
    def label_divider(y): return (y, None)
    agerace = '-agerace' in dataset_name
    raceage = '-raceage' in dataset_name
    gender = dataset_name.endswith('-gender')
    race = dataset_name.endswith('-race')
    age = dataset_name.endswith('-age')

    data_train = FairFace(DATA_ROOT, gender, race,
                          age, agerace, raceage,  True)
    data_test = FairFace(DATA_ROOT, gender, race, age, agerace, raceage, False)

    task_nc, adv_task_nc = 1, 1
    if dataset_name == 'fairface-genderrace':
        task_nc, adv_task_nc = 1, 7
    elif dataset_name == 'fairface-agerace':
        task_nc, adv_task_nc = 9, 7
    elif dataset_name == 'fairface-raceage':
        task_nc, adv_task_nc = 7, 9
    elif dataset_name == 'fairface-gender':
        task_nc = 1
    elif dataset_name == 'fairface-race':
        task_nc, adv_task_nc = 7, 0
    elif dataset_name == 'fairface-age':
        task_nc, adv_task_nc = 9, 0

    def label_divider(y):
        if gender or race or age:
            return (y, None)
        else:
            return (y[:, 0], y[:, 1])
    return data_train, data_test, label_divider, task_nc, adv_task_nc
