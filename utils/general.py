from os import makedirs
from os.path import join
import json
from datetime import datetime


def get_timestamp():
    ISOTIMEFORMAT = '%Y%m%d_%H%M%S_%f'
    timestamp = '{}'.format(datetime.utcnow().strftime(ISOTIMEFORMAT)[:-3])
    return timestamp


def get_result_path(dataset_name,
                    task_arch,
                    seed,
                    result_folder_name,
                    result_path='log/'):
    makedirs(result_path, exist_ok=True)
    timestamp = get_timestamp()
    model_id = f'{timestamp}_{dataset_name}_{task_arch}_{seed}'
    model_path = join(result_path, result_folder_name, model_id)
    makedirs(model_path, exist_ok=True)
    return model_path


def args2json(args, path, filename='arguments.json'):
    json_path = join(path, filename)
    with open(json_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
