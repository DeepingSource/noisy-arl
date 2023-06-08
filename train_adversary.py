from os import makedirs
from os.path import join
import sys

import shutil
import random
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import Adam

from tqdm import tqdm

from dataset.general import get_dataset
from models.general import get_classification_model, get_obfuscator_model
from utils.logger import get_logger
from utils.general import get_result_path, args2json, AverageMeter
from utils.metrics import get_metrics
from utils.loss import get_loss_fn
from validate import validate


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Adversary Classifier")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--dataset")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--print-freq", default=200, type=int)

    # Task network
    parser.add_argument("--task-arch")
    parser.add_argument("--task-loss")
    parser.add_argument("--task-lr", type=float, default=1e-3)
    parser.add_argument('--task-metrics', type=str, nargs="+")
    # Obfuscator
    parser.add_argument("--obfuscator-arch")
    parser.add_argument("--obfuscator-lr", type=float, default=1e-3)
    parser.add_argument('--obfuscator-weight')
    parser.add_argument('--std', type=float)

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    # Setting the seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Result path
    result_path = get_result_path(dataset_name=args.dataset,
                                  task_arch=args.task_arch,
                                  seed=args.seed,
                                  result_folder_name='train_adversary')

    # Logging
    logger = get_logger(result_path)

    python_version = sys.version.replace('\n', ' ')
    logger.info(f"Python version : {python_version}")
    logger.info(f"Torch version : {torch.__version__}")
    logger.info(f"Cudnn version : {torch.backends.cudnn.version()}")
    logger.info(f"Model Path : {result_path}")

    state = {k: v for k, v in args._get_kwargs()}
    for key, value in state.items():
        logger.info("{} : {}".format(key, value))

    # Save the arguments
    args2json(args, result_path)

    # Save this file
    file_save_path = join(result_path, 'code')
    makedirs(file_save_path, exist_ok=True)
    shutil.copy(sys.argv[0], join(file_save_path, sys.argv[0]))

    # Load Data
    data_train, data_test, label_divider, task_nc, _ = get_dataset(
        dataset_name=args.dataset)
    dataloader_train = DataLoader(data_train,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.workers)
    dataloader_test = DataLoader(data_test,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.workers)

    # Models
    task_net = get_classification_model(
        args.task_arch, num_classes=task_nc).to(device)
    obfuscator_net = get_obfuscator_model(
        args.obfuscator_arch, std=args.std).to(device)
    obfuscator_net.load_state_dict(torch.load(args.obfuscator_weight)[
                                   'state_dict_obfuscator'])
    obfuscator_net.eval()

    # Task Loss
    task_loss_fn = get_loss_fn(args.task_loss)

    # Optimizer
    optimizer_task = Adam(task_net.parameters(),
                          args.task_lr, betas=(0.9, 0.999))

    # Metrics
    metrics_train = get_metrics(args.task_metrics)
    metrics_test = get_metrics(args.task_metrics)

    # AverageMeters
    loss_avgm = AverageMeter()

    # Initialize best accuracy
    best_acc = 0
    for epoch in range(args.epochs):
        task_net.train()
        loss_avgm.reset()
        metrics_train.reset()

        for batch_idx, (x, y) in tqdm(enumerate(dataloader_train), total=len(dataloader_train),
                                      dynamic_ncols=True):
            x = x.to(device)
            y, _ = label_divider(y)
            y = y.to(device)

            with torch.no_grad():
                x = obfuscator_net(x)

            # Feed into net
            pred = task_net(x)

            # Task loss
            loss = task_loss_fn(pred, y)

            # Update
            loss_avgm.update(loss.item(), x.size(0))
            metrics_train.update(pred, y)

            # Optimization
            optimizer_task.zero_grad()
            loss.backward()
            optimizer_task.step()

            if (batch_idx + 1) % args.print_freq == 0:
                logger.info('+++Train+++ Epoch: {} LR: {lr:.5f}\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\n'
                            'Metrics: {metrics.val}\n\t({metrics.avg})'.format(
                                epoch,
                                lr=optimizer_task.param_groups[0]["lr"],
                                loss=loss_avgm,
                                metrics=metrics_train))

        # Validation after every epoch
        validate(val_loader=dataloader_test,
                 device=device,
                 task_net=task_net,
                 metrics=metrics_test,
                 obfuscator_net=obfuscator_net,
                 is_task=True,
                 label_divider=label_divider)
        logger.info(('+++Test+++ Epoch: {} Metrics: {metrics.avg}'
                     .format(epoch, metrics=metrics_test)))

        main_metric_val = metrics_test.get_main_metric()
        if main_metric_val > best_acc:
            best_acc = main_metric_val
            # Store best checkpoint
            save_path = join(result_path, 'checkpoint_best.pth')
            torch.save(
                {
                    'state_dict_task': task_net.state_dict(),
                    'optimizer_task': optimizer_task.state_dict(),
                }, save_path)
            logger.info(f'Best acc renewed: {best_acc}')

        # Store checkpoint
        save_path = join(result_path, 'checkpoint.pth')
        torch.save(
            {
                'state_dict_task': task_net.state_dict(),
                'optimizer_task': optimizer_task.state_dict(),
            }, save_path)


if __name__ == "__main__":
    main()
