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
    parser = argparse.ArgumentParser(description="Noisy ARL")
    # General
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
    # Proxy adversarial task network
    parser.add_argument("--adv-task-arch")
    parser.add_argument("--adv-task-loss")
    parser.add_argument("--adv-task-lr", type=float, default=1e-3)
    parser.add_argument('--adv-task-metrics', type=str, nargs="+")
    parser.add_argument("--adv-loss-weight", type=float, default=1e-2)
    # Obfuscator
    parser.add_argument("--obfuscator-arch")
    parser.add_argument("--obfuscator-lr", type=float, default=1e-3)
    parser.add_argument('--std', type=float)

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    # Fix the seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Result path
    result_path = get_result_path(dataset_name=args.dataset,
                                  task_arch=args.task_arch,
                                  seed=args.seed,
                                  result_folder_name='train_obfuscator')

    # Logging
    logger = get_logger(result_path)

    python_version = sys.version.replace('\n', ' ')
    logger.info(f"Python version : {python_version}")
    logger.info(f"Torch version : {torch.__version__}")
    logger.info(f"Cudnn version : {torch.backends.cudnn.version()}")
    logger.info(f"Model Path : {result_path}")

    state = {k: v for k, v in args._get_kwargs()}
    for key, value in state.items():
        logger.info(f"{key} : {value}")

    # Save the arguments
    args2json(args, result_path)

    # Save this training script
    file_save_path = join(result_path, 'code')
    makedirs(file_save_path, exist_ok=True)
    shutil.copy(sys.argv[0], join(file_save_path, sys.argv[0]))

    # Data
    data_train, data_test, label_divider, task_nc, adv_task_nc = get_dataset(
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
    adv_task_net = get_classification_model(
        args.adv_task_arch, num_classes=adv_task_nc).to(device)
    obfuscator_net = get_obfuscator_model(
        args.obfuscator_arch, std=args.std).to(device)

    # Loss
    task_loss_fn = get_loss_fn(args.task_loss)
    adv_task_loss_fn = get_loss_fn(args.adv_task_loss)

    # Optimizer
    optimizer_task = Adam(task_net.parameters(),
                          args.task_lr, betas=(0.9, 0.999))
    optimizer_adv_task = Adam(adv_task_net.parameters(), args.adv_task_lr,
                              betas=(0.9, 0.999))
    optimizer_obfuscator = Adam(obfuscator_net.parameters(),
                                args.obfuscator_lr, betas=(0.9, 0.999))

    # Metrics
    metrics_train = get_metrics(args.task_metrics)
    metrics_test = get_metrics(args.task_metrics)
    metrics_train_adv = get_metrics(args.adv_task_metrics)
    metrics_test_adv = get_metrics(args.adv_task_metrics)

    # AverageMeters
    loss_avgm = AverageMeter()
    task_loss_avgm = AverageMeter()
    acc_avgm = AverageMeter()
    adv_task_loss_avgm = AverageMeter()
    adv_acc_avgm = AverageMeter()

    # Initialize best accuracy
    best_acc = 0
    for epoch in range(args.epochs):
        loss_avgm.reset()
        task_loss_avgm.reset()
        acc_avgm.reset()
        adv_task_loss_avgm.reset()
        adv_acc_avgm.reset()

        task_net.train()
        adv_task_net.train()
        obfuscator_net.train()

        loss_avgm.reset()
        metrics_train.reset()
        metrics_train_adv.reset()

        for batch_idx, (x, y) in tqdm(enumerate(dataloader_train), total=len(dataloader_train),
                                      dynamic_ncols=True):
            x = x.to(device)
            y, y_adv = label_divider(y)
            y = y.to(device)
            y_adv = y_adv.to(device)

            # Run through obfusctor
            obf = obfuscator_net(x)

            # Run through task network
            pred = task_net(obf)
            pred_adv = adv_task_net(obf)

            # Loss
            task_loss = task_loss_fn(pred, y)
            adv_task_loss = adv_task_loss_fn(pred_adv, y_adv)
            loss = task_loss - args.adv_loss_weight * adv_task_loss

            # Optimization
            optimizer_obfuscator.zero_grad()
            optimizer_task.zero_grad()
            optimizer_adv_task.zero_grad()
            loss.backward()
            optimizer_obfuscator.step()
            optimizer_task.step()
            for p in adv_task_net.parameters():
                p.grad /= -args.adv_loss_weight
            optimizer_adv_task.step()

            # Update AverageMeters
            loss_avgm.update(loss.item(), x.size(0))
            task_loss_avgm.update(task_loss.item(), x.size(0))
            metrics_train.update(pred, y)

            adv_task_loss_avgm.update(adv_task_loss.item(), x.size(0))
            metrics_train_adv.update(pred_adv, y_adv)

            if (batch_idx + 1) % args.print_freq == 0:
                msg = f'+++Train+++ Epoch: {epoch:03d}\t'
                # LR
                lr_task = optimizer_task.param_groups[0]["lr"]
                lr_obf = optimizer_obfuscator.param_groups[0]["lr"]
                msg += f'LR Task {lr_task:5f} Obfuscator {lr_obf:.5f} '
                lr_adv_task = optimizer_adv_task.param_groups[0]["lr"]
                msg += f'Adv task {lr_adv_task:.5f} '
                msg += '\n'
                # Loss
                loss = loss_avgm
                loss_task = task_loss_avgm
                msg += f'Total Loss {loss.val:.4f} ({loss.avg:.4f}) '
                msg += f'Task {loss_task.val:.4f} ({loss_task.avg:.4f}) '
                if adv_task_net:
                    loss_adv_task = adv_task_loss_avgm
                    msg += f'Adv task {loss_adv_task.val:.4f} ({loss_adv_task.avg:.4f}) '
                msg += '\n'
                # Metrics
                metrics = metrics_train
                msg += f'Metrics: {metrics.val} ({metrics.avg}) '
                metrics_adv = metrics_train_adv
                msg += f'Adv metrics: {metrics_adv.val} ({metrics_adv.avg})'
                msg += '\n'

                # Print
                logger.info(msg)

        # Eval after each epoch
        validate(val_loader=dataloader_test,
                 device=device,
                 task_net=task_net,
                 metrics=metrics_test,
                 obfuscator_net=obfuscator_net,
                 is_task=True,
                 label_divider=label_divider)
        logger.info(f'+++Test+++ Epoch: {epoch} Metrics: {metrics_test.avg}')
        validate(val_loader=dataloader_test,
                 device=device,
                 task_net=adv_task_net,
                 metrics=metrics_test_adv,
                 obfuscator_net=obfuscator_net,
                 is_task=False,
                 label_divider=label_divider)
        logger.info(
            f'+++Test+++ Epoch: {epoch} Adv Metrics: {metrics_test_adv.avg}')

        main_metric_val = metrics_test.get_main_metric()

        if main_metric_val > best_acc:
            best_acc = main_metric_val
            # Store best checkpoint
            save_path = join(result_path, 'checkpoint_best.pth')
            torch.save(
                {
                    'state_dict_obfuscator': obfuscator_net.state_dict(),
                    'state_dict_task': task_net.state_dict(),
                    'optimizer_obfuscator': optimizer_obfuscator.state_dict(),
                    'optimizer_task': optimizer_task.state_dict(),
                    'state_dict_adv': adv_task_net.state_dict(),
                    'optimizer_adv': optimizer_adv_task.state_dict()
                }, save_path)
            logger.info(f'Best acc renewed: {best_acc}')

        # Store checkpoint
        save_path = join(result_path, 'checkpoint.pth')
        torch.save(
            {
                'state_dict_obfuscator': obfuscator_net.state_dict(),
                'state_dict_task': task_net.state_dict(),
                'optimizer_obfuscator': optimizer_obfuscator.state_dict(),
                'optimizer_task': optimizer_task.state_dict(),
                'state_dict_adv': adv_task_net.state_dict(),
                'optimizer_adv': optimizer_adv_task.state_dict()
            }, save_path)

    logger.info(f"+++ Best Accuracy: {best_acc}")


if __name__ == "__main__":
    main()
