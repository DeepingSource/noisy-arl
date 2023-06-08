from tqdm import tqdm
from torch import no_grad


@no_grad()
def validate(val_loader, device, task_net, metrics, obfuscator_net=None, is_task=True,
             label_divider=None):
    metrics.reset()
    if obfuscator_net:
        obfuscator_net.eval()
    task_net.eval()
    
    for _, (x, y) in tqdm(enumerate(val_loader), total=len(val_loader), dynamic_ncols=True):
        x = x.to(device)
        y = y.to(device)
        if is_task:
            y = label_divider(y)[0]
        else:
            y = label_divider(y)[1]

        # compute output
        if obfuscator_net:
            x = obfuscator_net(x)
        output = task_net(x)
        # Update
        metrics.update(output, y)
