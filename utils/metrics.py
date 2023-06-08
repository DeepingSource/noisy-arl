
def get_metrics(metric_list):
    metrics = []
    for idx, m in enumerate(metric_list):
        if m == 'AccuracyBinary':
            metrics.append(AccuracyBinary())
        elif m == 'AccuracyBinaryMulti':
            metrics.append(AccuracyBinary(idx))
        elif m == 'Accuracy@1':
            metrics.append(Accuracy(top=1))
        elif m == 'Accuracy@5':
            metrics.append(Accuracy(top=5))
        else:
            raise ValueError
    return MetricCompose(metrics)


class MetricCompose():
    def __init__(self, metrics):
        self.metrics = metrics
        self.reset()

    def update(self, prob, target):
        for m in self.metrics:
            m.update(prob, target)
            self.val[m.name] = m.val
            self.avg[m.name] = m.avg

    def reset(self):
        self.val = {}
        self.avg = {}
        for m in self.metrics:
            m.reset()
            self.val[m.name] = 0
            self.avg[m.name] = 0

    def get_main_metric(self):
        # The first metric is always the main metric
        return self.metrics[0].avg


class AccuracyBinary():
    '''Can be used for single class classification.
    logit and target sizes are expected to be [batch_size, 1] of [batch_size]
    '''

    def __init__(self, index=None):
        self.reset()
        self.name = f'AccuracyBinary{"" if index is None else index}'
        self.index = index

    def update(self, logit, target):
        if self.index is not None:
            logit = logit[:, self.index]
            target = target[:, self.index]

        if len(logit.shape) == 1:
            logit = logit.unsqueeze(1)
        if len(target.shape) == 1:
            target = target.unsqueeze(1)

        num_samples = logit.size(0)
        self.cnt += num_samples
        correct = ((logit.sigmoid() > 0.5) == target).sum().item()
        self.correct += correct
        self.val = correct / num_samples
        self.avg = self.correct / self.cnt

    def reset(self):
        self.cnt = 0
        self.correct = 0
        self.val = 0
        self.avg = 0


class Accuracy():
    def __init__(self, top=1):
        self.reset()
        self.top = top
        self.name = 'Accuracy@{}'.format(top)

    def update(self, logit, target):
        num_samples = target.size(0)
        self.cnt += num_samples

        _, pred = logit.topk(self.top, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_k = correct[:self.top].reshape(-1).float().sum(0, keepdim=True)
        self.correct += correct_k
        self.val = float(correct_k) / float(num_samples)
        self.avg = float(self.correct) / float(self.cnt)

    def reset(self):
        self.cnt = 0
        self.correct = 0
        self.val = 0
        self.avg = 0
