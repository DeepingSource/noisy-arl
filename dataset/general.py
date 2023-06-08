def get_dataset(dataset_name):
    def label_divider(y):
        return (y, None)

    if 'cifar10' in dataset_name:
        from dataset.cifar10 import gen
        sup = '_sup' in dataset_name
        sub = '_sub' in dataset_name
        data_train,data_test, label_divider, task_nc, adv_task_nc = gen(sup, sub)

    elif 'celeba' in dataset_name:
        from dataset.celeba import gen
        adv = []
        if '_' in dataset_name:
            task = dataset_name.split('_')[1:]
        else:
            task = ['Smiling']
            adv = ['Male']
        data_train, data_test, label_divider, task_nc, adv_task_nc = gen(
            task, adv)

    elif 'fairface' in dataset_name:
        from dataset.fairface import gen
        data_train, data_test, label_divider, task_nc, adv_task_nc = gen(
            dataset_name)
    else:
        raise NotImplementedError
    return data_train, data_test, label_divider, task_nc, adv_task_nc
