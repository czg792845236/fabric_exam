import code.transforms as T
import code.utils as utils
from code.coco_utils import get_coco
import torch
import torchvision
import time
import datetime
from code.engine import train_one_epoch
import os


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# 设置数据集，图片存储路径和标注文件路径
def get_dataset(name, image_set, transform, data_path):
    paths = {
        "coco": (data_path, get_coco, 21),
    }
    DATA_DIR, _, num_classes = paths[name]

    # DATA_DIR:图片、标注存储根目录
    # coco_fabric_dataset：布匹数据、标注具体存储位置

    coco_fabric_dataset = {
        "img_dir": "coco/images/train",
        "ann_file": "coco/annotations/instances_train.json"
    }
    datesets = get_coco(DATA_DIR, image_set=image_set, data_set=coco_fabric_dataset, transforms=transform)

    return datesets, num_classes


BATCH_SIZE = 5
NUMBER_WORKERS = 4
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
LR_STEP = [8, 11]
LR_GAMMA = 0.1
EPOCH = 10
PRINT_FRQ = 20
OUTPUT_DIR = '/home/makefile/PycharmProjects/exam1/code'
if __name__ == '__main__':
    dataset, num_classes = get_dataset('coco', "train", get_transform(train=True),
                                       '/home/makefile/PycharmProjects/exam1/code')
    train_sampler = torch.utils.data.RandomSampler(dataset)
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, BATCH_SIZE, drop_last=True)

    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=train_batch_sampler, num_workers=NUMBER_WORKERS,
                                              collate_fn=utils.collate_fn)
    print('Creating model')
    model = torchvision.models.detection.__dict__['fasterrcnn_resnet50_fpn'](num_classes=num_classes, pretrained=False)
    # print(model)

    device1 = torch.device('cuda')
    device2 = torch.device('cpu')
    model.to(device1)

    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.Adam(params,lr = LEARNING_RATE)
    optimizer = torch.optim.SGD(
        params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_STEP, gamma=LR_GAMMA)
    print("Start training")
    start_time = time.time()
    for epoch in range(EPOCH):
        train_one_epoch(model, optimizer, data_loader, device1, epoch, PRINT_FRQ)
        utils.save_on_master({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict()},
            os.path.join(OUTPUT_DIR, 'model_{}.pth'.format(epoch)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
