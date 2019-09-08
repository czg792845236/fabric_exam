# inference.py
import os
import time
import datetime
import torch
import code.utils as utils
import glob
import code.transforms as T
import torchvision
import numpy as np
import torchvision.models.detection
from code.coco_utils import get_coco
from code.engine import train_one_epoch
from PIL import Image
from code.coco_utils import ConvertCocoPolysToMask
print("torch.__version__:{}".format(torch.__version__))
print("torchvision.__version__:{}".format(torchvision.__version__))
class InferenceDataset(object):
    def __init__(self, img_list,transforms):
        t = [ConvertCocoPolysToMask()]

        if transforms is not None:
            t.append(transforms)
        transforms = T.Compose(t)
        # as you would do normally
        self.img_list=img_list
        self.transforms=transforms
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # load the image as a PIL Image
        img_path= self.img_list[idx]
        image = Image.open(img_path).convert('RGB')

        '''构建伪coco字段，方便与image一起传入transforms做变换'''
        bbox = [0, 0, 10, 10]
        labels = 10
        target={}
        target["id"]=idx
        target["image_id"]=idx
        target["bbox"]=bbox
        points=[[bbox[0],bbox[1]],[bbox[2],bbox[1]],[bbox[2],bbox[3]],[bbox[0],bbox[3]]]
        target["segmentation"]= [np.asarray(points).flatten().tolist()]
        target["category_id"]=labels
        target["iscrowd"]=0
        target["area"]=100
        target = dict(image_id=idx, annotations=[target])

        if self.transforms:
            image, target = self.transforms(image, target)

        # return the image, the boxlist and the idx in your dataset
        return image, target
    def get_img_info(self, idx):
        img_path =self.img_list[idx]
        return {"name": os.path.basename(img_path)}

'''数据扩增'''
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
'''设置数据集，图片存储路径和标注文件路径'''
def get_dataset_test(name, image_set, transform, data_path):
    num_classes=21
    img_list= glob.glob(data_path+"/*.jpg")
    print(len(img_list))
    datesets = InferenceDataset(img_list=img_list,transforms=transform)

    return datesets, num_classes
def main(args):
    '''data_loader &dataset'''
    dataset, num_classes = get_dataset_test(args.dataset, "test", get_transform(train=False), args.data_path)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    print("Creating model")
    model = torchvision.models.detection.__dict__[args.model](num_classes=num_classes,
                                                              pretrained=args.pretrained,
                                                              )

    device = torch.device(args.device)
    model.to(device)

    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    cpu_device = torch.device("cpu")
    results = []
    for data in data_loader:
        images,ids =data
        images = list(image.to(device) for image in images)
        model_time = time.time()
        with torch.no_grad():
            outputs = model(images)
        predictions = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time
        prediction = predictions[0]
        img_name = dataset.get_img_info(int(ids[0]["image_id"].numpy()))["name"]
        def select_top_predictions(predictions,confidence_threshold=0.05):
            scores = predictions["scores"]
            boxes = predictions["boxes"]
            labels = predictions["labels"]
            result=[]
            for score,box,label in zip(scores,boxes,labels):
                if score>confidence_threshold:
                    box = list(map(float, box[:4]))
                    tmp = {'name': img_name, 'category': int(label), 'bbox': box,
                           'score': float(score)}
                    result.append(tmp)
            return result
        results.extend(select_top_predictions(prediction))
        print(model_time,img_name)
    import json
    with open('result/faster_resnet50_fpn_1000.json', 'w') as fp:
        json.dump(results, fp, indent=4, separators=(',', ': '))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser.add_argument('--data-path', default='/workdir/data/coco_dataset/coco_fabric/images/testA/', help='dataset')
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=13, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[8, 11], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='outputs/fasterrcnn_fpn_50', help='path where to save')
    parser.add_argument('--resume', default='outputs/fasterrcnn_fpn_50/model_12.pth', help='resume from checkpoint')
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",

        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        # default=True,
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)