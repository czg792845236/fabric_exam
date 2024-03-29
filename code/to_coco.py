import numpy as np
import json

import os
import pandas as pd
import shutil

defect_name2label = {
    '破洞': 1, '水渍': 2, '油渍': 2, '污渍': 2, '三丝': 3, '结头': 4, '花板跳': 5, '百脚': 6, '毛粒': 7,
    '粗经': 8, '松经': 9, '断经': 10, '吊经': 11, '粗维': 12, '纬缩': 13, '浆斑': 14, '整经结': 15, '星跳': 16, '跳花': 16,
    '断氨纶': 17, '稀密档': 18, '浪纹档': 18, '色差档': 18, '磨痕': 19, '轧痕': 19, '修痕': 19, '烧毛痕': 19, '死皱': 20, '云织': 20,
    '双纬': 20, '双经': 20, '跳纱': 20, '筘路': 20, '纬纱不良': 20,
}


class Fabric2COCO():
    def __init__(self, mode='train'):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.mode = mode
        if not os.path.exists("coco/images/{}".format(self.mode)):
            os.makedirs("coco/images/{}".format((self.mode)))

    def _init_categories(self):
        for v in range(1, 21):
            print(v)
            category = {}
            category['id'] = v
            category['name'] = str(v)
            category['supercategory'] = 'defect_name'
            self.categories.append(category)

    def _image(self, path, h, w):
        image = {}
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path)
        return image

    def _cp_img(self, img_path):
        shutil.copy(img_path, os.path.join("coco/images/{}".format(self.mode), os.path.basename(img_path)))

    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]

    def _annotation(self, label, bbox):
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        points = [[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = label
        annotation['segmentation'] = [list(np.asarray(points).flatten())]
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = area
        return annotation

    def to_coco(self, anno_file, img_dir):
        self._init_categories()
        anno_result = pd.read_json(open(anno_file, 'r'))
        name_list = anno_result['name'].unique()
        for img_name in name_list:
            img_anno = anno_result[anno_result['name'] == img_name]
            bboxs = list(img_anno['bbox'])
            defect_names = list(img_anno['defect_name'])

            img_path = os.path.join(img_dir, img_name)
            h, w = 1000, 2446
            self.images.append(self._image(img_path, h, w))
            self._cp_img(img_path)

            for bbox, defect_names in zip(bboxs, defect_names):
                label = defect_name2label[defect_names]
                annotation = self._annotation(label, bbox)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'fabric defect'
        instance['license'] = ['none']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    def save_coco_json(self, instance, save_path):
        with open(save_path, 'w') as fp:
            json.dump(instance, fp, indent=1, separators=(',', ': '))


img_dir = "../data/guangdong1_round1_train1_20190818/defect_Images"
anno_dir = "../data/guangdong1_round1_train1_20190818/Annotations/anno_train.json"
fabric2coco = Fabric2COCO()
train_instance = fabric2coco.to_coco(anno_dir, img_dir)
if not os.path.exists("coco/annotations"):
    os.makedirs("coco/annotations/")
fabric2coco.save_coco_json(train_instance, "coco/annotations/" + 'instances_{}.json'.format("train"))
