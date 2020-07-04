from pycocotools.coco import COCO
import numpy as np
from PIL import Image
from os import path
import tensorflow as tf


class COCODatasetBBoxes(tf.keras.utils.Sequence):
    def __init__(self, cocopath, partition='val2017', return_boxes=True,
                 ignore_crowded=True, **kwargs):
        super().__init__(**kwargs)
        self.cocopath = cocopath
        self.partition = partition
        self.return_boxes = return_boxes
        self.ignore_crowded = ignore_crowded

        self.coco = COCO(path.join(cocopath, 'annotations',
                                   'instances_%s.json'%partition))
        self.img_ids = sorted(self.coco.getImgIds())


    def __len__(self):
        return len(self.img_ids)


    def __getitem__(self, idx):
        img_info = self.coco.loadImgs(self.img_ids[idx])[0]
        img_path = path.join(self.cocopath, self.partition, img_info['file_name'])
        if not self.return_boxes:
            return self.img_ids[idx], img_path
        ann_ids = self.coco.getAnnIds(self.img_ids[idx])
        boxes = self.parse_annotations(ann_ids)
        return self.img_ids[idx], img_path, boxes


    def parse_annotations(self, ann_ids):
        boxes = []
        for ann in self.coco.loadAnns(ann_ids):
            if 'iscrowd' in ann and ann['iscrowd'] > 0 and self.ignore_crowded:
                continue
            box = ann['bbox'] + [ann['category_id']]
            box = np.array(box, dtype=np.float32)
            box[2:4] += box[0:2]
            boxes.append(box)
        return boxes

        

        
