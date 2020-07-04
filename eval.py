import argparse
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tensorflow as tf
from tqdm import tqdm

from datasets import COCODatasetBBoxes
import models
from utils import preprocess_image, read_jpeg_image, absolute2relative, xyxy2xywh


parser = argparse.ArgumentParser('DETR evalutaion script for the COCO dataset.')

parser.add_argument('--coco_path', type=str,
                    help='Path to the COCO dataset root directory. For evaluation, only the' +
                    ' validation data needs to be downloaded.')
parser.add_argument('--backbone', type=str, default=None,
                    choices=('resnet50', 'resnet50-dc5', 'resnet101', 'resnet101-dc5'),
                    help='Choice of backbone CNN for the model.')
parser.add_argument('--frozen_weights', type=str, default=None,
                    help='Path to the pretrained weights file. Please use the pth2pickle.py' +
                    ' script to convert the oficial PyTorch versions into .pickle.')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--results_file', type=str, default='results.json',
                    help='.json file to save the results in the COCO format.')
parser.add_argument('--from_file', action='store_true',
                    help='If specified, will compute the results using the predictions in' +
                    ' the --results_file, instead of performing inference on the whole' +
                    ' validation set again.')

args = parser.parse_args()


coco_data = COCODatasetBBoxes(args.coco_path, partition='val2017', return_boxes=False)


def evaluate(results):
    coco_dt = COCO.loadRes(coco_data.coco, args.results_file)
    cocoEval = COCOeval(coco_data.coco, coco_dt, iouType='bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if args.from_file:
    evaluate(args.results_file)
    exit()

if args.backbone is None or args.frozen_weights is None:
    raise Exception('If --from_file is not provided, both --backbone and --frozen_weights' +
                    'must be provided.')

model_fns = {
    'resnet50': models.build_detr_resnet50,
    'resnet50-dc5': models.build_detr_resnet50_dc5,
    'resnet101': models.build_detr_resnet101,
    'resnet101-dc5': models.build_detr_resnet101_dc5
}

detr = model_fns[args.backbone](num_classes=91)
detr.build()
detr.load_from_pickle(args.frozen_weights)


dataset = tf.data.Dataset.from_generator(lambda: coco_data, (tf.int32, tf.string))
dataset = dataset.map(lambda img_id, img_path: (img_id, read_jpeg_image(img_path)))
dataset = dataset.map(lambda img_id, image: (img_id, *preprocess_image(image)))

dataset = dataset.padded_batch(batch_size=args.batch_size,
                               padded_shapes=((), (None,None,3), (None,None)),
                               padding_values=(None, tf.constant(0.0), tf.constant(True)))

results = []

with tqdm(total=len(coco_data)) as pbar:
    for img_ids, images, masks in dataset:
        outputs = detr((images, masks), post_process=True)

        for img_id, scores, labels, boxes in zip(
                img_ids, outputs['scores'], outputs['labels'], outputs['boxes']):
            img_id = img_id.numpy()

            img_info = coco_data.coco.loadImgs([img_id])[0]
            img_height = img_info['height']
            img_width = img_info['width']

            for score, label, box in zip(scores, labels, boxes):
                score = score.numpy()
                label = label.numpy()
                box = absolute2relative(box, (img_width, img_height))
                box = xyxy2xywh(box).numpy()

                results.append({
                    "image_id": int(img_id),
                    "category_id": int(label),
                    "bbox": box.tolist(),
                    "score": float(score)
                })

        pbar.update(int(len(images)))

json_object = json.dumps(results, indent=2)
with open(args.results_file, 'w') as f:
    f.write(json_object)

evaluate(args.results_file)
