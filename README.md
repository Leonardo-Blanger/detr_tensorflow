# DETR Tensorflow

This project is my attempt at a Tensorflow implementation of the DETR architecture for Object Detection, from the paper *End-to-end Object Detection with Transformers* [(Carion *et al.*)](https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers).

**Attention:** This is a work in progress. It still does not offer all the functionality from the original implementation. If you only want to perform detection using DETR in Tensorflow, this is already possible. If you want to perform Panoptic Segmentation, fully replicate the paper's experiments, or train on your own dataset, this is still not possible.

## Overview

DETR, which stands for **De**tection **Tr**ansformers, was proposed by a team from the Facebook AI group, and it is, as of today, a radical shift from the current approaches to perform Deep Learning based Object Detection.

Instead of filtering and refining a set of object proposals, as done by two-stage techniques like Faster-RCNN and its adaptations, or generating dense detection grids, as done by single-stage techniques like SSD and YOLO, DETR frames the detection problem as an image to set mapping. With this formulation, both the architecture and the training process become significantly easier. There is no need for hand-designed anchor matching schemes or post-processing steps like Non Max Suppression to discard redundant detections.

DETR uses a CNN backbone to extract a higher level feature representation of the image, which is then fed into a Transformer model. The Transformer Encoder is responsible for processing this image representation, while the Decoder maps a fixed set of learned object queries to detections, performing attention over the Encoder's output.

DETR is trained with a set-based global loss that finds a bipartite matching between the set of detections and ground-truth objects (non matched detections are assigned to a special _no object_ class), which in turn forces unique detections.

For more details into the technique, please refer to their [paper](https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers) (Carion *et al.*) and [blog post](https://ai.facebook.com/blog/end-to-end-object-detection-with-transformers). Both are very well written.

## About this Implementation

After spending some time working with Object Detection for my Master's degree, and wanting to learn more about this apparently useful thing called Transformers that everybody keeps talking about, I came across this very cool idea that proposes a completely different way of doing Object Detection. So I decided to make it accessible to the Tensorflow community as well. This implementation had the main purpose of allowing myself to understand the technique more in depth, while also being an exercise on the Tensorflow framework. 

I tried my best to replicate the precise behavior of the original Pytorch implementation, trying to account for small details like the difference between how convolutions use padding in the two frameworks. This way, we can convert the existing Pytorch weights to an intermediate format and load them in this implementation. This turned out to be a great exercise to better understand not only the DETR architecture, but also how both frameworks work at a greater level of detail.

Currently, I still have not implemented any training related code, so the only way to use this implementation is by loading the converted Pytorch weights. I also did not implement the Panoptic Segmentation part yet. Regarding the Object Detection part, that is already working.

## Evaluation Results

Bellow are the results for the COCO val2017 dataset, as reported by the official Pytorch version, and achieved by this implementation using the converted weights. The small deviations are probably mostly due to the differences between how the two frameworks and implementations perform image loading and resizing, as well as floating point errors from differences in how they perform certain low level operations.

**name** | **backbone** | **box AP (official)** | **box AP (ours)**
-------- | ------------ | --------------------- | -----------------
DETR | R50 | 42.0 | 41.9
DETR-DC5 | R50 | 43.3 | 43.2
DETR | R101 | 43.5 | 43.4
DETR-DC5 | R101 | 44.9 | 44.8

## Requirements

The code was tested with `python 3.7.5` and `tensorflow 2.2.0`. In order to run the evaluation, you'll also need the `pycocotools` library.

```bash
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

## How to Use

First, download the frozen Pytorch weights `.pth` file for the model version you want to use from the official implementation's Github repo [here](https://github.com/facebookresearch/detr). Then use the provided `pth2pickle.py` script to convert the weights into a pickled python dictionary. For instance:

```bash
wget https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth
python pth2pickle.py --checkpoint=detr-r50-e632da11.pth
```

You can use one of the pre-built loading methods from the `model` package to instantiate one of the four versions provided by the original implementation.

```python
from models import build_detr_resnet50
detr = build_detr_resnet50(num_classes=91) # 91 classes for the COCO dataset
detr.build()
detr.load_from_pickle("detr-r50-e632da11.pickle")
```

Or directly instantiate the `models.DETR` class to create your own custom combination of backbone CNN, transformer architecture, and positional encoding scheme. Please, check the files `models/__init__.py` and `models/detr.py` for more details.

The `utils.preprocess_image` function is designed to perform all the preprocessing required before running the model, including data normalization, resizing following the scheme used for training, and generating the image masks. It is completely implemented using only Tensorflow operations, so you can use it in combination with the `map` functionality from `tf.data.Dataset`.

Finally, to get the final detections, call the model on your data with the `post_processing` flag. This way, it returns softmax scores instead of the pre-activation logits, and also discards the `no-object` dimension from the output. It doesn't discard low scored detections tough, but the output from DETR is simple enough that this isn't hard to do.

```python
from utils import preprocess_image

inp_image, mask = preprocess_image(image)
inp_image = tf.expand_dims(inp_image, axis=0)
mask = tf.expand_dims(mask, axis=0)

outputs = detr((inp_image, mask), post_process=True)
labels, scores, boxes = [outputs[k][0].numpy() for k in ['labels', 'scores', 'boxes']]

keep = scores > 0.7
labels = labels[keep]
scores = scores[keep]
boxes = boxes[keep]
boxes = absolute2relative(boxes, (image.shape[1], image.shape[0])).numpy()
```

(so much easier than anchor decoding + Non Max Suppression)


### Demo

Short demo script that summarizes the above instructions.

```bash
python demo.py
```

### Running Evaluation

I provided an `eval.py` script that evaluates the model on the COCO val2017 dataset, same as reported in the paper. Note that you don't need to download the whole COCO dataset for this, only the val2017 partition (~1GB) and annotations (~241MB), from [here](https://cocodataset.org/#download).

```bash
python eval.py --coco_path=/path/to/coco \
               --backbone=resnet50-dc5 \
			   --frozen_weights=detr-r50-dc5-f0fb7ef5.pickle \
			   --results_file=resnet50_dc5_results.json --batch_size=1
```

It will save the detections into the `resnet50_dc5_results.json` file, in the COCO dictionary format, so you can run evaluation again with the `--from_file` flag, and it won't need to perform image inference this time.


## Detection Samples

![sample](/samples/sample_1_boxes.png)


## TODOs

- [ ] Provide pretrained weights already in `hdf5` format.
- [ ] Design a less hacky way of converting the weights. Convert directly into `hdf5` instead of `pickle`.
- [ ] Implement the training related code.
- [ ] Repeat the paper's experiments.


## References

* **The REDR paper:** Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko, *End-to-end Object Detection with Transformers*, 2020, from the Facebook AI group. [link to paper](https://arxiv.org/abs/2005.12872)

* **The official Pytorch implementation:** https://github.com/facebookresearch/detr
