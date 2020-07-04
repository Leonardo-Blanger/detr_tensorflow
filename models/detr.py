import pickle
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU

from .backbone import ResNet50Backbone
from .custom_layers import Linear
from .position_embeddings import PositionEmbeddingSine
from .transformer import Transformer
from utils import cxcywh2xyxy


class DETR(tf.keras.Model):
    def __init__(self, num_classes=91, num_queries=100,
                 backbone=None,
                 pos_encoder=None,
                 transformer=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_queries = num_queries

        self.backbone = backbone or ResNet50Backbone(name='backbone/0/body')
        self.transformer = transformer or Transformer(return_intermediate_dec=True,
                                                      name='transformer')
        self.model_dim = self.transformer.model_dim

        self.pos_encoder = pos_encoder or PositionEmbeddingSine(
            num_pos_features=self.model_dim // 2, normalize=True)

        self.input_proj = Conv2D(self.model_dim, kernel_size=1, name='input_proj')

        self.query_embed = tf.Variable(
            tf.zeros((num_queries, self.model_dim), dtype=tf.float32),
            name='query_embed/kernel')

        self.class_embed = Linear(num_classes + 1, name='class_embed')

        self.bbox_embed = tf.keras.Sequential([
            Linear(self.model_dim, name='layers/0'),
            ReLU(),
            Linear(self.model_dim, name='layers/1'),
            ReLU(),
            Linear(4, name='layers/2')
        ], name='bbox_embed')


    def call(self, inp, training=False, post_process=False):
        x, masks = inp
        x = self.backbone(x, training=training)
        masks = self.downsample_masks(masks, x)
        pos_encoding = self.pos_encoder(masks)

        hs = self.transformer(self.input_proj(x), masks, self.query_embed,
                              pos_encoding, training=training)[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = tf.sigmoid(self.bbox_embed(hs))

        output = {'pred_logits': outputs_class[-1],
                  'pred_boxes': outputs_coord[-1]}

        if post_process:
            output = self.post_process(output)
        return output


    def build(self, input_shape=None, **kwargs):
        if input_shape is None:
            input_shape = [(None, None, None, 3), (None, None, None)]
        super().build(input_shape, **kwargs)


    def downsample_masks(self, masks, x):
        masks = tf.cast(masks, tf.int32)
        masks = tf.expand_dims(masks, -1)
        # The existing tf.image.resize with method='nearest'
        # does not expose the half_pixel_centers option in TF 2.2.0
        # The original Pytorch F.interpolate uses it like this
        masks = tf.compat.v1.image.resize_nearest_neighbor(
            masks, tf.shape(x)[1:3], align_corners=False, half_pixel_centers=False)
        masks = tf.squeeze(masks, -1)
        masks = tf.cast(masks, tf.bool)
        return masks


    def post_process(self, output):
        logits, boxes = [output[k] for k in ['pred_logits', 'pred_boxes']]
        
        probs = tf.nn.softmax(logits, axis=-1)[..., :-1]
        scores = tf.reduce_max(probs, axis=-1)
        labels = tf.argmax(probs, axis=-1)
        boxes = cxcywh2xyxy(boxes)

        output = {'scores': scores,
                  'labels': labels,
                  'boxes': boxes}
        return output

    def load_from_pickle(self, pickle_file, verbose=False):
        with open(pickle_file, 'rb') as f:
            detr_weights = pickle.load(f)

        for var in self.variables:
            if verbose:
                print('Loading', var.name)
            var = var.assign(detr_weights[var.name[:-2]])

