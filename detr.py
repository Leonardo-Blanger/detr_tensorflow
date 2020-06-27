import pickle
import tensorflow as tf

from backbone import ResNet50Backbone
from position_embeddings import PositionEmbeddingSine


class DETR(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone = ResNet50Backbone(name='backbone/0/body')
        self.pos_encoder = PositionEmbeddingSine(num_pos_features=128,
                                                 normalize=True)

    def call(self, inp):
        x, masks = inp
        x = self.backbone(x)
        masks = self.downsample_masks(masks, x)
        return x, masks

    def downsample_masks(self, masks, x):
        masks = tf.cast(masks, tf.float32)
        masks = tf.expand_dims(masks, -1)
        # The existing tf.image.resize with method='nearest'
        # does not expose the half_pixel_centers option in TF 2.2.0
        # The original Pytorch F.interpolate uses it like this
        masks = tf.compat.v1.image.resize_nearest_neighbor(
            masks, tf.shape(x)[1:3], align_corners=False, half_pixel_centers=False)
        masks = tf.squeeze(masks, -1)
        return masks
        

    def load_from_pickle(self, pickle_file):
        with open(pickle_file, 'rb') as f:
            detr_weights = pickle.load(f)

        for var in self.variables:
            var = var.assign(detr_weights[var.name[:-2]])

