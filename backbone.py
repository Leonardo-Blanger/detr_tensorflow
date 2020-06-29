import tensorflow as tf
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, ReLU, MaxPool2D

from custom_layers import FrozenBatchNorm2D


class ResNet50Backbone(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.channel_avg = tf.constant([0.485, 0.456, 0.406])
        self.channel_std = tf.constant([0.229, 0.224, 0.225])

        self.pad1 = ZeroPadding2D(3, name='pad1')
        self.conv1 = Conv2D(64, kernel_size=7, strides=2, padding='valid',
                            use_bias=False, name='conv1')
        self.bn1 = FrozenBatchNorm2D(name='bn1')
        self.relu = ReLU(name='relu')
        self.pad2 = ZeroPadding2D(1, name='pad2')
        self.maxpool = MaxPool2D(pool_size=3, strides=2, padding='valid')

        self.layer1 = tf.keras.Sequential([
            BottleNeck(64, 256, downsample=True, name='0'),
            BottleNeck(64, 256, name='1'),
            BottleNeck(64, 256, name='2'),
        ], name='layer1')

        self.layer2 = tf.keras.Sequential([
            BottleNeck(128, 512, strides=2, downsample=True, name='0'),
            BottleNeck(128, 512, name='1'),
            BottleNeck(128, 512, name='2'),
            BottleNeck(128, 512, name='3'),
        ], name='layer2')

        self.layer3 = tf.keras.Sequential([
            BottleNeck(256, 1024, strides=2, downsample=True, name='0'),
            BottleNeck(256, 1024, name='1'),
            BottleNeck(256, 1024, name='2'),
            BottleNeck(256, 1024, name='3'),
            BottleNeck(256, 1024, name='4'),
            BottleNeck(256, 1024, name='5'),
        ], name='layer3')

        self.layer4 = tf.keras.Sequential([
            BottleNeck(512, 2048, strides=2, downsample=True, name='0'),
            BottleNeck(512, 2048, name='1'),
            BottleNeck(512, 2048, name='2'),
        ], name='layer4')


    def call(self, x):
        x = (x - self.channel_avg) / self.channel_std

        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pad2(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class BottleNeck(tf.keras.Model):
    def __init__(self, dim1, dim2, strides=1, downsample=False, **kwargs):
        super().__init__(**kwargs)

        self.downsample = downsample
        self.pad = ZeroPadding2D(1)
        self.relu = ReLU(name='relu')
        
        self.conv1 = Conv2D(dim1, kernel_size=1, use_bias=False, name='conv1')
        self.bn1 = FrozenBatchNorm2D(name='bn1')

        self.conv2 = Conv2D(dim1, kernel_size=3, strides=strides, use_bias=False, name='conv2')
        self.bn2 = FrozenBatchNorm2D(name='bn2')
        
        self.conv3 = Conv2D(dim2, kernel_size=1, use_bias=False, name='conv3')
        self.bn3 = FrozenBatchNorm2D(name='bn3')

        if self.downsample:
            self.downsample = tf.keras.Sequential([
                Conv2D(dim2, kernel_size=1, strides=strides, use_bias=False, name='0'),
                FrozenBatchNorm2D(name='1')
            ], name='downsample')
        else:
            self.downsample = None


    def call(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.pad(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
            
