# coding: utf-8

from tensorflow.keras import layers
from tensorflow.keras.layers import BatchNormalization, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras import models


class NetWork:
    """"""

    def __init__(self, image_size=224, num_classes=4, learning_rate=0.001):
        self.input_shape = image_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.channel = 3
        self.seed = 19990119

    def model(self):
        """

        :return: model
        """
        model = models.Sequential()
        model.add(self.conv_2d(64, bias_init=0, input_shape=(self.input_shape, self.input_shape, self.channel),
                               name='block1_conv1'))
        model.add(self.conv_2d(64, name='block1_conv2'))
        model.add(self.max_pooling_2d(name='block1_pool'))
        model.add(BatchNormalization(name='bn1'))
        model.add(self.conv_2d(128, name='block2_conv1'))
        model.add(self.conv_2d(128, name='block2_conv2'))
        model.add(self.max_pooling_2d(name='block2_pool'))
        model.add(BatchNormalization(name='bn2'))
        model.add(self.conv_2d(256, name='block3_conv1'))
        model.add(self.conv_2d(256, name='block3_conv2'))
        model.add(self.conv_2d(256, name='block3_conv3'))
        model.add(self.max_pooling_2d(name='block3_pool'))
        model.add(BatchNormalization(name='bn3'))
        model.add(self.conv_2d(512, name='block4_conv1'))
        model.add(self.conv_2d(512, name='block4_conv2'))
        model.add(self.conv_2d(512, name='block4_conv3'))
        model.add(self.max_pooling_2d(name='block4_pool'))
        model.add(BatchNormalization(name='bn4'))
        model.add(self.conv_2d(512, name='block5_conv1'))
        model.add(self.conv_2d(512, name='block5_conv2'))
        model.add(self.conv_2d(512, name='block5_conv3'))
        model.add(self.max_pooling_2d(name='block5_pool'))
        model.add(BatchNormalization(name='bn5'))
        model.add(Flatten())
        model.add(self.dense(4096))
        model.add(self.dense(4096))
        model.add(self.dense(self.num_classes, activation='softmax'))

        model.compile(optimizer=optimizers.SGD(lr=self.learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    @staticmethod
    def conv_2d(filters, kernel_size=3, strides=(1, 1), padding='same', bias_init=1, **kwargs):
        """
        畳み込み層のモデルをlayerを定義する

        Args:
            filters(integer): 出力フィルタの数
            kernel_size(tuple or list, integer): 2次元の畳込みウィンドウの幅と高さを指定
            strides(tuple or list ,default (1, 1)): 畳込みの縦横のストライドを設定
            padding(same or valid, default same): same -> ゼロパディング
                                                valid -> 入力画像より小さくなる
            bias_init(float or tensor, default 1): 重みの初期値

        Returns:
            layers.Conv2D(object):
        Note:
            see: https://keras.io/ja/layers/convolutional/
        """
        # initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.01)
        # initializer = initializers.he_normal()
        # regularizer = regularizers.l2(0.01)
        # const = initializers.Constant(value=bias_init)
        # return layers.Conv2D(filters, kernel_size, strides=strides, padding=padding,
        #                      activation='relu', kernel_initializer=initializer,
        #                      bias_initializer=const, kernel_regularizer=regularizer,
        #                      bias_regularizer=regularizer, activity_regularizer=regularizer,
        #                      **kwargs)
        initializer = initializers.he_normal()
        const = initializers.Constant(value=bias_init)
        return layers.Conv2D(filters, kernel_size, strides=strides, padding=padding,
                             activation='relu', kernel_initializer=initializer,
                             bias_initializer=const, **kwargs)

    @staticmethod
    def max_pooling_2d(pool_size=(2, 2), strides=(2, 2), padding='same', name=None):
        return layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding, name=name)

    @staticmethod
    def dense(units, activation='relu'):
        """全結合層を作成し、返す

        Args:
            units(int): 出力空間の次元数
            activation(str): 活性化関数
        Returns:
            layers.Dense(layer): 全結合層
        """
        # truncate = initializers.TruncatedNormal(mean=0.0, stddev=0.01)
        # regularizer = regularizers.l2(0.01)
        # const = initializers.Constant(value=1)
        # return layers.Dense(units, activation=activation, kernel_initializer=truncate, bias_initializer=const,
        #                     kernel_regularizer=regularizer, bias_regularizer=regularizer,
        #                     activity_regularizer=regularizer)
        truncate = initializers.TruncatedNormal(mean=0.0, stddev=0.01)
        const = initializers.Constant(value=1)
        return layers.Dense(units, activation=activation,
                            kernel_initializer=truncate,
                            bias_initializer=const)