"""
Created on Feb 12 2021
@author: met
"""
import sys
from loguru import logger
import tensorflow as tf
import tensorflow_addons as tfa
from typing import Optional

class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filter_num: int, dropout_rate: float, stride: int = 2, activation: str = 'relu', bn: bool = True,
                 groups: int = 8, kernel_initializer: str = 'he_normal', kernel_regularizer: float = 1.e-4,
                 n_conv_row: int = 1, mult_factor: int = 1, **kwargs):
        """
        ResNet-like encoder bottleneck block for 3D tensors.

        Structure: - Input -|> Conv > BN > Conv > BN > Conv > BN > Dropout > Add -
                            \___________________> Conv > BN >_________________|
        :param filter_num: base number of used filters.
        :param dropout_rate: used dropout rate. A 0 value means no dropout.
        :param stride: applied downsampling stride.
        :param activation: a registered tf2 activation function.
        :param bn: whether use batch normalization (BN), Group Normalization (GN), or None
        :param groups: the number of groups for Group Normalization.
        :param kernel_initializer: initializer for the kernel weights matrix (see keras.initializers).
        :param kernel_regularizer: regularizer that applies a L2 regularization penalty of the given value.
        :param mult_factor: middle filter multiplicative factor
        """
        super(BottleNeck, self).__init__(**kwargs)
        self.activation = activation
        self.filter_num = filter_num
        self.dropout_rate = dropout_rate
        self.stride = stride
        self.bn = bn
        self.mult_factor = mult_factor

        kernel_regularizer = tf.keras.regularizers.l2(kernel_regularizer)

        self.conv1 = tf.keras.layers.Conv3D(filters=filter_num,
                                            kernel_size=(1, 1, 1),
                                            strides=stride,
                                            padding='same',
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=kernel_regularizer
                                            )
        self.conv2 = tf.keras.layers.Conv3D(filters=filter_num * self.mult_factor,
                                            # TODO: check if this operation makes sense
                                            kernel_size=(3, 3, 3),
                                            strides=1,
                                            padding='same',
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=kernel_regularizer
                                            )
        self.conv3 = tf.keras.layers.Conv3D(filters=filter_num * 4,
                                            kernel_size=(1, 1, 1),
                                            strides=1,
                                            padding='same',
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=kernel_regularizer
                                            )

        if self.bn == 'BN':
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.bn2 = tf.keras.layers.BatchNormalization()
            self.bn3 = tf.keras.layers.BatchNormalization()
        elif self.bn == 'GN':
            self.bn1 = tfa.layers.GroupNormalization(groups=min(groups,filter_num))
            self.bn2 = tfa.layers.GroupNormalization(groups=min(groups,filter_num))
            self.bn3 = tfa.layers.GroupNormalization(groups=min(groups,filter_num))

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        self.downsample = tf.keras.Sequential()
        self.downsample.add(tf.keras.layers.Conv3D(filters=filter_num * 4,
                                                   kernel_size=(1, 1, 1),
                                                   strides=stride,
                                                   kernel_initializer=kernel_initializer,
                                                   kernel_regularizer=kernel_regularizer
                                                   )
                            )
        if self.bn:
            self.downsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.downsample.add(tfa.layers.GroupNormalization(groups=min(groups,filter_num)))

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        if self.bn in ['BN', 'GN']:
            x = self.bn1(x, training=training)
        x = getattr(tf.nn, self.activation)(x)
        x = self.conv2(x)
        if self.bn in ['BN', 'GN']:
            x = self.bn2(x, training=training)
        x = getattr(tf.nn, self.activation)(x)
        x = self.conv3(x)
        if self.bn in ['BN', 'GN']:
            x = self.bn3(x, training=training)
        if training:
            x = self.dropout(x)
        output = getattr(tf.nn, self.activation)(tf.keras.layers.add([residual, x]))

        return output

    def get_config(self):
        return {"filter_num": self.filter_num,
                "dropout_rate": self.dropout_rate,
                "stride": self.stride,
                "activation": self.activation,
                "bn": self.bn,
                "mult_factor": self.mult_factor
                }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)

class Plain(tf.keras.layers.Layer):
    def __init__(self, filter_num: int, dropout_rate: float, stride: int = 2, activation: str = 'relu', bn: bool = True,
                 groups: int = 8, kernel_initializer: str = 'he_normal', kernel_regularizer: float = 1.e-4,
                 n_conv_row: int = 1, mult_factor: int = 1, **kwargs):
        """
        VGG-like encoder block for 3D tensors.

        Structure: - Input -|> Conv > Conv (DS) > BN > Dropout -

        :param filter_num: base number of used filters.
        :param dropout_rate: used dropout rate. A 0 value means no dropout.
        :param stride: applied downsampling stride.
        :param activation: a registered tf2 activation function.
        :param bn: whether use batch normalization (BN), Group Normalization (GN), or None
        :param groups: the number of groups for Group Normalization.
        :param kernel_initializer: initializer for the kernel weights matrix (see keras.initializers).
        :param kernel_regularizer: regularizer that applies a L2 regularization penalty of the given value.
        :param mult_factor: middle filter multiplicative factor
        """
        super(Plain, self).__init__(**kwargs)
        self.activation = activation
        self.filter_num = filter_num
        self.dropout_rate = dropout_rate
        self.stride = stride
        self.bn = bn
        kernel_regularizer = tf.keras.regularizers.l2(kernel_regularizer)
        self.n_conv_row = n_conv_row
        self.mult_factor = mult_factor

        # Define conv layers with Convs, BN, and activation
        self.convs = tf.keras.Sequential()
        for i in range(self.n_conv_row):
            self.convs.add(tf.keras.layers.Conv3D(filters=filter_num * self.mult_factor,  # Conv
                                                  kernel_size=(3, 3, 3),
                                                  strides=1,
                                                  padding='same',
                                                  kernel_initializer=kernel_initializer,
                                                  kernel_regularizer=kernel_regularizer,
                                                  ))
            if self.bn == 'BN':  # Batch norm
                self.convs.add(tf.keras.layers.BatchNormalization())
            elif self.bn == 'GN':
                self.convs.add(tfa.layers.GroupNormalization(groups=min(groups,filter_num)))
            self.convs.add(tf.keras.layers.Activation(self.activation))  # Activation

        self.down_conv = tf.keras.layers.Conv3D(filters=4 * filter_num,
                                                kernel_size=(stride, stride, stride),
                                                strides=stride,
                                                padding='same',
                                                kernel_initializer=kernel_initializer,
                                                kernel_regularizer=kernel_regularizer
                                                )

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, inputs, training=None, **kwargs):
        x = self.convs(inputs, training=training)
        if training:
            x = self.dropout(x)  # dropout
        x = self.down_conv(x)
        x = getattr(tf.nn, self.activation)(x)
        return x

    def get_config(self):
        return {"filter_num": self.filter_num,
                "dropout_rate": self.dropout_rate,
                "stride": self.stride,
                "activation": self.activation,
                "bn": self.bn,
                "n_conv_row": self.n_conv_row,
                "mult_factor": self.mult_factor
                }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)

class UpBottleNeck(tf.keras.layers.Layer):
    def __init__(self, filter_num: int, dropout_rate: float, stride: int = 2, activation: str = 'relu', bn: bool = True,
                 groups=8, kernel_initializer='he_normal', kernel_regularizer=1.e-4, n_conv_row=1, mult_factor: int = 1,
                 **kwargs):
        """
        ResNet-like decoder bottleneck block for 3D tensors.

        Structure: - Input -|> Conv > BN > ConvT > BN > Conv > BN > Dropout > Add -
                            \___________________> ConvT > BN >_________________|
        :param filter_num: base number of used filters.
        :param dropout_rate: used dropout rate. A 0 value means no dropout.
        :param stride: applied upsampling stride.
        :param activation: a registered tf2 activation function.
        :param bn: whether use batch normalization (BN), Group Normalization (GN), or None
        :param groups: the number of groups for Group Normalization.
        :param kernel_initializer: initializer for the kernel weights matrix (see keras.initializers).
        :param kernel_regularizer: regularizer that applies a L2 regularization penalty of the given value.
        :param mult_factor: middle filter multiplicative factor
        """
        super(UpBottleNeck, self).__init__(**kwargs)
        self.activation = activation
        self.filter_num = filter_num
        self.dropout_rate = dropout_rate
        self.stride = stride
        self.bn = bn
        self.mult_factor = mult_factor

        kernel_regularizer = tf.keras.regularizers.l2(kernel_regularizer)

        self.conv1 = tf.keras.layers.Conv3D(filters=filter_num,
                                            kernel_size=(1, 1, 1),
                                            strides=1,
                                            padding='same',
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=kernel_regularizer
                                            )
        self.conv2_up = tf.keras.layers.Conv3DTranspose(filters=filter_num * self.mult_factor,
                                                        kernel_size=(3, 3, 3),
                                                        strides=stride,
                                                        padding='same',
                                                        kernel_initializer=kernel_initializer,
                                                        kernel_regularizer=kernel_regularizer
                                                        )
        self.conv3 = tf.keras.layers.Conv3D(filters=filter_num * 4,
                                            kernel_size=(1, 1, 1),
                                            strides=1,
                                            padding='same',
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=kernel_regularizer)

        if self.bn == 'BN':
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.bn2 = tf.keras.layers.BatchNormalization()
            self.bn3 = tf.keras.layers.BatchNormalization()
        elif self.bn == 'GN':
            self.bn1 = tfa.layers.GroupNormalization(groups=min(groups,filter_num))
            self.bn2 = tfa.layers.GroupNormalization(groups=min(groups,filter_num))
            self.bn3 = tfa.layers.GroupNormalization(groups=min(groups,filter_num))

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        self.upsample = tf.keras.Sequential()
        self.upsample.add(tf.keras.layers.Conv3DTranspose(filters=filter_num * 4,
                                                          kernel_size=(1, 1, 1),
                                                          strides=stride,
                                                          kernel_initializer=kernel_initializer,
                                                          kernel_regularizer=kernel_regularizer
                                                          )
                          )
        if self.bn:
            self.upsample.add(tf.keras.layers.BatchNormalization())
        else:
            self.upsample.add(tfa.layers.GroupNormalization(groups=min(groups,filter_num)))

    def call(self, inputs, training=None, **kwargs):
        residual = self.upsample(inputs)

        x = self.conv1(inputs)
        if self.bn in ['BN', 'GN']:
            x = self.bn1(x, training=training)
        x = getattr(tf.nn, self.activation)(x)
        x = self.conv2_up(x)
        if self.bn in ['BN', 'GN']:
            x = self.bn2(x, training=training)
        x = getattr(tf.nn, self.activation)(x)
        x = self.conv3(x)
        if self.bn in ['BN', 'GN']:
            x = self.bn3(x, training=training)
        if training:
            x = self.dropout(x)
        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output

    def get_config(self):
        return {"filter_num": self.filter_num,
                "dropout_rate": self.dropout_rate,
                "stride": self.stride,
                "activation": self.activation,
                "bn": self.bn,
                "mult_factor": self.mult_factor
                }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)

class UpPlain(tf.keras.layers.Layer):
    def __init__(self, filter_num: int, dropout_rate: float, stride: int = 2, activation: str = 'relu', bn: bool = True,
                 groups=8, kernel_initializer='he_normal', kernel_regularizer=1.e-4, n_conv_row=1, mult_factor: int = 1,
                 **kwargs):
        """
        VGG-like encoder block for 3D tensors.

        Structure: - Input -|> Conv > Conv Transpose > BN > Dropout -

        :param filter_num: base number of used filters.
        :param dropout_rate: used dropout rate. A 0 value means no dropout.
        :param stride: applied downsampling stride.
        :param activation: a registered tf2 activation function.
        :param bn: whether use batch normalization (BN), Group Normalization (GN), or None
        :param groups: the number of groups for Group Normalization.
        :param kernel_initializer: initializer for the kernel weights matrix (see keras.initializers).
        :param kernel_regularizer: regularizer that applies a L2 regularization penalty of the given value.
        :param mult_factor: middle filter multiplicative factor
        """
        super(UpPlain, self).__init__(**kwargs)
        self.activation = activation
        self.filter_num = filter_num
        self.dropout_rate = dropout_rate
        self.stride = stride
        self.bn = bn
        kernel_regularizer = tf.keras.regularizers.l2(kernel_regularizer)
        self.n_conv_row = n_conv_row
        self.mult_factor = mult_factor

        # Define conv layers with Convs, BN, and activation
        self.convs = tf.keras.Sequential()
        for i in range(self.n_conv_row):
            self.convs.add(tf.keras.layers.Conv3D(filters=filter_num * self.mult_factor,  # Conv
                                                  kernel_size=(3, 3, 3),
                                                  strides=1,
                                                  padding='same',
                                                  kernel_initializer=kernel_initializer,
                                                  kernel_regularizer=kernel_regularizer,
                                                  ))
            if self.bn == 'BN':  # Batch norm
                self.convs.add(tf.keras.layers.BatchNormalization())
            elif self.bn == 'GN':
                self.convs.add(tfa.layers.GroupNormalization(groups=min(groups,filter_num)))
            self.convs.add(tf.keras.layers.Activation(self.activation))  # Activation

        self.up_conv = tf.keras.layers.Conv3DTranspose(filters=filter_num * 4,
                                                       kernel_size=(stride, stride, stride),
                                                       strides=stride,
                                                       padding='same',
                                                       kernel_initializer=kernel_initializer,
                                                       kernel_regularizer=kernel_regularizer
                                                       )

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, inputs, training=None, **kwargs):
        x = self.convs(inputs, training=training)
        if training:
            x = self.dropout(x)
        x = self.up_conv(x)
        x = getattr(tf.nn, self.activation)(x)
        return x

    def get_config(self):
        return {"filter_num": self.filter_num,
                "dropout_rate": self.dropout_rate,
                "stride": self.stride,
                "activation": self.activation,
                "bn": self.bn,
                "n_conv_row": self.n_conv_row,
                "mult_factor": self.mult_factor
                }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)

    #     # Define layers
    #     self.convs = []
    #     self.bns = []
    #     for i in range(self.n_conv_row):
    #         self.convs.append(tf.keras.layers.Conv3D(filters=filter_num,
    #                                         kernel_size=(3, 3, 3),
    #                                         strides=1,
    #                                         padding='same',
    #                                         kernel_initializer=kernel_initializer,
    #                                         kernel_regularizer=kernel_regularizer,
    #                                         ))

    #         if self.bn == 'BN':
    #             self.bns.append(tf.keras.layers.BatchNormalization())
    #         elif self.bn == 'GN':
    #             self.bns.append(tfa.layers.GroupNormalization(groups=groups))

    #     self.down_conv = tf.keras.layers.Conv3D(filters=4 * filter_num,
    #                                             kernel_size=(stride, stride, stride),
    #                                             strides=stride,
    #                                             padding='same',
    #                                             kernel_initializer=kernel_initializer,
    #                                             kernel_regularizer=kernel_regularizer
    #                                             )

    #     self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    # def call(self, inputs, training=None, **kwargs):

    #     x = inputs
    #     for i_conv, i_bn in zip(self.convs, self.bns):
    #         x = i_conv(x)                               # conv
    #         if self.bn in ['BN', 'GN']:                 # bn
    #             x = i_bn(x, training=training)
    #         x = getattr(tf.nn, self.activation)(x)      # relu

    #     if training:
    #         x = self.dropout(x)                        # dropout

    #     x = self.down_conv(x)
    #     x = getattr(tf.nn, self.activation)(x)        

    #     return x

    # def get_config(self):
    #     return {"filter_num": self.filter_num,
    #             "dropout_rate": self.dropout_rate,
    #             "stride": self.stride,
    #             "activation": self.activation,
    #             "bn": self.bn,
    #             "n_conv_row": self.n_conv_row,
    #             }
    
class BottleNeck_v2(tf.keras.layers.Layer):
    def __init__(self, filter_num: int, dropout_rate: float = 0.1, stride: int = 2, activation: str = 'relu', bn: bool = True,
                 groups: int = 8, kernel_initializer: str = 'he_normal', kernel_regularizer: float = 1.e-4,
                 n_conv_row: int = 1, mult_factor: int = 1, downsampling: str = 'conv', **kwargs):
        """
        ResNet-like encoder bottleneck block for 3D tensors.
        Structure: - Input -|> Conv > BN > Act > (Conv > BN > Act) * 4 '!= kernels' > Conv > BN >  Act > Dropout > Add > Activation -
                            \___________________> Conv > BN >____________________________________/
        :param filter_num: base number of used filters.
        :param dropout_rate: used dropout rate. A 0 value means no dropout.
        :param stride: applied downsampling stride.
        :param activation: a registered tf2 activation function.
        :param bn: whether use batch normalization (BN), Group Normalization (GN), or None
        :param groups: the number of groups for Group Normalization.
        :param kernel_initializer: initializer for the kernel weights matrix (see keras.initializers).
        :param kernel_regularizer: regularizer that applies a L2 regularization penalty of the given value.
        :param mult_factor: middle filter division factor
        """
        super(BottleNeck_v2, self).__init__(**kwargs)
        self.activation = activation
        self.filter_num = filter_num
        self.dropout_rate = dropout_rate
        self.stride = stride
        self.bn = bn
        self.mult_factor = mult_factor
        self.downsampling = downsampling

        kernel_regularizer = tf.keras.regularizers.l2(kernel_regularizer)

        self.conv1 = tf.keras.layers.Conv3D(filters=filter_num,
                                            kernel_size=(1, 1, 1),
                                            strides=1,
                                            padding='same',
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=kernel_regularizer
                                            )
        self.conv2 = tf.keras.layers.Conv3D(filters=int(filter_num * self.mult_factor),
                                            kernel_size=(1, 1, 1),
                                            strides=stride,
                                            padding='same',
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=kernel_regularizer
                                            )
        self.conv3 = tf.keras.layers.Conv3D(filters=int(filter_num * self.mult_factor),
                                            kernel_size=(3, 3, 3),
                                            strides=stride,
                                            padding='same',
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=kernel_regularizer
                                            )
        self.conv4 = tf.keras.layers.Conv3D(filters=int(filter_num * self.mult_factor),
                                            kernel_size=(5, 5, 5),
                                            strides=stride,
                                            padding='same',
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=kernel_regularizer
                                            )
        self.conv5 = tf.keras.layers.Conv3D(filters=int(filter_num * self.mult_factor),
                                            kernel_size=(7, 7, 7),
                                            strides=stride,
                                            padding='same',
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=kernel_regularizer
                                            )
        self.conv6 = tf.keras.layers.Conv3D(filters=filter_num,
                                            kernel_size=(1, 1, 1),
                                            strides=1,
                                            padding='same',
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=kernel_regularizer
                                            )

        if self.bn == 'BN':
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.bn2 = tf.keras.layers.BatchNormalization()
            self.bn3 = tf.keras.layers.BatchNormalization()
            self.bn4 = tf.keras.layers.BatchNormalization()
            self.bn5 = tf.keras.layers.BatchNormalization()
            self.bn6 = tf.keras.layers.BatchNormalization()
        elif self.bn == 'GN':
            self.bn1 = tfa.layers.GroupNormalization(groups=min(groups,filter_num))
            self.bn2 = tfa.layers.GroupNormalization(groups=min(groups,filter_num))
            self.bn3 = tfa.layers.GroupNormalization(groups=min(groups,filter_num))
            self.bn4 = tfa.layers.GroupNormalization(groups=min(groups,filter_num))
            self.bn5 = tfa.layers.GroupNormalization(groups=min(groups,filter_num))
            self.bn6 = tfa.layers.GroupNormalization(groups=min(groups,filter_num))

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        # Downsampling
        self.downsample = tf.keras.Sequential()
        if downsampling == 'conv':
            self.downsample.add(tf.keras.layers.Conv3D(filters = filter_num,
                                                    kernel_size = (1, 1, 1),
                                                    strides = self.stride,
                                                    padding = 'same',
                                                    kernel_initializer = kernel_initializer,
                                                    kernel_regularizer = kernel_regularizer,
                                                    ))
        elif downsampling == 'pooling':
            self.downsample.add(tf.keras.layers.MaxPool3D(pool_size = self.stride))
        else:
            logger.error(f'ERROR 19.0: config.network.downsampling {downsampling} not recognised!')
            sys.exit(0)

        if self.bn == 'BN':
            self.downsample.add(tf.keras.layers.BatchNormalization())
        elif self.bn == 'GN':
            self.downsample.add(tfa.layers.GroupNormalization(groups=min(groups,filter_num)))


    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        if self.bn in ['BN', 'GN']:
            x = self.bn1(x, training=training)
        x = getattr(tf.nn, self.activation)(x)

        x_2 = self.conv2(x)
        if self.bn in ['BN', 'GN']:
            x_2 = self.bn2(x_2, training=training)
        x_2 = getattr(tf.nn, self.activation)(x_2)
        x_3 = self.conv3(x)
        if self.bn in ['BN', 'GN']:
            x_3 = self.bn3(x_3, training=training)
        x_3 = getattr(tf.nn, self.activation)(x_3)
        x_4 = self.conv4(x)
        if self.bn in ['BN', 'GN']:
            x_4 = self.bn4(x_4, training=training)
        x_4 = getattr(tf.nn, self.activation)(x_4)
        x_5 = self.conv5(x)
        if self.bn in ['BN', 'GN']:
            x_5 = self.bn5(x_5, training=training)
        x_5 = getattr(tf.nn, self.activation)(x_5)
        x = tf.keras.layers.concatenate([x_2, x_3, x_4, x_5])

        x = self.conv6(x)
        if self.bn in ['BN', 'GN']:
            x = self.bn6(x, training=training)
        if training:
            x = self.dropout(x)
        output = getattr(tf.nn, self.activation)(tf.keras.layers.add([residual, x]))

        return output

    def get_config(self):
        return {"filter_num": self.filter_num,
                "dropout_rate": self.dropout_rate,
                "stride": self.stride,
                "activation": self.activation,
                "bn": self.bn,
                "mult_factor": self.mult_factor
                }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)

class BottleNeck_v3(tf.keras.layers.Layer):
    def __init__(self, filter_num: int, dropout_rate: float = 0.1, stride: int = 2, activation: str = 'relu', bn: bool = True,
                 groups: int = 8, kernel_initializer: str = 'he_normal', kernel_regularizer: float = 1.e-4,
                 n_conv_row: int = 1, mult_factor: int = 1, downsampling: str = 'conv', **kwargs):
        """
        ResNet-like encoder bottleneck block for 3D tensors.
        Structure: - Input -|> Conv > BN > Act > (Conv > BN > Act) * 4 '!= kernels' > Conv > BN >  Act > Dropout > Add > Activation -
                            \___________________> Conv > BN >____________________________________/
        :param filter_num: base number of used filters.
        :param dropout_rate: used dropout rate. A 0 value means no dropout.
        :param stride: applied downsampling stride.
        :param activation: a registered tf2 activation function.
        :param bn: whether use batch normalization (BN), Group Normalization (GN), or None
        :param groups: the number of groups for Group Normalization.
        :param kernel_initializer: initializer for the kernel weights matrix (see keras.initializers).
        :param kernel_regularizer: regularizer that applies a L2 regularization penalty of the given value.
        :param mult_factor: middle filter division factor
        """
        super(BottleNeck_v3, self).__init__(**kwargs)
        self.activation = activation
        self.filter_num = filter_num
        self.dropout_rate = dropout_rate
        self.stride = stride
        self.bn = bn
        self.mult_factor = mult_factor
        self.downsampling = downsampling

        kernel_regularizer = tf.keras.regularizers.l2(kernel_regularizer)

        self.conv1 = tf.keras.layers.Conv3D(filters=filter_num,
                                            kernel_size=(1, 1, 1),
                                            strides=1,
                                            padding='same',
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=kernel_regularizer
                                            )
        self.conv3 = tf.keras.layers.Conv3D(filters=int(filter_num * self.mult_factor),
                                            kernel_size=(3, 3, 3),
                                            strides=stride,
                                            padding='same',
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=kernel_regularizer
                                            )
        self.conv5 = tf.keras.layers.Conv3D(filters=int(filter_num * self.mult_factor),
                                            kernel_size=(7, 7, 7),
                                            strides=stride,
                                            padding='same',
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=kernel_regularizer
                                            )
        self.conv6 = tf.keras.layers.Conv3D(filters=filter_num,
                                            kernel_size=(1, 1, 1),
                                            strides=1,
                                            padding='same',
                                            kernel_initializer=kernel_initializer,
                                            kernel_regularizer=kernel_regularizer
                                            )

        if self.bn == 'BN':
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.bn3 = tf.keras.layers.BatchNormalization()
            self.bn5 = tf.keras.layers.BatchNormalization()
            self.bn6 = tf.keras.layers.BatchNormalization()
        elif self.bn == 'GN':
            self.bn1 = tfa.layers.GroupNormalization(groups=min(groups,filter_num))
            self.bn3 = tfa.layers.GroupNormalization(groups=min(groups,filter_num))
            self.bn5 = tfa.layers.GroupNormalization(groups=min(groups,filter_num))
            self.bn6 = tfa.layers.GroupNormalization(groups=min(groups,filter_num))

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        # Downsampling
        self.downsample = tf.keras.Sequential()
        if downsampling == 'conv':
            self.downsample.add(tf.keras.layers.Conv3D(filters = filter_num,
                                                    kernel_size = (1, 1, 1),
                                                    strides = self.stride,
                                                    padding = 'same',
                                                    kernel_initializer = kernel_initializer,
                                                    kernel_regularizer = kernel_regularizer,
                                                    ))
        elif downsampling == 'pooling':
            self.downsample.add(tf.keras.layers.MaxPool3D(pool_size = self.stride))
        else:
            logger.error(f'ERROR 19.0: config.network.downsampling {downsampling} not recognised!')
            sys.exit(0)

        if self.bn == 'BN':
            self.downsample.add(tf.keras.layers.BatchNormalization())
        elif self.bn == 'GN':
            self.downsample.add(tfa.layers.GroupNormalization(groups=min(groups,filter_num)))


    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        if self.bn in ['BN', 'GN']:
            x = self.bn1(x, training=training)
        x = getattr(tf.nn, self.activation)(x)

        x_3 = self.conv3(x)
        if self.bn in ['BN', 'GN']:
            x_3 = self.bn3(x_3, training=training)
        x_3 = getattr(tf.nn, self.activation)(x_3)
        x_5 = self.conv5(x)
        if self.bn in ['BN', 'GN']:
            x_5 = self.bn5(x_5, training=training)
        x_5 = getattr(tf.nn, self.activation)(x_5)
        x = tf.keras.layers.concatenate([x_3, x_5])

        x = self.conv6(x)
        if self.bn in ['BN', 'GN']:
            x = self.bn6(x, training=training)
        if training:
            x = self.dropout(x)
        output = getattr(tf.nn, self.activation)(tf.keras.layers.add([residual, x]))

        return output

    def get_config(self):
        return {"filter_num": self.filter_num,
                "dropout_rate": self.dropout_rate,
                "stride": self.stride,
                "activation": self.activation,
                "bn": self.bn,
                "mult_factor": self.mult_factor
                }

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
    
#Copied from https://github.com/tensorflow/models/blob/v2.10.0/official/vision/modeling/layers/nn_layers.py#L88-L203

def make_divisible(value: float,
                   divisor: int,
                   min_value: Optional[float] = None,
                   round_down_protect: bool = True,
                   ) -> int:
  """This is to ensure that all layers have channels that are divisible by 8.

  Args:
    value: A `float` of original value.
    divisor: An `int` of the divisor that need to be checked upon.
    min_value: A `float` of  minimum value threshold.
    round_down_protect: A `bool` indicating whether round down more than 10%
      will be allowed.

  Returns:
    The adjusted value in `int` that is divisible against divisor.
  """
  if min_value is None:
    min_value = divisor
  new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if round_down_protect and new_value < 0.9 * value:
    new_value += divisor
  return int(new_value)

class SqueezeExcitation(tf.keras.layers.Layer):
  """Creates a squeeze and excitation layer."""

  def __init__(self,
               in_filters,
               out_filters,
               se_ratio,
               divisible_by=1,
               use_3d_input=True,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               activation='relu',
               gating_activation='sigmoid',
               round_down_protect=True,
               **kwargs):
    """Initializes a squeeze and excitation layer.

    Args:
      in_filters: An `int` number of filters of the input tensor.
      out_filters: An `int` number of filters of the output tensor.
      se_ratio: A `float` or None. If not None, se ratio for the squeeze and
        excitation layer.
      divisible_by: An `int` that ensures all inner dimensions are divisible by
        this number.
      use_3d_input: A `bool` of whether input is 2D or 3D image.
      kernel_initializer: A `str` of kernel_initializer for convolutional
        layers.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2d.
        Default to None.
      activation: A `str` name of the activation function.
      gating_activation: A `str` name of the activation function for final
        gating function.
      round_down_protect: A `bool` of whether round down more than 10% will be
        allowed.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(SqueezeExcitation, self).__init__(**kwargs)

    self._in_filters = in_filters
    self._out_filters = out_filters
    self._se_ratio = se_ratio
    self._divisible_by = divisible_by
    self._round_down_protect = round_down_protect
    self._use_3d_input = use_3d_input
    self._activation = activation
    self._gating_activation = gating_activation
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    if tf.keras.backend.image_data_format() == 'channels_last':
      if not use_3d_input:
        self._spatial_axis = [1, 2]
      else:
        self._spatial_axis = [1, 2, 3]
    else:
      if not use_3d_input:
        self._spatial_axis = [2, 3]
      else:
        self._spatial_axis = [2, 3, 4]
    self._activation_fn = getattr(tf.nn, activation)
    self._gating_activation_fn = getattr(tf.nn, gating_activation)

  def build(self, input_shape):
    num_reduced_filters = make_divisible(
        max(1, int(self._in_filters * self._se_ratio)),
        divisor=self._divisible_by,
        round_down_protect=self._round_down_protect)

    self._se_reduce = tf.keras.layers.Conv2D(
        filters=num_reduced_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=True,
        kernel_initializer= self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)

    self._se_expand = tf.keras.layers.Conv2D(
        filters=self._out_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=True,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)

    super(SqueezeExcitation, self).build(input_shape)

  def get_config(self):
    config = {
        'in_filters': self._in_filters,
        'out_filters': self._out_filters,
        'se_ratio': self._se_ratio,
        'divisible_by': self._divisible_by,
        'use_3d_input': self._use_3d_input,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activation': self._activation,
        'gating_activation': self._gating_activation,
        'round_down_protect': self._round_down_protect,
    }
    base_config = super(SqueezeExcitation, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):
    x = tf.reduce_mean(inputs, self._spatial_axis, keepdims=True)
    x = self._activation_fn(self._se_reduce(x))
    x = self._gating_activation_fn(self._se_expand(x))
    return x * inputs
