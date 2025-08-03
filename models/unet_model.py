import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

def conv_block(x, filters, dropout_rate=0.3):
    x = layers.Conv2D(filters, 3, padding='same',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.SeparableConv2D(filters, 3, padding='same',
                               depthwise_regularizer=regularizers.l2(1e-4),
                               pointwise_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)
    return x


def encoder_block(x, filters, dropout_rate=0.3):
    f = conv_block(x, filters, dropout_rate)
    p = layers.MaxPooling2D(pool_size=(2, 2))(f)
    return f, p

def decoder_block(x, skip, filters, dropout_rate=0.3):
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters, dropout_rate)
    return x

def build_unet(input_shape=(128, 128, 3), base_filters=16):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    s1, p1 = encoder_block(inputs, base_filters)
    s2, p2 = encoder_block(p1, base_filters * 2)
    s3, p3 = encoder_block(p2, base_filters * 4)

    # Bottleneck
    b = conv_block(p3, base_filters * 8)

    # Decoder
    d1 = decoder_block(b, s3, base_filters * 4)
    d2 = decoder_block(d1, s2, base_filters * 2)
    d3 = decoder_block(d2, s1, base_filters)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(d3)

    return Model(inputs, outputs)
