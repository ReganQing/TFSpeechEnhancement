import tensorflow as tf
from TimeFrequencyLosses import loss_function_4, loss_function_5, loss_function_6, loss_function_7, loss_function_8

print(tf.__version__)


# Unet network
def unet(pretrained_weights=None, input_size=(128, 128, 1)):
    # size filter input
    size_filter_in = 16
    # normal initialization of weights
    kernel_init = 'he_normal'
    # To apply leaky relu after the conv layer
    activation_layer = None
    inputs = tf.keras.layers.Input(input_size)
    conv1 = tf.keras.layers.Conv2D(size_filter_in, 3, activation=activation_layer, padding='same',
                                   kernel_initializer=kernel_init)(inputs)

    conv1 = tf.keras.layers.ELU()(conv1)
    conv1 = tf.keras.layers.Conv2D(size_filter_in, 3, activation=activation_layer, padding='same',
                                   kernel_initializer=kernel_init)(conv1)
    conv1 = tf.keras.layers.ELU()(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(size_filter_in * 2, 3, activation=activation_layer, padding='same',
                                   kernel_initializer=kernel_init)(pool1)
    conv2 = tf.keras.layers.ELU()(conv2)
    conv2 = tf.keras.layers.Conv2D(size_filter_in * 2, 3, activation=activation_layer, padding='same',
                                   kernel_initializer=kernel_init)(conv2)
    conv2 = tf.keras.layers.ELU()(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(size_filter_in * 4, 3, activation=activation_layer, padding='same',
                                   kernel_initializer=kernel_init)(pool2)
    conv3 = tf.keras.layers.ELU()(conv3)
    conv3 = tf.keras.layers.Conv2D(size_filter_in * 4, 3, activation=activation_layer, padding='same',
                                   kernel_initializer=kernel_init)(conv3)
    conv3 = tf.keras.layers.ELU()(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = tf.keras.layers.Conv2D(size_filter_in * 8, 3, activation=activation_layer, padding='same',
                                   kernel_initializer=kernel_init)(pool3)
    conv4 = tf.keras.layers.ELU()(conv4)
    conv4 = tf.keras.layers.Conv2D(size_filter_in * 8, 3, activation=activation_layer, padding='same',
                                   kernel_initializer=kernel_init)(conv4)
    conv4 = tf.keras.layers.ELU()(conv4)
    drop4 = tf.keras.layers.Dropout(0.5)(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = tf.keras.layers.Conv2D(size_filter_in * 16, 3, activation=activation_layer, padding='same',
                                   kernel_initializer=kernel_init)(pool4)
    conv5 = tf.keras.layers.ELU()(conv5)
    conv5 = tf.keras.layers.Conv2D(size_filter_in * 16, 3, activation=activation_layer, padding='same',
                                   kernel_initializer=kernel_init)(conv5)
    conv5 = tf.keras.layers.ELU()(conv5)
    drop5 = tf.keras.layers.Dropout(0.5)(conv5)

    up6 = tf.keras.layers.Conv2D(size_filter_in * 8, 2, activation=activation_layer, padding='same',
                                 kernel_initializer=kernel_init)(tf.keras.layers.UpSampling2D(size=(2, 2))(drop5))
    up6 = tf.keras.layers.ELU()(up6)
    merge6 = tf.keras.layers.concatenate([drop4, up6], axis=3)
    conv6 = tf.keras.layers.Conv2D(size_filter_in * 8, 3, activation=activation_layer, padding='same',
                                   kernel_initializer=kernel_init)(merge6)
    conv6 = tf.keras.layers.ELU()(conv6)
    conv6 = tf.keras.layers.Conv2D(size_filter_in * 8, 3, activation=activation_layer, padding='same',
                                   kernel_initializer=kernel_init)(conv6)
    conv6 = tf.keras.layers.ELU()(conv6)
    up7 = tf.keras.layers.Conv2D(size_filter_in * 4, 2, activation=activation_layer, padding='same',
                                 kernel_initializer=kernel_init)(tf.keras.layers.UpSampling2D(size=(2, 2))(conv6))
    up7 = tf.keras.layers.ELU()(up7)
    merge7 = tf.keras.layers.concatenate([conv3, up7], axis=3)
    conv7 = tf.keras.layers.Conv2D(size_filter_in * 4, 3, activation=activation_layer, padding='same',
                                   kernel_initializer=kernel_init)(merge7)
    conv7 = tf.keras.layers.ELU()(conv7)
    conv7 = tf.keras.layers.Conv2D(size_filter_in * 4, 3, activation=activation_layer, padding='same',
                                   kernel_initializer=kernel_init)(conv7)
    conv7 = tf.keras.layers.ELU()(conv7)
    up8 = tf.keras.layers.Conv2D(size_filter_in * 2, 2, activation=activation_layer, padding='same',
                                 kernel_initializer=kernel_init)(tf.keras.layers.UpSampling2D(size=(2, 2))(conv7))
    up8 = tf.keras.layers.ELU()(up8)
    merge8 = tf.keras.layers.concatenate([conv2, up8], axis=3)
    conv8 = tf.keras.layers.Conv2D(size_filter_in * 2, 3, activation=activation_layer, padding='same',
                                   kernel_initializer=kernel_init)(merge8)
    conv8 = tf.keras.layers.ELU()(conv8)
    conv8 = tf.keras.layers.Conv2D(size_filter_in * 2, 3, activation=activation_layer, padding='same',
                                   kernel_initializer=kernel_init)(conv8)
    conv8 = tf.keras.layers.ELU()(conv8)

    up9 = tf.keras.layers.Conv2D(size_filter_in, 2, activation=activation_layer, padding='same',
                                 kernel_initializer=kernel_init)(tf.keras.layers.UpSampling2D(size=(2, 2))(conv8))
    up9 = tf.keras.layers.ELU()(up9)
    merge9 = tf.keras.layers.concatenate([conv1, up9], axis=3)
    conv9 = tf.keras.layers.Conv2D(size_filter_in, 3, activation=activation_layer, padding='same',
                                   kernel_initializer=kernel_init)(merge9)
    conv9 = tf.keras.layers.ELU()(conv9)
    conv9 = tf.keras.layers.Conv2D(size_filter_in, 3, activation=activation_layer, padding='same',
                                   kernel_initializer=kernel_init)(conv9)
    conv9 = tf.keras.layers.ELU()(conv9)
    conv9 = tf.keras.layers.Conv2D(2, 3, activation=activation_layer, padding='same', kernel_initializer=kernel_init)(
        conv9)
    conv9 = tf.keras.layers.ELU()(conv9)
    conv10 = tf.keras.layers.Conv2D(1, 1, activation='tanh')(conv9)

    model1 = tf.keras.models.Model(inputs, conv10)

    model1.compile(optimizer='adam', loss=loss_function_5)  # 此处设置的损失函数调整为自定义损失函数

    # model.summary()

    if (pretrained_weights):
        model1.load_weights(pretrained_weights)

    return model1
