
import tensorflow as tf
from tensorflow.keras import layers, models


def fcnn(input_shape=(64, 64, 3)):
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])
    return model


# U-Net para predecir la IR


def simple_unet(input_shape=(64, 64, 3)):
    inputs = layers.Input(input_shape)

    # Encoder
    conv1 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    conv1 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    conv2 = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(pool1)
    conv2 = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    # Bottleneck
    conv3 = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(pool2)
    conv3 = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(conv3)

    # Decoder
    up1 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation="relu")(conv3)
    up1 = layers.Concatenate()([up1, conv2])
    conv4 = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(up1)
    conv4 = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(conv4)

    up2 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same", activation="relu")(conv4)
    up2 = layers.Concatenate()([up2, conv1])
    conv5 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(up2)
    conv5 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(conv5)

    # Output layer (IR predicha)
    ir_output = layers.Conv2D(1, (1, 1), activation="sigmoid", name="ir_output")(conv5)

    return models.Model(inputs=inputs, outputs=ir_output)


class ExtLayer(layers.Layer):
        def __init__(self, initial_sigma=0.1):
            super(ExtLayer, self).__init__()
            # Initialize the trainable variable
            self.var = tf.Variable(initial_value=[[initial_sigma]], dtype=tf.float32, trainable=True)

        def call(self, inputs):
            batch_size = tf.shape(inputs)[0]  # Get the current batch size
            # Tile sigma to match the batch size
            tensor = tf.tile(self.var, [batch_size, 1])
            return tensor

#Multi-task model


def MT_band(reg_sh = (64, 64, 3), pri_sh = (64, 64, 4)):
    unet_model = simple_unet(input_shape = reg_sh)
    classifier_model = fcnn(input_shape = pri_sh) 

    rgb_input = layers.Input(shape=reg_sh)
    ir_predicted = unet_model(rgb_input)  

    concatenated = layers.Concatenate(axis=-1)([rgb_input, ir_predicted])

    classification_output = classifier_model(concatenated)

    classification_output_expanded = layers.Reshape((1, 1, 1))(classification_output)
    classification_output_upsampled = layers.UpSampling2D(size=(64, 64))(classification_output_expanded) 


    #External parameters
    sigma_output = ExtLayer()(rgb_input)
    temp_output = ExtLayer()(rgb_input)


    sigma_out_expand = layers.Reshape((1, 1, 1))(sigma_output)
    sigma = layers.UpSampling2D(size=(64, 64))(sigma_out_expand) 

    temperature_out_expand = layers.Reshape((1, 1, 1))(temp_output)
    temperature = layers.UpSampling2D(size=(64, 64))(temperature_out_expand)


    conc= layers.Concatenate(axis=-1)([ir_predicted, classification_output_upsampled,
                                        sigma, temperature])

    multi_task_model = models.Model(inputs=rgb_input, outputs=[conc])
    
    return multi_task_model


