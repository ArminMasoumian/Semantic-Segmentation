# Author: Armin Masoumian (masoumian.armin@gmail.com)

import tensorflow as tf

class SemanticSegmentationModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
        
    def build_model(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        
        # Encoder
        conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
        
        conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
        
        conv3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = tf.keras.layers.MaxPooling2D((2, 2))(conv3)
        
        # Decoder
        up4 = tf.keras.layers.UpSampling2D((2, 2))(pool3)
        concat4 = tf.keras.layers.Concatenate()([conv3, up4])
        conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(concat4)
        conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        
        up5 = tf.keras.layers.UpSampling2D((2, 2))(conv4)
        concat5 = tf.keras.layers.Concatenate()([conv2, up5])
        conv5 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(concat5)
        conv5 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
        
        up6 = tf.keras.layers.UpSampling2D((2, 2))(conv5)
        concat6 = tf.keras.layers.Concatenate()([conv1, up6])
        conv6 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat6)
        conv6 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
        
        outputs = tf.keras.layers.Conv2D(self.num_classes, (1, 1), activation='softmax')(conv6)
        
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        
        return model
