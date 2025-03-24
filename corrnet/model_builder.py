# corrnet/model_builder.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_vgg16_model(input_shape=(32, 32, 3), num_classes=10):
    """Build a VGG16 model for CIFAR-10."""
    model = tf.keras.applications.VGG16(weights=None, include_top=False, input_shape=input_shape)
    x = model.output
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=model.input, outputs=x)
