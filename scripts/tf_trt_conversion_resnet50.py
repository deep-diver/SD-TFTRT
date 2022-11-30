import numpy as np

import tensorflow as tf
from tensorflow.experimental import tensorrt
from tensorflow.keras.applications.resnet50 import ResNet50

model = ResNet50(weights='imagenet')
model.save('resnet50_saved_model')

params = tensorrt.ConversionParams(precision_mode='FP16')

converter = tf.experimental.tensorrt.Converter(
    input_saved_model_dir="resnet50_saved_model", conversion_params=params)
converter.convert()
converter.save('resnet50_saved_model_tftrt')
