import keras_cv
from tensorflow.experimental import tensorrt

from keras_cv.models.generative.stable_diffusion.stable_diffusion import TextEncoder
from keras_cv.models.generative.stable_diffusion.stable_diffusion import DiffusionModel
from keras_cv.models.generative.stable_diffusion.stable_diffusion import Decoder

MAX_PROMPT_LENGTH = 77

IMG_HEIGHT = 512
IMG_WIDTH = 512

encoder = TextEncoder(MAX_PROMPT_LENGTH)
stable_diffusion = DiffusionModel(IMG_HEIGHT, IMG_WIDTH, MAX_PROMPT_LENGTH)
decoder = Decoder(IMG_HEIGHT, IMG_WIDTH)

encoder.save('encoder')
stable_diffusion.save('stable_diffusion')
decoder.save('decoder')

trt_params = tensorrt.ConversionParams(precision_mode='FP16')

converter = tensorrt.Converter(
    input_saved_model_dir="encoder", conversion_params=trt_params)
converter.convert()
converter.save('encoder_trt')

converter = tensorrt.Converter(
    input_saved_model_dir="stable_diffusion", conversion_params=trt_params)
converter.convert()
converter.save('stable_diffusion_trt')

converter = tensorrt.Converter(
    input_saved_model_dir="decoder", conversion_params=trt_params)
converter.convert()
converter.save('decoder_trt')
