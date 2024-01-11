import numpy as np
from PIL import Image
from keras.models import load_model

from metrics import *
from preprocessing import preprocessing_images, segmentation_result, resize_file, segmentation_result_2
import cv2

height, width = 256, 256


def predict(src: str, dest: str):
    images = preprocessing_images(src)
    p_model = load_model('../../models/Pigment_model_3.h5', custom_objects={'customized_loss': customized_loss,
                                                                           'dice_coefficient': dice_coefficient})
    for i, image in enumerate(images):
        prediction = p_model.predict(image)
        prediction = np.squeeze(prediction, axis=0)
        color = segmentation_result_2(prediction)
        processed_image = cv2.medianBlur(color, 5)
        img = Image.fromarray(processed_image, 'RGB')
        img.save(dest + str(i + 1) + '.png')


if __name__ == '__main__':
    #predict('../../data/dmo/inject/3/Images/', '../../data/dmo/inject/3/Predictions2/')
    predict('../../data/dmo/laser/3/Images/', '../../data/dmo/laser/3/Predictions2/')



