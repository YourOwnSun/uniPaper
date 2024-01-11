import numpy as np
from keras import optimizers
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
import os

from matplotlib import pyplot as plt
from skimage import io
from sklearn.model_selection import train_test_split

from preprocessing import labels_encoding
from metrics import *
from model import get_model

EPOCHS = 50
BATCH_SIZE = 16


def get_dataset(images_path, retina_path, pigment_path):

    images = [io.imread(images_path + im, as_gray=True) for im in os.listdir(path=images_path)]
    retina = [io.imread(retina_path + r, as_gray=True) for r in os.listdir(path=retina_path)]
    pigment = [io.imread(pigment_path + p, as_gray=True) for p in os.listdir(path=pigment_path)]

    images = np.array(images)
    retina = np.array(retina)
    pigment = np.array(pigment)
    train_images = images.reshape((images.shape[0], 256, 256, 1))
    train_labels = labels_encoding(retina, pigment)

    train_images = train_images.astype('float32')
    train_labels = train_labels.astype('float32')

    return train_test_split(train_images, train_labels, test_size=0.2, random_state=42)


def save_reports(history):
    epochs = range(1, 50 + 1)

    plt.plot(epochs, history.history['dice_coefficient'][0:50], 'r', label='Training dice_coefficient')
    plt.plot(epochs, history.history['val_dice_coefficient'][0:50], 'b', label='Validation dice_coefficient')
    plt.legend()
    plt.grid()
    plt.ylabel('dice_coefficient')
    plt.xlabel('epoch')
    plt.show()
    plt.savefig('reports/dice_2.png')

    plt.plot(epochs, history.history['loss'][0:50], 'r', label='Training loss')
    plt.plot(epochs, history.history['val_loss'][0:50], 'b', label='Validation loss')
    plt.legend()
    plt.grid()
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
    plt.savefig('reports/loss_2.png')


def train_model():
    model = get_model()

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=customized_loss,
                  metrics=['accuracy', dice_coefficient], sample_weight_mode='temporal')
    lr_reducer = ReduceLROnPlateau(factor=0.5, cooldown=0, patience=6, min_lr=0.5e-6)
    csv_logger = CSVLogger('Model/ModelSettings/Model.csv')
    model_checkpoint = ModelCheckpoint("Model/ModelSettings/2_layers_model.hdf5", monitor='val_loss', verbose=1,
                                       save_best_only=True)

    X, y = get_dataset('data/dataset/image_for_training/', 'data/dataset/labels_for_training/')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    h = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test),
                  callbacks=[lr_reducer, csv_logger, model_checkpoint])

    save_reports(h)
    model.load_weights('models/2_layers_model.hdf5')
    model.save('models/2_layers_model.h5')


if __name__ == '__main__':
    train_model()
