import keras

from simple_multi_unet_model import multi_unet_model #Uses softmax

from keras.utils import normalize
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from PIL import Image
import segmentation_models as sm


matplotlib.use('TkAgg')



originalPath = 'C:/Code/uniPaper/data/learning/Original/'
intraPath = 'C:/Code/uniPaper/data/learning/Intra/'
subPath = 'C:/Code/uniPaper/data/learning/Sub/'
pesPath = 'C:/Code/uniPaper/data/learning/PES/'


#Resizing images, if needed
SIZE_X = 256
SIZE_Y = 256
n_classes = 4 #Number of classes for segmentation

#Capture training image info as a list
train_images = []

original_paths = [(originalPath + im) for im in os.listdir(path=originalPath)]

for img_path in original_paths:
    img = Image.open(img_path).convert('L')

    img = img.resize((256, 256))

    train_images.append(np.array(img))
    train_images.append(np.array(img.transpose(Image.FLIP_LEFT_RIGHT)))


#Convert list to array for machine learning processing
train_images = np.array(train_images)

#Capture mask/label info as a list
train_masks = []

intra_paths = [(intraPath + im) for im in os.listdir(path=intraPath)]
sub_paths = [(subPath + im) for im in os.listdir(path=subPath)]
pes_paths = [(pesPath + im) for im in os.listdir(path=pesPath)]

for img_path in intra_paths:
    img = Image.open(img_path).convert('L')

    img=img.resize((256, 256))
    flip=img.transpose(Image.FLIP_LEFT_RIGHT)

    img = np.array(img)
    flip = np.array(flip)

    img[img > 0] = 1
    flip[flip > 0] = 1

    train_masks.append(img)
    train_masks.append(flip)


for i in range(len(sub_paths)):
    img = Image.open(sub_paths[i]).convert('L')

    img = img.resize((256, 256))
    flip = img.transpose(Image.FLIP_LEFT_RIGHT)

    img = np.array(img)
    flip = np.array(flip)

    for x in range(len(img)):
        for y in range(len(img[x])):
            if img[x][y] > 0: train_masks[i*2][x][y] = 2

    for x in range(len(flip)):
        for y in range(len(flip[x])):
            if flip[x][y] > 0: train_masks[(i*2) + 1][x][y] = 2


for i in range(len(pes_paths)):
    img = Image.open(pes_paths[i]).convert('L')

    img = img.resize((256, 256))
    flip = img.transpose(Image.FLIP_LEFT_RIGHT)

    img = np.array(img)
    flip = np.array(flip)

    for x in range(len(img)):
        for y in range(len(img[x])):
            if img[x][y] > 0: train_masks[i*2][x][y] = 3

    for x in range(len(flip)):
        for y in range(len(flip[x])):
            if flip[x][y] > 0: train_masks[(i*2) + 1][x][y] = 3

#Convert list to array for machine learning processing
train_masks = np.array(train_masks)

###############################################
# Encode labels... but multi dim array so need to flatten, encode and reshape
from sklearn.preprocessing import LabelEncoder


print(np.unique(train_images))
print(np.unique(train_masks))


#################################################
train_images = np.expand_dims(train_images, axis=3)
train_images = normalize(train_images, axis=1)

train_masks_input = np.expand_dims(train_masks, axis=3)

# Create a subset of data for quick testing
# Picking 10% for testing and remaining for training
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_images, train_masks_input, test_size=0.20, random_state=0)


print("Class values in the dataset are ... ", np.unique(y_train))  # 0 is the background/few unlabeled

from keras.utils import to_categorical

train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

###############################################################

###############################################################
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                  classes=np.unique(np.ravel(train_masks, order='C')),
                                                  y=np.ravel(train_masks, order='C'))
print("Class weights are...:", class_weights)

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]


import random
for i in range(15):
    image_number = random.randint(0, len(X_train) - 1)
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(X_train[image_number], cmap='gray')
    plt.subplot(122)
    plt.imshow(y_train[image_number], cmap='gray')
    plt.show()



def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)


model = get_model()
model.compile(optimizer='adam', loss=keras.losses.CategoricalFocalCrossentropy(), metrics=[keras.metrics.Accuracy(), keras.metrics.IoU(num_classes=n_classes, target_class_ids=[1, 2, 3])])
model.summary()

# If starting with pre-trained weights.
# model.load_weights('???.hdf5')

history = model.fit(X_train, y_train_cat,
                    batch_size=16,
                    verbose=1,
                    epochs=50,
                    validation_data=(X_test, y_test_cat),
                    class_weight={i: class_weights[i] for i in range(n_classes)},
                    shuffle=False)

modelpath = 'C:/Code/uniPaper/savedmodels/simpleWithMirrorAndFocalNewData.hdf5'

model.save(modelpath)
#model.save('sandstone_50_epochs_catXentropy_acc_with_weights.hdf5')
############################################################
#Evaluate the model
	# evaluate model
test_loss, test_acc, *is_anything_else_being_returned = model.evaluate(X_test, y_test_cat)
print("Accuracy is = ", (test_acc * 100.0), "%")

print("Loss is = ", (test_loss * 100.0), "%")

print("ELSE: ", is_anything_else_being_returned)



###
#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


print("KEYS: ", history.history.keys())

acc = history.history['io_u']
#acc = history.history['accuracy']
val_acc = history.history['val_io_u']
#val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training IOU')
plt.plot(epochs, val_acc, 'r', label='Validation IOU')
plt.title('Training and validation IOU')
plt.xlabel('Epochs')
plt.ylabel('IOU')
plt.legend()
plt.show()


##################################
#model = get_model()
model.load_weights(modelpath)
#model.load_weights('sandstone_50_epochs_catXentropy_acc_with_weights.hdf5')

#IOU
y_pred=model.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)

##################################################

#Using built in keras function
from keras.metrics import MeanIoU
IOU_keras = MeanIoU(num_classes=n_classes)
IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0] + values[2,0] + values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1] + values[2,1] + values[3,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2] + values[1,2] + values[3,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3] + values[1,3] + values[2,3])


print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)

plt.imshow(train_images[0, :,:,0], cmap='gray')
plt.imshow(train_masks[0], cmap='gray')
#######################################################################
#Predict on a few images
#model = get_model()
#model.load_weights('???.hdf5')


import random
test_img_number = random.randint(0, len(X_test) - 1)
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_input=np.expand_dims(test_img, 0)


prediction = (model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]


c = matplotlib.colors.ListedColormap(['black', 'red', 'white', 'blue'])
n = matplotlib.colors.Normalize(vmin=0, vmax=3)


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap=c, norm=n)
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img, cmap=c, norm=n)
plt.show()
