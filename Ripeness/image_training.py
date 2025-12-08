import random
from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.applications.resnet import ResNet50
from tensorflow.keras.utils import to_categorical
import cv2
import splitfolders
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
import tensorflow as tf
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


input_folder = "C:/Fruits Jackfruit/fruit_ripeness_dataset/archive (1)/dataset/dataset/train"
output_folder = "C:/Users/Mohit.K/OneDrive\Desktop/Fruits Jackfruit/Output folder"

split_ratio = (0.8, 0.1, 0.1)
splitfolders.ratio(
    input_folder,
    output=output_folder,
    seed=500,
    ratio=split_ratio,
    group_prefix=None,
)

img_size = (224, 224)
batch_size = 32

# To define parameters
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_dir = os.path.join(output_folder, 'train')
val_dir = os.path.join(output_folder, 'val')
test_dir = os.path.join(output_folder, 'test')

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

valid_data = valid_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

images, labels = next(valid_data)

""" idx = random.randint(0, images.shape[0]-1)
plt.imshow(images[idx])
 plt.show() """

base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=(img_size[0], img_size[1], 3))

# Freeze the convolution base
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),  # Rectified Linear Unit
    layers.Dropout(0.5),
    layers.Dense(9, activation='softmax')
])


model.compile(optimizer='adam',  # Adaptive Moment Estimation
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, epochs=30, validation_data=valid_data)

test_loss, test_accuracy = model.evaluate(test_data)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

class_names = {0: 'Fresh_apple', 1: 'Fresh_banana', 2: 'Fresh_orange',
               3: 'Rotten_apple', 4: 'Rotten_banana', 5: 'Rotten_orange',
               6: 'Apple_unripe', 7: 'Banana_unripe', 8: 'Orange_unripe'}

model.save('fruit_image_training.keras')
