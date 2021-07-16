import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from keras import layers
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


train_filepath = '../input/a-large-scale-fish-dataset/Fish_Dataset/Fish_Dataset'
test_filepath = '../input/a-large-scale-fish-dataset/NA_Fish_Dataset'

# Hyperparameters
epochs = 20
batch_size = 32
image_size = (224, 224)
num_classes = 9

def get_image_data(filepath):
    class_labels = []
    image_links = []
    for class_directory in tqdm(sorted(os.listdir(filepath))):
        label_dir = os.path.join(filepath, class_directory)
        if os.path.isdir(label_dir): 
            for rgb_directory in os.listdir(label_dir):
                if 'GT' not in rgb_directory:
                    image_dir = os.path.join(label_dir, rgb_directory)
                    for img in os.listdir(image_dir):
                        image_links.append(os.path.join(image_dir, img))
                        class_labels.append(class_directory)

    return class_labels, image_links

labels, images = get_image_data(train_filepath)
df = pd.DataFrame({'image': images, 'label': labels})
df.head()
df.describe()
labels_to_classes = {label:i for i, label in enumerate(df.label.unique())}
labels_to_classes
train_data, val_data = train_test_split(df, test_size=0.2)
train_data.label.unique()
val_data.describe()

# Perform image visualization
fig = plt.figure(figsize=(10, 10))
for i, fish in enumerate(train_data.label.unique()):
    img_index = df.loc[df['label'] == fish].index[0]
    image = plt.imread(df.iloc[img_index, 0])
    plt.subplot(3, 3, i+1)
    plt.title(df.iloc[img_index, 1])
    plt.imshow(image)
    plt.axis('off')

# Load image data
data_generator = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
)

test_generator = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
)

train_generator = data_generator.flow_from_dataframe(
    dataframe=train_data,
    x_col='image',
    y_col='label',
    target_size=image_size,
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    class_mode='categorical'
)

val_generator = data_generator.flow_from_dataframe(
    dataframe=val_data,
    x_col='image',
    y_col='label',
    target_size=image_size,
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    class_mode='categorical'
)

test_data = test_generator.flow_from_directory(
    test_filepath,
    shuffle=False,
    seed=42
)

# Free unused memory
del images
del labels
gc.collect()

# CNN model
base_model = ResNet50V2(input_shape=image_size + (3,), include_top=False, weights='imagenet', pooling='avg')
base_model.trainable = False

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

model = make_model(image_size + (3,), num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(train_generator, epochs=epochs, validation_data=val_generator)
model.save('fish-cnn.h5')

hist_df = pd.DataFrame(history.history)
hist_df[['accuracy', 'val_accuracy']].plot()
hist_df[['loss', 'val_loss']].plot()

predictions = model.predict(test_data)
predictions = np.argmax(predictions, axis=1)
accuracy = accuracy_score(test_data.classes, predictions)
classification_report(test_data.classes, predictions)