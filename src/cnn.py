import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

SAMPLE_LABEL = {0, 1, 2, 3}
OTHER_FOOD = 1
LABEL_OTHER_FOOD = 4
NBR_OTHER_FOOD = 500
REPO_O = 'photos'
REPO = 'photos2'
EPOCHS = 2
MODEL_SAVE = 0
MODEL_SAVE_REPO = './resnet50_5Label6'

NB_LABEL = len(SAMPLE_LABEL) + OTHER_FOOD

# 'C:/Users/maxen/switchdrive/HEIAFR/CNNAppWeb/posts/saveModel'



def train_and_save_model():
    split_sample()
    df_train = dframe('training')
    df_val = dframe('validation')

    df_train.head()

    # Generate batches and augment the images
    train_generator = sample_augmentation("training", df_train)
    val_generator = sample_augmentation("validation", df_val)

    model = initialize_model(NB_LABEL)

    train_model(model, train_generator, val_generator, EPOCHS)

    evaluate_model(model)

    empty_repo()

    if MODEL_SAVE == 1:
        model.save(MODEL_SAVE_REPO)


# transforms an image into a numpy array with a label
def dframe(dtype):
    X = []
    y = []
    path = f'../ressources/{REPO}/' + dtype + '/'
    for i in os.listdir(path):
        # Image
        X.append(i)
        # Label
        y.append(i.split('_')[0])
    X = np.array(X)
    y = np.array(y)
    df = pd.DataFrame()
    df['filename'] = X
    df['label'] = y
    return df


# separates the dataset into three categories: training, validation, evaluation
def split_sample():
    for i in os.listdir(f'../ressources/{REPO_O}/training'):
        if i != '.ipynb_checkpoints':
            if int(i.split('_')[0]) in SAMPLE_LABEL or (OTHER_FOOD == 1 and int(i.split('_')[0]) == LABEL_OTHER_FOOD and int(i.split('_')[1].split('.')[0]) < NBR_OTHER_FOOD):
                with Image.open(f'../ressources/{REPO_O}/training/' + i) as image:
                    a = random.randint(0, 9)

                    if a < 7:
                        image.save(f'../ressources/{REPO}/training/' + i)
                    elif a < 9:
                        image.save(f'../ressources/{REPO}/validation/' + i)
                    else:
                        image.save(f'../ressources/{REPO}/evaluation/' + i)


def x_train():
    instances = np.empty((2718,224,224,3))
    j = 0
    for i in os.listdir(f'../ressources/photosR/training'):
      if i != '.ipynb_checkpoints':
            with Image.open(f'../ressources/photosR/training/' + i) as image:
               image2 = np.array(image)
               instances[j] = image2
               j += 1
    return instances



# create new samples from existing ones
def sample_augmentation(sample_type, sample):
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
    )

    #train = x_train()

    #datagen.fit(train)

    generator = datagen.flow_from_dataframe(
        sample,
        directory=f'../ressources/{REPO}/{sample_type}/',
        x_col='filename',
        y_col='label',
        target_size=(224, 224),
    )

    return generator


# create the neural network with the different layers
def initialize_model(nbLabel):
    # Initialize the Pretrained Model
    feature_extractor = ResNet50(weights='imagenet',
                                 input_shape=(224, 224, 3),
                                 include_top=False)

    # Set this parameter to make sure it's not being trained
    feature_extractor.trainable = False

    # Set the input layer
    input_ = tf.keras.Input(shape=(224, 224, 3))

    # Set the feature extractor layer
    x = feature_extractor(input_, training=False)

    # Set the pooling layer
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # let's add a fully-connected layer
    #x = tf.keras.layers.Dense(128, activation='relu')(x)

    # Set the final layer with sigmoid activation function
    output_ = tf.keras.layers.Dense(nbLabel, activation='softmax')(x)

    # Create the new model object
    model = tf.keras.Model(input_, output_)

    return model


# drives the model with the samples
def train_model(model, train_generator, val_generator, epochs):
    # Compile it
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Print The Summary of The Model
    model.summary()

    model.fit(train_generator, epochs=epochs, validation_data=val_generator)


# evaluates the model with the evaluation set and prints out the accuracy statistics
def evaluate_model(model):
    y_true = []
    y_pred = []

    for i in os.listdir(f'../ressources/{REPO}/evaluation'):
        if i != '.ipynb_checkpoints':
            img = Image.open(f'../ressources/{REPO}/evaluation/' + i)
            img = img.resize((224, 224))
            img = np.array(img)
            img = np.expand_dims(img, 0)

            y_true.append(int(i.split('_')[0]))

            predictions = model.predict(img)

            score = tf.nn.softmax(predictions[0]).numpy()

            y_pred.append(np.argmax(score))

    print(classification_report(y_true, y_pred))
    print()
    print(confusion_matrix(y_true, y_pred))


# empty training, validation and evaluation set sample
def empty_repo():
    for i in ['training', 'evaluation', 'validation']:
        for file in os.listdir(f'../ressources/{REPO}/{i}'):
            if file != '.ipynb_checkpoints':
                os.remove(f'../ressources/{REPO}/{i}/' + file)