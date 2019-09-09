from sklearn.cluster import KMeans
from collections import Counter
import cv2  # for resizing image

import wcag_contrast_ratio as contrast
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os, sys
from os import listdir, makedirs
from os.path import join, exists, expanduser
import scipy as sc



def get_dominant_color(image, k=4, image_processing_size=None):
    """
    takes an image as input
    returns the dominant color of the image as a list

    dominant color is found by running k means on the
    pixels & returning the centroid of the largest cluster

    processing time is sped up by working with a smaller image;
    this resizing can be done with the image_processing_size param
    which takes a tuple of image dims as input


    """
    # resize image if new dims provided
    if image_processing_size is not None:
        image = cv2.resize(image, image_processing_size,
                           interpolation=cv2.INTER_AREA)

    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # cluster and assign labels to the pixels
    clt = KMeans(n_clusters=k)
    labels = clt.fit_predict(image)

    # count labels to find most popular
    label_counts = Counter(labels)

    # subset out most popular centroid
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

    return list(dominant_color)


def get_mean_color(image):
    """get mean color for one picture"""
    mean = np.sum(np.sum(image,axis=0),axis=0)
    mean[:] = mean[:] / (image.shape[0]*image.shape[1])

    return (mean)


def black_or_white(color):
    """give the contrast between white/black and color and give wich of black
    or white give the best contrast (return black or white)"""

    # input are number between 0 and 1
    black = (1, 1, 1)
    white = (0, 0, 0)


    color[:] = color[:] / 255

    # calculate the contrast between black and color
    b_c = contrast.rgb(black, color)

    # calculate the contrast between white and color
    w_c = contrast.rgb(black, color)

    output_color = 'empty'

    if (b_c > w_c):
        output_color = 'black'
    else:
        output_color = 'white'

    return (output_color)


def recommand(image,dataset):
    """recommand similar pictures"""
    if (dataset == 'perso'):
        n = 6
        labels = pd.read_csv('./static/images/labels_perso.csv')
        matrix = np.load('./static/images/matrix_perso.npy')

    if (dataset == 'autre'):
        labels = pd.read_csv('./static/images/labels_autre.csv')
        matrix = np.load('./static/images/matrix_autre.npy')
        n = 10

    # norme euclidienne :
    index = labels[(labels['labels'] == str(image))].iloc[0,1]
    # nom de l'image :


    diff = np.zeros((matrix.shape[0], matrix.shape[1]))

    diff[:, :] = matrix[:, :] - matrix[index, :]
    diff[:, :] = np.multiply(diff, diff)

    norm = np.sum(diff, axis=1)

    norm_sort = np.sort(norm, axis=-1)
    norm_argsort = np.argsort(norm, axis=-1)

    recom_pict = []
    for i in range(len(norm_argsort)):
        if (i< n):
            recom_pict.append(labels.iloc[norm_argsort[i], 2])

    return (recom_pict)


def recommand_couleur(image,dataset):
    """recommand similar pictures"""
    if (dataset == 'perso'):
        n = 30
        labels = pd.read_csv('./static/images/labels_perso.csv')
        matrix = np.load('./static/images/matrix_color_perso.npy')
        

    if (dataset == 'autre'):
        labels = pd.read_csv('./static/images/labels_autre.csv')
        matrix = np.load('./static/images/matrix_color_autre.npy')
        n = 40

    # norme euclidienne :
    index = labels[(labels['labels'] == str(image))].iloc[0, 1]
    # nom de l'image :


    diff = np.zeros((matrix.shape[0], matrix.shape[1]))

    diff[:, :] = matrix[:, :] - matrix[index, :]
    diff[:, :] = np.multiply(diff, diff)

    norm = np.sum(diff, axis=1)

    norm_sort = np.sort(norm, axis=-1)
    norm_argsort = np.argsort(norm, axis=-1)

    recom_pict = []
    for i in range(len(norm_argsort)):
        if (i < n):
            recom_pict.append(labels.iloc[norm_argsort[i], 2])

    return (recom_pict)


def plot_value_counts(col_name, df):
    values_count = pd.DataFrame(df[col_name].dropna().value_counts())
    # print (values_count.shape)
    values_count.columns = ['count']
    # convert the index column into a regular column.
    values_count[col_name] = [str(i) for i in values_count.index]
    # add a column with the percentage of each data point to the sum of all data points.
    values_count['percent'] = values_count['count'].div(values_count['count'].sum()).multiply(100).round(2)
    # change the order of the columns.
    values_count = values_count.reindex([col_name, 'count', 'percent'], axis=1)
    values_count.reset_index(drop=True, inplace=True)
    return (values_count)


def predict(predictions, breed):
    arg_predictions = [np.argmax(prediction) for prediction in predictions]
    race = breed[arg_predictions[0]]
    prob = predictions[0, arg_predictions[0]]
    predictions[0, arg_predictions[0]] = 0

    return (race, prob, predictions)


def predict_dog():
    from PIL import Image

    from keras.applications.xception import Xception
    from keras.applications.xception import preprocess_input as preprocess_input_Xception
    from keras.models import load_model
    from keras.layers.pooling import GlobalAveragePooling2D
    from keras.layers.merge import Concatenate
    from keras.layers import Input, Dense
    from keras.layers.core import Dropout, Activation
    from keras.callbacks import ModelCheckpoint
    from keras.layers.normalization import BatchNormalization
    from keras.models import Model
    from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

    def input_branch(input_shape=None):
        size = int(input_shape[2] / 4)

        branch_input = Input(shape=input_shape)
        branch = GlobalAveragePooling2D()(branch_input)
        branch = Dense(size, use_bias=False, kernel_initializer='uniform')(branch)
        branch = BatchNormalization()(branch)
        branch = Activation("relu")(branch)
        return branch, branch_input

    Xception_branch, Xception_input = input_branch(input_shape=(10, 10, 2048))

    net = Dropout(0.34)(Xception_branch)
    net = Dense(1024, use_bias=False, kernel_initializer='uniform')(net)
    net = BatchNormalization()(net)
    net = Activation("relu")(net)
    net = Dropout(0.34)(net)
    net = Dense(120, kernel_initializer='uniform', activation="softmax")(net)

    model = Model(inputs=[Xception_input], outputs=[net])
    model.summary()

    model.load_weights('classification.h5')
    breed = np.load('names.npy')



    X_test = np.zeros((1, 299, 299, 3))

   #  img = load_img('./static/images/image_test.png', target_size=(299, 299))  # this is a PIL image
    img = Image.open('./static/images/image_test.png')
    img.thumbnail((299, 299))
    matrix = img_to_array(img)
    X_test[0, 0:matrix.shape[0], 0:matrix.shape[1], :] = matrix[:, :, :]
    #X_test[0, :, :, :] = matrix[:, :, :]

    input_test_Xception = preprocess_input_Xception(X_test)

    test_Xception = Xception(weights='imagenet', include_top=False).predict(input_test_Xception)


    predictions = model.predict([test_Xception])

    dog_breed = []
    dog_breed_prob = []
    for i in range(3):
        r, p, predictions = predict(predictions=predictions, breed=breed)
        predictions = predictions

        print (p)
        dog_breed.append(r[10::])
        dog_breed_prob.append(p * 100)

    return (dog_breed, dog_breed_prob)









