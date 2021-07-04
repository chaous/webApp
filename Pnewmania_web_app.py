import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
import streamlit as st
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import img_to_array
import gdown
import TB_webPage
from PIL import Image
import urllib.request






def get_analis():
    st.title("Pneumonia detection")
    uploaded_file = st.file_uploader("Choose a photo")
    click = st.button("delete model")
    if click:
        if os.path.exists("Pnevmania_detection/variables/variables.data-00000-of-00001"):
            os.remove("Pnevmania_detection/variables/variables.data-00000-of-00001")
            st.write("delited")
        else:
            st.write("already doesn't exist")
    if uploaded_file is not None:
        if not os.path.exists("Pnevmania_detection/variables/variables.data-00000-of-00001"):
            output = 'Pnevmania_detection/variables/variables.data-00000-of-00001'
            url = "https://github.com/chaous/webApp/releases/download/1.0.0/variables.data-00000-of-00001"
            st.write("downloading a model this might take a while")
            #gdown.download(url, output, quiet=False)
            urllib.request.urlretrieve(url, output)
            st.write("Done")
        model = keras.models.load_model("Pnevmania_detection")
        image = Image.open(uploaded_file, mode='r').convert('RGB')
        image = image.resize((224, 224), 3)
        #st.image(image)
        img_array = img_to_array(image)
        #img_array = img_array.reshape( 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        #if img_array.shape != (1, 224, 224, 3):
         #   img_array.resize(1, 224, 224, 3)
        #st.write(img_array.shape)
        img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
        #test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input), image)
        #image = tf.keras.applications.vgg16.preprocess_input(image)
        #image = img_to_array(image)
        #image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        #image = tf.keras.applications.vgg16.preprocess_input(image)
        predictions = model.predict(img_array)
        st.image(image, caption='Uploaded MRI.', use_column_width=True)
        st.write("Classifying...")
        if predictions[0][0] > predictions[0][1]:
            st.write("It's normal")
        else:
            st.write("It's pneumonia")
        #st.write(predictions)