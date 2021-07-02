import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
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




def get_analis():
    st.title("Brain tumor detection")
    uploaded_file = st.file_uploader("Choose a photo")
    click = st.button("delete modedel")
    if click:
        if os.path.exists("Brein_Tumor_detection/variables/variables.data-00000-of-00001"):
            os.remove("Brein_Tumor_detection/variables/variables.data-00000-of-00001")
            st.write("delited")
        else:
            st.write("already doesn't exist")
    if uploaded_file is not None:
        if not os.path.exists("Brein_Tumor_detection/variables/variables.data-00000-of-00001"):
            output = 'Brein_Tumor_detection/variables/variables.data-00000-of-00001'
            url = "https://drive.google.com/uc?export=download&confirm=no_antivirus&id=1KhZlkRDgLauVoEzJmof3SjRKsb2BlWlp"
            st.write("dowlading a model this might take a while")
            gdown.download(url, output, quiet=False)
            st.write("Done")
        model = keras.models.load_model("Brein_Tumor_detection")
        image = Image.open(uploaded_file)
        image = image.resize((224, 224), 3).convert('RGB')
        #st.image(image)
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        #if img_array.shape != (1, 224, 224, 3):
        #    img_array.reshape(1, 224, 224, 3)
        img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
        #test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input), image)
        #image = tf.keras.applications.vgg16.preprocess_input(image)
        #image = img_to_array(image)
        #image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        #image = tf.keras.applications.vgg16.preprocess_input(image)
        predictions = model.predict(img_array)
        st.image(image, caption='Uploaded MRI.', use_column_width=True)
        st.write("Classifying as")
        #st.write(predictions[0][3] == max(predictions[0]))
        if predictions[0][0] == max(predictions[0]):
            st.write("It's pituitary tumor")
        if predictions[0][1] == max(predictions[0]):
            st.write("It's normal")
        if predictions[0][2] == max(predictions[0]):
            st.write("It's meningioma tumor")
        if predictions[0][3] == max(predictions[0]):
            st.write("It's glioma tumor")



