import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
import streamlit as st

def get_analis():
    st.header("Disease prediction")
    df = pd.read_csv('desiseList.csv')
    desise_list = df['prognosis'].unique()
    df = df.drop(labels='prognosis', axis=1)
    symptops_list = []
    for i in df:
        symptops_list.append(i)

    symptoms = st.multiselect("pick sumptops", symptops_list)
    symptoms_for_ai = np.zeros(len(symptops_list))
    for i in symptoms:
        symptoms_for_ai[symptops_list.index(i)] = 1
    if not np.array_equal(symptoms_for_ai, np.zeros(len(symptops_list))):
        model = keras.models.load_model('DesisePredictions')
        prediction = model.predict(np.expand_dims(symptoms_for_ai, axis=0))[0]
        desise_type = np.where(prediction == max(prediction))[0][0]
        st.write(desise_list[desise_type])


