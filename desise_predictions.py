import pandas as pd
import numpy as np
from tensorflow import keras
import streamlit as st
import os
import gdown

def get_analis():
    st.header("Disease prediction")
    df = pd.read_csv('desiseList.csv')
    desise_list = df['prognosis'].unique()
    df = df.drop(labels='prognosis', axis=1)
    symptops_list = []
    for i in df:
        symptops_list.append(i)

    symptoms = st.multiselect("pick symptoms", symptops_list)
    symptoms_for_ai = np.zeros(len(symptops_list))
    if not os.path.exists("DesisePredictions/variables/variables.data-00000-of-00001"):
        #output = 'DesisePredictions/variables/variables.data-00000-of-00001'
        url = "https://drive.google.com/uc?export=download&confirm=no_antivirus&id=1kY0I1cTkvpaQsPXIjdKhzRtu5mTXJyys"
        st.write("downloading a model this might take a while")
        gdown.download(url, 'DesisePredictions/variables/variables.data-00000-of-00001', quiet=False)
    for i in symptoms:
        symptoms_for_ai[symptops_list.index(i)] = 1
    if not np.array_equal(symptoms_for_ai, np.zeros(len(symptops_list))):
        model = keras.models.load_model('DesisePredictions')
        prediction = model.predict(np.expand_dims(symptoms_for_ai, axis=0))[0]
        desise_type = np.where(prediction == max(prediction))[0][0]
        st.write(desise_list[desise_type])


