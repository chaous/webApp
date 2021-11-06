import pandas as pd
import numpy as np
from tensorflow import keras
import streamlit as st
import os
import gdown

def get_number(array, element):
    unique_values = array.unique()
    return np.where(element == unique_values)[0][0]


def get_analis():

    st.header("Stroke prediction")
    df = pd.read_csv("stroke.csv")
    #df_numbers = pd.read_csv()
    data = np.zeros(10)
    gender = st.radio("Your gender", df['gender'].unique())
    data[0] = get_number(df['gender'], gender)
    data[1] = st.number_input('age', min_value=0, max_value=125)
    data[2] = int(st.radio('hypertension', [1, 0]))
    data[3] = int(st.radio('heart disease', [1, 0]))
    data[4] = get_number(df['ever_married'], st.radio('ever married', df['ever_married'].unique()))
    data[5] = get_number(df['work_type'], st.radio('work type', df['work_type'].unique()))
    data[6] = get_number(df['Residence_type'], st.radio('Residence type', df['Residence_type'].unique()))
    data[7] = st.number_input('avg glucose level')
    data[8] = st.number_input('bmi')
    data[9] = get_number(df['smoking_status'], st.radio('smoking status', df['smoking_status'].unique()))
    click = st.button("predict")
    if not os.path.exists("StrokePredictions/variables/variables.data-00000-of-00001"):
        #output = 'DesisePredictions/variables/variables.data-00000-of-00001'
        url = "https://drive.google.com/uc?export=download&confirm=no_antivirus&id=1vDAL7RxR5sOlGsT5Igcdfh5EvprLwS-y"
        st.write("downloading a model this might take a while")
        gdown.download(url, 'StrokePredictions/variables/variables.data-00000-of-00001', quiet=False)
    if click:
        model = keras.models.load_model('StrokePredictions')
        prediction = model.predict(np.expand_dims(data, axis=0))[0]
        if prediction[0] > prediction[1]:
            st.write("You probably won't have stroke")
        else:
            st.write("yoi probably will have stroke")

