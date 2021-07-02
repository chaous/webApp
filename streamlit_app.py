import streamlit as st
import TB_webPage
import Pnewmania_web_app
import brein_Tumor_prediction
import desise_predictions

PAGES = {
    "TB detection": TB_webPage,
    "Pneumonia detection": Pnewmania_web_app,
    "Brain tumor detection": brein_Tumor_prediction,
    "Disease prediction": desise_predictions
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.get_analis()









