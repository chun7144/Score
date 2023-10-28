import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('./saved_model.pkl', 'rb') as file:
        ScoreML_loaded = pickle.load(file)
    return ScoreML_loaded

ScoreML_loaded = load_model()

def show_predict_page():
    from Score_App import X_mean
    from Score_App import X_std
    st.title("Score Prediction")

    st.write("""### We need some information to predict the Score """)

    time_to_study = st.number_input('공부시간 입력', 1, 100)

    ok = st.button("Calculate Score")

    if ok:
        value = ScoreML_loaded.predict(np.array([[(time_to_study-X_mean)/X_std]]))
        st.write('점수', value)
