import streamlit as st
import pandas as pd
import pickle

data = pd.read_csv(r'C:\Users\shrav\OneDrive\Desktop\Python VS\DA Project\heart_2020_cleaned.csv')

age = data['AgeCategory'].unique()
race = data['Race'].unique()
sex = data['Sex'].unique()
genHealth =  data['GenHealth'].unique()


st.title('in just a few seconds, you can calculate your risk of developing heart disease!')

st.sidebar.title('Please, fill your informations to predict your heart condition')

##bmi = st.sidebar.selectbox('Select your BMI', ['Normal weight BMI (18.5-25)', 'Underweight BMI (<18.5)' ,'Overweight BMI (25-30)'])

bmi = st.sidebar.number_input('Select your BMI')

age = st.sidebar.selectbox('Select your age', age)

race = st.sidebar.selectbox('Select your race', race)

gender = st.sidebar.selectbox('Select your Gender', sex)

smoker = st.sidebar.selectbox('Have you smoked at least 100 cigarettes in your entire life?', ['No', 'Yes'])

drinker = st.sidebar.selectbox('Are you a Heavy drinker?', ['No', 'Yes'])

stroke = st.sidebar.selectbox('Did you have a Stroke', ['No', 'Yes'])

genHealth = st.sidebar.selectbox('Hows your General health', genHealth )

sleep = st.sidebar.number_input('Hours of sleep per 24h', 0, 23, 7)

physicalHealth = st.sidebar.number_input('For how many days during the past 30 days was your physical health not good?', 0, 30, 0)

mentalHealth = st.sidebar.number_input('For how many days during the past 30 days was your mental health not good?', 0, 30, 0)

phyActivity = st.sidebar.selectbox('Physical activity in the past month' , ['No', 'Yes'])

diff = st.sidebar.selectbox('Do you have serious difficulty walking or climbing stairs?', ['No', 'Yes'])

diabetic = st.sidebar.selectbox('Have you ever had diabetes?', ['No', 'Yes'])

asthama = st.sidebar.selectbox('Do you have asthma?', ['No', 'Yes'])

kidney = st.sidebar.selectbox('Do you have kidney disease?', ['No', 'Yes'])

skinCancer = st.sidebar.selectbox('Do you have skin cancer?', ['No', 'Yes'])

data = {'BMI': bmi,
        'Smoking': smoker,
        'AlcoholDrinking': drinker,
        'Stroke' : stroke ,
        'PhysicalHealth': physicalHealth,
        'MentalHealth': mentalHealth,
        'DiffWalking': diff,
        'Sex': gender ,
        'AgeCategory': age,
        'Race': race ,
        'Diabetic': diabetic ,
        'PhysicalActivity': phyActivity,
        'GenHealth': genHealth ,
        'SleepTime': sleep,
        'Asthma':asthama,
        'KidneyDisease':kidney,
        'SkinCancer':skinCancer
    }

#data

features = pd.DataFrame(data, index=[0])

#features

def predict():
    # Load the model
    with open('final_model.sav', 'rb') as f:
        model = pickle.load(f)
    pred_proba = model.predict_proba(features)
    st.write('Prediction of heart disease is ',pred_proba[0][1]*100)

if st.button('Predict'):

    predict()









