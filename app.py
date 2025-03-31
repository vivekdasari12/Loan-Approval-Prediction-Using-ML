import numpy as np
import pickle as pk
import streamlit as st
import pandas as pd

data = pd.read_csv('loan_approval_dataset.csv')

data.columns=data.columns.str.strip()

def clean_data(x):
    return x.strip()
data['education']=data['education'].apply(clean_data)
data['self_employed']=data.self_employed.apply(clean_data)


model=pk.load(open('model.pkl','rb'))
scaler=pk.load(open('scaler.pkl','rb'))

st.header('Loan Prediction App')

no_of_dependents=st.slider('No Of Dependents ',0,5)
education_type=st.selectbox('Eduaction Type',data.education.unique())
self_employed=st.selectbox('Self Employed',data.self_employed.unique())
income=st.slider('Per Annum Income',200000,9900000)
loan_amount=st.slider('Loan Amount',300000,39500000)
duration=st.slider('Loan Term',2,20)
score=st.slider('Cibil Score ',300,900)
assets=st.slider('Assests',400000,90700000)

if st.button('Predict'):
    test_data=pd.DataFrame([[no_of_dependents,education_type,self_employed,income,loan_amount,duration,score,assets]],columns=['no_of_dependents', 'education', 'self_employed', 'income_annum',
       'loan_amount', 'loan_term', 'cibil_score', 'assets'])
    test_data['education'].replace(['Graduate', 'Not Graduate'],[1,0],inplace=True)
    test_data['self_employed'].replace(['No', 'Yes'],[0,1],inplace = True)
    test_data=scaler.transform(test_data)
    prediction=model.predict(test_data)
    if prediction[0]== 0 :
        st.markdown('Loan is Rejected')
    else:
        st.markdown('Loan is Approved')    



