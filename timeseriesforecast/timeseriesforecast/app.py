import streamlit as st
import matplotlib.pyplot as plt
import pickle
from prophet import Prophet

df = []
df.append(pickle.load(open('furniture_df.pkl', 'rb')))
df.append(pickle.load(open('office_df.pkl', 'rb')))
df.append(pickle.load(open('tech_df.pkl', 'rb')))

st.title("Product demand Forecast using FB Prophet")
Category = {'Furniture': 0,
            'Office supplies': 1,
            'Technology': 2}

selected_category = st.selectbox(
    'Select your category', (Category)
)
fperiods = st.number_input('Insert the period for prediction', value = 36)
fperiods = int(fperiods)

if st.button('Forecast'):

    st.write("Forecating the ",selected_category," data for ",fperiods,"months in future." )

    model = Prophet(interval_width=0.95)
    model.fit(df[Category[selected_category]])
    future_df = model.make_future_dataframe(periods=fperiods, freq='MS')
    forecast = model.predict(future_df)
    fig = model.plot(forecast, xlabel='Date', ylabel='Sales')

    st.write(fig)
