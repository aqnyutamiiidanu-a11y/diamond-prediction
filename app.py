import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.title("💎 Diamond Price Prediction")

st.write("Aplikasi ini memprediksi harga diamond menggunakan model Random Forest.")

# Load dataset
url = "https://raw.githubusercontent.com/tidyverse/ggplot2/main/data-raw/diamonds.csv"
df = pd.read_csv(url)

# Encoding
le = LabelEncoder()
df['cut'] = le.fit_transform(df['cut'])
df['color'] = le.fit_transform(df['color'])
df['clarity'] = le.fit_transform(df['clarity'])

# Feature dan target
X = df.drop('price', axis=1)
y = df['price']

# Training model
model = RandomForestRegressor()
model.fit(X, y)

st.subheader("Masukkan Data Diamond")

carat = st.number_input("Carat")
cut = st.number_input("Cut")
color = st.number_input("Color")
clarity = st.number_input("Clarity")
depth = st.number_input("Depth")
table = st.number_input("Table")
x = st.number_input("X")
y_input = st.number_input("Y")
z = st.number_input("Z")

if st.button("Predict Price"):

    data = np.array([[carat,cut,color,clarity,depth,table,x,y_input,z]])

    prediction = model.predict(data)

    st.success(f"Estimated Diamond Price: ${prediction[0]:,.2f}")
