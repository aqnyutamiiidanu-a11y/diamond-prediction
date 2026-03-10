import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model_diamond_rf.pkl", "rb"))

st.title("Prediksi Harga Diamond")

carat = st.number_input("Carat")
cut = st.number_input("Cut")
color = st.number_input("Color")
clarity = st.number_input("Clarity")
depth = st.number_input("Depth")
table = st.number_input("Table")
x = st.number_input("X")
y = st.number_input("Y")
z = st.number_input("Z")

if st.button("Prediksi Harga"):

    data = np.array([[carat,cut,color,clarity,depth,table,x,y,z]])

    pred = model.predict(data)

    st.success(f"Perkiraan Harga Diamond: {pred[0]}")
