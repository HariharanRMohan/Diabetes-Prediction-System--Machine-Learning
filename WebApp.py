#Our Aim to detect someone have diabetes or not

#Import the libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

#Create title and sub
st.write("""
# Diabetes Detection
Detect if someone has diabetes using ML @ Python
""")

#Open and Display cover image
image = Image.open('C:/Users/User/Desktop/ML Project/posterr.jpg')
st.image(image, caption='ML', use_column_width=True)

#Get the data
df = pd.read_csv('C:/Users/User/Desktop/ML Project/diabetes.csv')

#Set subheader
st.subheader('Data Information:')

#Show the data as a table
st.dataframe(df)

#Show statistic on the data
st.write(df.describe())

#Show the data as a chart
chart = st.bar_chart(df)


























    
    





























