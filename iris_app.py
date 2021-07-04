# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Loading the dataset.
iris_df = pd.read_csv("iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating a model for Support Vector classification to classify the flower types into labels '0', '1', and '2'.

# Creating features and target DataFrames.
X = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df['Label']

# Splitting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Creating the SVC model and storing the accuracy score in a variable 'score'.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
score = svc_model.score(X_train, y_train)

@st.cache
def prediction(s_l,s_w,p_l,p_w):
	pred = svc_model.predict([[s_l,s_w,p_l,p_w]])
	pred = pred[0]
	if pred == 0:
		return 'Iris-setosa'
	elif pred == 1:
		return 'Iris-virginica'
	else :
		return 'Iris-versicolor'

st.title('Flowers')

s_l = st.slider('SepalLengthCm',1.0,99.0)
s_w = st.slider('SepalWidthCm',1.0,99.0)
p_l = st.slider('PetalLengthCm',1.0,99.0)
p_w = st.slider('PetalWidthCm',1.0,99.0)

if st.button('Predict'):
	var = prediction(s_l,s_w,p_l,p_w)
	st.write('Species -:',var)
	st.write('Score -:',score)