import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://nishitdb:Cnd_Cdr_423#152@cluster0.7opr5pi.mongodb.net/?appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['SVM']
collection = db['Type_Of_Flower_SVM_Binary']

def load_model():
    with open("svm_binary.pkl" , "rb") as file:
        svm_binary, scaler = joblib.load(file)
    return svm_binary, scaler

def preprocessing_input_data(data, scaler):
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

def predict_data(data):
    svm_binary, scaler = load_model()
    preprocessed_data = preprocessing_input_data(data, scaler)
    prediction = svm_binary.predict(preprocessed_data)
    return prediction

def main():
    st.title("Predict Type Of Flower")
    st.write("Enter Data to Know Type Of Flower")

    sepal_length = st.number_input("Sepal Length", min_value=4.300000, max_value=7.900000, value=5.843333, step=0.1)
    sepal_width = st.number_input("Sepal Width", min_value=2.000000, max_value=4.400000, value=3.057333, step=0.1)
    petal_length = st.number_input("Petal Length", min_value=1.000000, max_value=6.900000, value=3.758000, step=0.1)
    petal_width = st.number_input("Petal Width", min_value=0.100000, max_value=2.500000, value=1.199333, step=0.1)

    if st.button("Know Type Of Flower"):
        user_data = {
            "sepal length (cm)": sepal_length,
            "sepal width (cm)" : sepal_width,
            "petal length (cm)" : petal_length,
            "petal width (cm)" : petal_width
        }

        prediction = predict_data(user_data)

        prediction_type = prediction[0]

        type_of_flower = {
            0 : "setosa",
            1 : "versicolor"
        }
        st.success(f"Type Of Flower Is :- {type_of_flower[prediction_type]}")
        
        user_data['prediction'] = type_of_flower[prediction_type]
        collection.insert_one(user_data)
        
if __name__ == "__main__":
    main()
