import os
import numpy as np
import streamlit as st
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from bs4 import BeautifulSoup
import requests
import cv2
import json
from tensorflow.keras.applications.resnet50 import preprocess_input

@st.cache_resource
def load_main_classifier():
    return tf.keras.models.load_model("FV.keras")


@st.cache_resource
def load_freshness_model():
    return tf.keras.models.load_model("fruit_image_training.keras")


fv_model = load_main_classifier()
freshness_model = load_freshness_model()

#Class labels for Fruits/Vegetables Model

labels = {0: 'Apple', 1: 'Banana', 2: 'Beetroot', 3: 'Bellpepper', 4: 'Cabbage',
          5: 'Capsicum', 6: 'Carrot', 7: 'Cauliflower', 8: 'Chilli pepper',
          9: 'Corn', 10: 'Cucumber', 11: 'Eggplant', 12: 'Garlic', 13: 'Ginger',
          14: 'Grapes', 15: 'Jalapeno', 16: 'Kiwi', 17: 'Lemon', 18: 'Lettuce',
          19: 'Mango', 20: 'Onion', 21: 'Orange', 22: 'Paprika', 23: 'Pear',
          24: 'Peas', 25: 'Pineapple', 26: 'Pomegranate', 27: 'Potato',
          28: 'Raddish', 29: 'Soya beans', 30: 'Spinach', 31: 'Sweetcorn',
          32: 'Sweetpotato', 33: 'Tomato', 34: 'Turnip', 35: 'Watermelon'}

fruits = [
    'Apple', 'Banana', 'Bellpepper', 'Chilli pepper', 'Grapes', 'Jalapeno',
    'Kiwi', 'Orange', 'Mango', 'Lemon', 'Paprika', 'Pear', 'Pineapple',
    'Pomegranate', 'Corn', 'Tomato', 'Watermelon'
]

vegetables = [
    'Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Sweetcorn',
    'Cucumber', 'Eggplant', 'Garlic', 'Ginger', 'Lettuce', 'Onion', 'Peas',
    'Sweetpotato', 'Raddish', 'Soya beans', 'Spinach', 'Potato', 'Turnip'
]

#Class labels for Ripeness Model

with open("class_indices.json") as f:
    class_indices = json.load(f)

class_names = {v: k for k, v in class_indices.items()}

pretty_names = {
    "freshapples": "Fresh Apple",
    "freshbanana": "Fresh Banana",
    "freshoranges": "Fresh Orange",
    "rottenapples": "Rotten Apple",
    "rottenbanana": "Rotten Banana",
    "rottenoranges": "Rotten Orange",
    "unripe apple": "Unripe Apple",
    "unripe banana": "Unripe Banana",
    "unripe orange": "Unripe Orange"
}

calories = {'apple': '52', 'banana': '89', 'beetroot': '43', 'bellpepper': '20',
            'cabbage': '25', 'capsicum': '20', 'carrot': '41', 'cauliflower': '25',
            'chilli pepper': '40', 'corn': '86', 'cucumber': '15', 'eggplant': '25',
            'garlic': '149', 'ginger': '80', 'grapes': '69', 'jalapeno': '40',
            'kiwi': '61', 'lemon': '29', 'lettuce': '15', 'mango': '60',
            'onion': '40', 'orange': '47', 'paprika': '20', 'pear': '57',
            'peas': '81', 'pineapple': '50', 'pomegranate': '83', 'potato': '77',
            'raddish': '16', 'soya beans': '446', 'spinach': '23', 'sweetcorn': '86',
            'sweetpotato': '86', 'tomato': '18', 'turnip': '28', 'watermelon': '30'}


def fetch_calories(prediction):
    pred_key = prediction.lower().strip()
    cal = calories.get(pred_key)

    if cal:
        return f"{cal} kcal"

    # Fallback to scraping Google
    try:
        url = f'https://www.google.com/search?q=calories+in+{pred_key.replace(" ", "+")}'
        req = requests.get(url, timeout=5).text
        scrap = BeautifulSoup(req, 'html.parser')
        web_cal = scrap.find("div", class_="BNeawe iBp4i AP7Wnd")
        return web_cal.text if web_cal else "No data"
    except:
        return "No data"

#Prediction Functions

def predict_fv(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = fv_model.predict(img, verbose=0)
    class_id = np.argmax(preds)
    return labels[class_id]


def predict_freshness(pil_image):
    img = np.array(pil_image)
    img = cv2.resize(img, (224, 224)).astype(np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    preds = freshness_model.predict(img, verbose=0)
    class_id = np.argmax(preds)
    readable = pretty_names[class_names[class_id]]
    conf = float(preds[0][class_id])
    return readable, conf

st.title("🥕 Unified Fruit & Vegetable Classifier + Freshness Checker")

uploaded_file = st.file_uploader(
    "Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    # save file temporarily
    save_path = "temp_image.jpg"
    img.save(save_path)

    tab1, tab2 = st.tabs(
        ["🔍 Identify Fruit/Vegetable", "🧪 Freshness Detection"])

    with tab1:
        st.subheader("Fruit/Vegetable Classifier")
        result = predict_fv(save_path)

        # Category
        if result in fruits:
            st.info("🍎 **Category: Fruit**")
        elif result in vegetables:
            st.info("🥬 **Category: Vegetable**")
        else:
            st.warning("❓ Category: Unknown")

        st.success(f"### Predicted: **{result}**")

        cal = fetch_calories(result)
        st.warning(f"### Calories (per 100g): **{cal}**")

    with tab2:
        st.subheader("Freshness Status")
        freshness, conf = predict_freshness(img)

        st.success(f"### Freshness: **{freshness}**")
        st.info(f"### Confidence: **{round(conf * 100, 2)}%**")
