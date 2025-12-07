import os
import numpy as np
import streamlit as st
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from bs4 import BeautifulSoup
import requests

# Loading model
model = tf.keras.models.load_model('FV.keras')

labels = {0:'Apple',1: 'Banana', 2: 'Beetroot', 3: 'Bellpepper', 4: 'Cabbage', 
          5: 'Capsicum', 6: 'Carrot', 7: 'Cauliflower', 8: 'Chilli pepper', 
          9: 'Corn', 10: 'Cucumber', 11: 'Eggplant', 12: 'Garlic', 13: 'Ginger', 
          14: 'Grapes', 15: 'Jalapeno', 16: 'Kiwi', 17: 'Lemon', 18: 'Lettuce', 
          19: 'Mango', 20: 'Onion', 21: 'Orange', 22: 'Paprika', 23: 'Pear', 
          24: 'Peas', 25: 'Pineapple', 26: 'Pomegranate', 27: 'Potato', 
          28: 'Raddish', 29: 'Soya beans', 30: 'Spinach', 31: 'Sweetcorn', 
          32: 'Sweetpotato', 33: 'Tomato', 34: 'Turnip', 35: 'Watermelon'}

fruits = ['Apple','Banana','Bellpepper','Chilli pepper','Grapes','Jalapeno','Kiwi','Orange','Mango','Lemon','Paprika','Pear','Pineapple','Pomegranate','Corn','Tomato','Watermelon']
vegetables = ['Beetroot','Cabbage','Capsicum','Carrot','Cauliflower','Sweetcorn','Cucumber','Eggplant','Garlic','Ginger','Lettuce','Onion','Peas','Sweetpotato','Raddish','Soya beans','Spinach','Potato','Turnip']

calories = {'apple': '52','banana': '89','beetroot': '43','bellpepper': '20','cabbage': '25','capsicum': '20','carrot': '41','cauliflower': '25','chilli pepper': '40','corn': '86','cucumber': '15','eggplant': '25','garlic': '149','ginger': '80','grapes': '69','jalapeno': '40','kiwi': '61','lemon': '29','lettuce': '15','mango': '60','onion': '40','orange': '47','paprika': '20','pear': '57','peas': '81','pineapple': '50','pomegranate': '83','potato': '77','raddish': '16','soya beans': '446','spinach': '23','sweetcorn': '86','sweetpotato': '86','tomato': '18','turnip': '28','watermelon': '30'}

def fetch_calories(prediction):
    cal = calories.get(prediction.lower().strip(), None)
    if cal:
        return f"{cal} kcal"
    
    try:
        url = 'https://www.google.com/search?q=calories+in+' + prediction.lower().replace(' ', '+')
        req = requests.get(url, timeout=5).text
        scrap = BeautifulSoup(req, 'html.parser')
        web_cal = scrap.find("div", class_="BNeawe iBp4i AP7Wnd")
        if web_cal:
            return web_cal.text + " kcal"
        return "No data"
    except:
        return "No data"

def processed_img(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0  
    img = np.expand_dims(img, axis=0)  
    answer = model.predict(img, verbose=0)
    y_class = np.argmax(answer, axis=1)  
    y = int(y_class[0])
    res = labels.get(y, f"Class {y}")  
    return res

def run():
    st.title("🍎 Fruits-Vegetable Classification")
    img_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
    
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=True)
        
        upload_dir = './upload/upload_images/'
        os.makedirs(upload_dir, exist_ok=True)
        save_image_path = os.path.join(upload_dir, img_file.name)  # FIXED: save_image_path
        
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())
        
        
        st.write("**Processing...**")
        result = processed_img(save_image_path)
        
        # Category check (exact match)
        if result in vegetables:
            st.info('🥬 **Category: Vegetables**')
        elif result in fruits:
            st.info('🍊 **Category: Fruits**')
        else:
            st.warning('❓ **Category: Unknown**')
        
        st.success(f"**Predicted: {result}**")
        
        cal = fetch_calories(result)
        st.warning(f"**Calories: {cal} (100g)**")

run()
