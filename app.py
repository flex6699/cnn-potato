import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image

#load the trained model
model=load_model("cnn_model.h5")

#define class labels
class_labels=["Early blight","Late blight","Healthy"];

def preprocess_image(img):
    img=img.resize((64,64))
    img_array=image.img_to_array(img)
    img_array=np.expand_dims(img_array,axis=0)
    img_array/=255.0
    return img_array

st.title("Potato Diseases Classification with CNN")
st.write("Upload an image to classify it")
uploaded_file=st.file_uploader("Choose an image....",type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img=Image.open(uploaded_file)
    st.image(img,caption="Uploaded Image",use_column_width=True)
    img_array=preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_index]
    print(predictions)
    st.write(f'Predicted class: {predicted_class_label}')


