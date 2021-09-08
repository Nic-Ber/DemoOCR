# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 15:07:54 2021

@author: Admin
"""

#     "F:\MES DOCUMENTS\DATA SCIENTEST\PROJET\Notebooks\2021-08-27 Test Canvas Streamlit.py"



#import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import cv2
import numpy as np
from pickle5 import load


# Specify canvas parameters in application
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
)
realtime_update = st.sidebar.checkbox("Update in realtime", True)
lettre_ou_mot = st.sidebar.radio('Reconnaissance de lettre ou de mot',
                                 ('Lettre', 'Mot'))

if lettre_ou_mot == 'Lettre':
    height = 28
    width = 28
    model_path = "./2021-08-29 model_conv/model_conv"
    encoder_path = "Label_enco.pkl"
elif lettre_ou_mot == 'Mot':
    height = 32
    width = 132
    model_path = None

canvas_h = 32
canvas_w = 132

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=canvas_h,
    width=canvas_w,
    drawing_mode=drawing_mode,
    key="canvas",
)

#resize ?

if model_path is not None:
    model = tf.keras.models.load_model(model_path)
if encoder_path is not None:
    encoder = load(open(encoder_path, 'rb'))

# Do something interesting with the image data and paths
if canvas_result.image_data is not None:

    if False:
        #on garde le plus petit ratio pour le redimensionnement
        r = min(height / canvas_h, width / canvas_w)
        dim = (int(canvas_w * r), int(canvas_h * r))
        #on redimensionne l'image pour faire les dimensions souhaitées
        img = cv2.resize(canvas_result.image_data.astype(np.float32), dim, interpolation = cv2.INTER_AREA)
        #on complète l'image par du blanc pour qu'elles aient toutes la même taille
        dx = width - dim[0]
        dy = height - dim[1]
        img = cv2.copyMakeBorder(img,0,dy,0,dx,cv2.BORDER_CONSTANT,value=[255,255,255])
    else:
        img = cv2.resize(canvas_result.image_data.astype(np.float32), (height, width), interpolation = cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY) / 255.
    st.image(img)

    img = img.reshape([1, height, width, 1])
    prob = model.predict(img)
    pred = prob.argmax(axis=-1)
    label_pred = encoder.inverse_transform([pred])
    st.write(f'label prédit : {label_pred[0]}')

if canvas_result.json_data is not None:
    pass
    #st.write(canvas_result.json_data["objects"])
    #st.dataframe(pd.json_normalize(canvas_result.json_data["objects"]))
    

