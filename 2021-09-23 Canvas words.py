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



model_ocr = tf.keras.Sequential(layers=[
    # Convolution Part : Extraction Feature
    # Layer 1
    tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), padding='SAME', input_shape = (128, 32, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    # Layer 2
    tf.keras.layers.Conv2D(filters=64, kernel_size=(5,5), padding='SAME'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    # Layer 3
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='SAME'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.MaxPooling2D(pool_size=(1,2), strides=(1,2)),
    # Layer 4
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='SAME'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.MaxPooling2D(pool_size=(1,2), strides=(1,2)),
    # Layer 5
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='SAME'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.MaxPooling2D(pool_size=(1,2), strides=(1,2)),
    # Remove axis 2
    tf.keras.layers.Lambda(lambda x : tf.squeeze(x, axis=2)),
    # Bidirectionnal RNN
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True)),
    # Classification of characters
    tf.keras.layers.Dense(100)
])

model_ocr.load_weights('model_words_2.h5')

vocab = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def decode_codes(codes, charList):
    table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(np.arange(len(charList)), charList,  key_dtype=tf.int32), '', name='id2char')
    return table.lookup(codes)

def greedy_decoder(logits, charList):
    # ctc beam search decoder
    predicted_codes, _ = tf.nn.ctc_greedy_decoder(
        # shape of tensor [max_time x batch_size x num_classes] 
        tf.transpose(logits, (1, 0, 2)),
        [logits.shape[1]]*logits.shape[0]
    )
    # convert to int32
    codes = tf.cast(predicted_codes[0], tf.int32)
    # Decode the index of caracter
    text = decode_codes(codes, charList)
    # Convert a SparseTensor to string
    text = tf.sparse.to_dense(text).numpy().astype(str)
    return list(map(lambda x: ''.join(x), text))


# Specify canvas parameters in application
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
)
realtime_update = st.sidebar.checkbox("Update in realtime", True)

height = 32
width = 128

canvas_h = 64
canvas_w = 256

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



# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype(np.float32), (width, height), interpolation = cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY) / 255.
    st.image(img)
    img = img.reshape([1, width, height, 1])
    st.write(img.shape) 

    pred = greedy_decoder(model_ocr(img), vocab)
    st.write(f'Texte prédit :', pred[0])

if canvas_result.json_data is not None:
    pass
    #st.write(canvas_result.json_data["objects"])
    #st.dataframe(pd.json_normalize(canvas_result.json_data["objects"]))
    

