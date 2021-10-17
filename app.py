# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 15:07:54 2021

@author: Nicolas BERNARDIN
"""


from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
#import tensorflow.keras.backend as K
import cv2
import numpy as np

vocab = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
numHidden = 256

def create_model(var):
    model = tf.keras.Sequential()
    # Convolution Part : Extraction Feature
    # Layer 1
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), padding='SAME', input_shape = (32, 128, 1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    # Layer 2
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(5,5), padding='SAME'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    # Layer 3
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='SAME'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,1), strides=(2,1)))
    # Layer 4
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding='SAME'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,1), strides=(2,1)))
    # Layer 5
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding='SAME'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,1), strides=(2,1)))
    # Remove axis 2
    model.add(tf.keras.layers.Lambda(lambda x : tf.squeeze(x, axis=1)))
    if var == 'GRU':
        # Bidirectionnal RNN
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(numHidden, return_sequences=True)))
    elif var == 'LSTM':
        # Bidirectionnal RNN
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(numHidden, return_sequences=True)))
    elif var == 'Conv':
        # Consolution 1D
        model.add(tf.keras.layers.Conv1D(filters=numHidden, kernel_size=3))
    else:
        raise ValueError()
    # Classification of characters
    model.add(tf.keras.layers.Dense(len(vocab)+1))
    return model

model_gru = create_model('GRU')
model_lstm = create_model('LSTM')
model_conv1d = create_model('Conv')

model_gru.load_weights('model_bi_gru_5e.h5')
model_lstm.load_weights('model_bi_lstm_5e.h5')
model_conv1d.load_weights('model_conv1d_5e.h5')


# définition des fonctions pour décoder
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
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 8)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
)
realtime_update = st.sidebar.checkbox("Update in realtime", True)

height = 32
width = 128

canvas_h = height * 5
canvas_w = width * 5

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
    img1 = img.reshape([1, height, width, 1])

    #st.image(img)
    pred_gru = greedy_decoder(model_gru(img1), vocab)
    pred_lstm = greedy_decoder(model_lstm(img1), vocab)
    pred_conv = greedy_decoder(model_conv1d(img1), vocab)
    
    st.write(f'Texte prédit (modèle GRU) :', pred_gru[0])
    st.write(f'Texte prédit (modèle LSTM) :', pred_lstm[0])
    st.write(f'Texte prédit (modèle Conv1d) :', pred_conv[0])

    # ici, tentative de mise en place de la technique Grad-CAM, mais j'ai encore un problème avec le calcul de class_out, donc ce n'est pas opérationnel...
    if False:
        with tf.GradientTape() as tape:
            last_conv_layer = model_3.get_layer('conv2d_4')
            iterate = tf.keras.models.Model([model_3.inputs], [model_3.output, last_conv_layer.output])
            model_out, last_conv_layer = iterate(img1)
            #st.write(model_out.shape)      #(1,32,100)
            #text = ''
            #for i in np.argmax(model_out, axis=2)[0,:]:
            #    text += vocab[i]
            class_out = model_out[:,np.argmax(model_out, axis=2)[0,:]]
            grads = tape.gradient(class_out, last_conv_layer)
            pooled_grads = K.mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        heatmap = heatmap.reshape((8, 8))
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
        img = heatmap * 0.5 + img
        st.image(cv2.resize(img, (width, height)))



#if canvas_result.json_data is not None:
    #pass
    #st.write(canvas_result.json_data["objects"])
    #st.dataframe(pd.json_normalize(canvas_result.json_data["objects"]))
    

