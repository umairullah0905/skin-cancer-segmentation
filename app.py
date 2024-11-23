import streamlit as st
import cv2
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from io import StringIO
@st.cache_resource
def read_image(_x):
    x = cv2.imread(path, cv2.IMREAD_COLOR)  ## (H, W, 3)
    x = cv2.resize(x, (256, 256))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    # print(x)
    return  x 




def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)





with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef}):
        model = tf.keras.models.load_model("/home/umair123/mlApp/model.h5")





def make_prediction(img):
    img_processed = read_image(img)
    prediction = model.predict(img_processed)
    return prediction



st.title("skin lesion detection using semantic segmentation")
upload = st.file_uploader(label="upload image",type=["jpeg","png","jpg"])


if upload:

    
    orimage=Image.open(upload)
    img = Image.open(upload)
    img = img.resize((256,256))
    img = np.array(img)
    


    
    img = img.astype(np.float32)
    img = img/255.0
    # print(img.dtype)
    
    img = np.expand_dims(img, axis=0)
    print(img.shape)

    prediction = model.predict(img)[0]
    print(prediction.shape)

    fig = plt.figure(figsize=(12,12))
    ax=fig.add_subplot(111)
    plt.imshow(orimage)
    plt.imshow(prediction)
    plt.xticks([],[])
    plt.yticks([],[])
    st.pyplot(fig, use_container_width=True)