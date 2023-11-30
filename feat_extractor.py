# import pickle
import cv2
from keras.models import load_model
import numpy as np
import tensorflow as tf
from skimage import io
import pandas as pd 
import glob
import os

# load modified vgg-16 model 
cnn_model = load_model('./models/best_model.h5')

# defining layer to extract features
features_layer1 = tf.keras.models.Model(
    inputs=cnn_model.inputs,
    outputs=cnn_model.get_layer(name="layer1024").output,
)

# function to extract 1024 features from 2nd FC layer 
def get_features(img):
  img = cv2.resize(img, (150, 150))
  img = cv2.bilateralFilter(img,3,90,90)
  img = np.array(img)
  img = np.expand_dims(img, axis=0)
  feat = features_layer1(img)
  feat = np.array(feat)
  return feat