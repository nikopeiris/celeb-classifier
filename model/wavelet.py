import numpy as np
import pywt
import cv2
import json
import joblib
import os

def w2d(img, mode='haar', level=1):
  imArray = img
  imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
  imArray =  np.float32(imArray)   
  imArray /= 255
  
  coeffs=pywt.wavedec2(imArray, mode, level=level)

  coeffs_H=list(coeffs)  
  coeffs_H[0] *= 0

  imArray_H = pywt.waverec2(coeffs_H, mode)
  imArray_H *= 255
  imArray_H = np.uint8(imArray_H)

  return imArray_H

x, y, celebrity_names = [], [], []

with open("./celebrity_file_names.json", "r") as f:
    celebrity_file_names_dict = json.load(f)

for index, (celebrity_name, training_files) in enumerate(celebrity_file_names_dict.items()):
  celebrity_names.append(celebrity_name)
  for training_image in training_files:
    path = "./images_cropped/" + celebrity_name + "/" + training_image
    if not os.path.exists(path):
      print(f"Error processing image {training_image}: File does not exist.")
      continue
    try:
      print(f"Processing image: {path}")
      img = cv2.imread(path)
      scalled_raw_img = cv2.resize(img, (32, 32))
      img_har = w2d(img,'db1',5)
      scalled_img_har = cv2.resize(img_har, (32, 32))
      combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
      x.append(combined_img)
      y.append(index)
    except Exception as e:
      print(f"Error processing image {training_image}: {e}")
      
x = np.array(x).reshape(len(x),4096).astype(float)
joblib.dump((x, y, celebrity_names),'data_bundle.joblib')