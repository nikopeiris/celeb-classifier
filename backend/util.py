import joblib
import numpy as np
import base64
import cv2
from wavelet import w2d

__celebrity_list = None
__model = None

def classify_image(image_base64_data, file_path=None):

    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)

    result = []
    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

        len_image_array = 32*32*3 + 32*32

        final = combined_img.reshape(1,len_image_array).astype(float)
        predicted_class = __model.predict(final)[0]
        probability = np.around(__model.predict_proba(final)*100,2).tolist()[0]
        result.append({
            'class': __celebrity_list[predicted_class],
            'probability': probability[predicted_class],
            'class_probability': probability,
            'celebrity_list': __celebrity_list
        })

    return result

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __celebrity_list
    global __model
    if __model is None:
      __model = joblib.load("./data/stacked_classifier.joblib")
    if __celebrity_list is None:
      __celebrity_list = joblib.load("./data/celebrity_names.joblib")
    print("loading saved artifacts...done")


def get_cv2_image_from_base64_string(b64str):
    if "," in b64str:
      b64str = b64str.split(",")[1]
    
    encoded_data = b64str
    encoded_data = encoded_data.strip().replace(" ", "+")
    remaining_padding = len(encoded_data) % 4
    if remaining_padding > 0:
        encoded_data += "=" * (4 - remaining_padding)
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier('./data/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./data/haarcascades/haarcascade_eye.xml')

    try:
      
      if image_path:
          img = cv2.imread(image_path)
      else:
          img = get_cv2_image_from_base64_string(image_base64_data)

      if img is None:
          return []

      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray, 1.3, 5)

      cropped_faces = []
      for (x,y,w,h) in faces:
              roi_gray = gray[y:y+h, x:x+w]
              roi_color = img[y:y+h, x:x+w]
              eyes = eye_cascade.detectMultiScale(roi_gray)
              if len(eyes) >= 2:
                  cropped_faces.append(roi_color)
      return cropped_faces
    
    except Exception as e:
      print(f"Error during image processing: {e}")
      return []

def get_b64_test_image_for_virat():
    with open("./data/brad.txt", "r") as f:
        return f.read()