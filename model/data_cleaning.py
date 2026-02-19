import cv2
import os
import shutil

face_cascade = cv2.cuda.CascadeClassifier_create(r"C:\Users\nikop\opencv-4.12.0\opencv-4.12.0\data\haarcascades_cuda\haarcascade_frontalface_default.xml")
eye_cascade = cv2.cuda.CascadeClassifier_create(r"C:\Users\nikop\opencv-4.12.0\opencv-4.12.0\data\haarcascades_cuda\haarcascade_eye.xml")

path_to_data = "./image_dataset"
path_to_cr_data = "./images_cropped/"

img_dirs = []

for entry in os.scandir(path_to_data):
    if entry.is_dir():
        img_dirs.append(entry.path)
        
if os.path.exists(path_to_cr_data):
     shutil.rmtree(path_to_cr_data)
os.mkdir(path_to_cr_data)

cropped_image_dirs = []
celebrity_file_names_dict = {}

def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    
    gpu_img = cv2.cuda_GpuMat()
    gpu_img.upload(img)
    gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
    faces_gpu = face_cascade.detectMultiScale(gpu_gray)
    faces = faces_gpu.download() 
    
    if faces is None: return None

    for (x, y, w, h) in faces[0]:
      roi_gray = cv2.cuda_GpuMat(gpu_gray, (x, y, w, h))
      roi_gray_aligned = cv2.cuda_GpuMat(roi_gray.size(), roi_gray.type())
      roi_gray.copyTo(roi_gray_aligned)
      
      try:
        eyes_gpu = eye_cascade.detectMultiScale(roi_gray_aligned)
        eyes = eyes_gpu.download()
      except cv2.error as e:
        print(f"Skipping a face due to GPU alignment issue: {e}")
        continue
      
      if eyes is not None and len(eyes[0]) >= 2:
          return img[y:y+h, x:x+w]
        
    return None

for img_dir in img_dirs:
    count = 1
    celebrity_name = img_dir.split("\\")[-1]
    print(celebrity_name)
    
    celebrity_file_names_dict[celebrity_name] = []
    cropped_folder = path_to_cr_data + celebrity_name
    if not os.path.exists(cropped_folder):
        os.makedirs(cropped_folder)
        cropped_image_dirs.append(cropped_folder)
        print("Generating cropped images in folder: ",cropped_folder)
    
    for entry in os.scandir(img_dir):
        roi_color = get_cropped_image_if_2_eyes(entry.path)
        if roi_color is not None:
            cropped_file_name = celebrity_name + str(count) + ".png"
            cropped_file_path = cropped_folder + "/" + cropped_file_name 
            cv2.imwrite(cropped_file_path, roi_color)
            celebrity_file_names_dict[celebrity_name].append(cropped_file_name)
            count += 1    