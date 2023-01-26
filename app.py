# -*- coding: utf-8 -*-
import cv2
import tensorflow as tf
import numpy as np

Gen_model_path='C:/Users/santh/Desktop/project4/genmodel.h5'
age_model_path='C:/Users/santh/Desktop/project4/agemodel.h5'


genmodel=tf.keras.models.load_model(Gen_model_path)
agemodel=tf.keras.models.load_model(age_model_path)

face_cascade = cv2.CascadeClassifier('/Users/santh/Downloads/project1/haarcascades/haarcascade_frontalface_default.xml')

cap= cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
    input_image_resize=cv2.resize(roi_color, (64,64))
    input_image_scaled = input_image_resize/255

    image_reshaped = np.reshape(input_image_scaled, [1,64,64,3])
    pred1=genmodel.predict(image_reshaped)
    pred2=agemodel.predict(image_reshaped)
    cv2.imshow('OpenCV Feed', frame)
    
    print(pred1,pred2)
    if cv2.waitKey(10) &0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()

