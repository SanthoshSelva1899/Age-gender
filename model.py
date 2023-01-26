import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras import optimizers
import cv2


base_Dir='/Users/santh/Desktop/project4/UTKFace/'

age_list=[]
gender_list=[]

image_path_list=[]

for files in os.listdir(base_Dir):
    image_path=base_Dir+files
    splitting=files.split("_")
    age= int(splitting[0])
    gender=int(splitting[1])
    age_list.append(age)
    gender_list.append(gender)
    image_path_list.append(image_path)
    
    
df=pd.DataFrame()
df['age'],df['gender'],df['image_path']=age_list,gender_list,image_path_list

print(df.head())

length=df.shape[0]
print(length)
train_array=[]
image_path=df.loc[1]['image_path']
print(image_path)

for i in range (0,length) :
    image_path=df.iloc[i]['image_path']
    img=Image.open(image_path)
    img=img.resize((64,64))
    x=np.array(img)
    train_array.append(x)

train_array=np.array(train_array)
train_array=train_array.reshape(len(train_array),64,64,3)
print(train_array.shape)
train_array = train_array.astype('uint8')


gender_array=np.array(df['gender'])
age_array=np.array(df['age'])
x_train_age, x_test_age, y_train_age, y_test_age = train_test_split(train_array, age_array, test_size=0.4)
x_train_gender, x_test_gender, y_train_gender, y_test_gender = train_test_split(train_array, gender_array, test_size=0.4)

x_train_age=x_train_age/255.0
x_test_age=x_test_age/255.0
x_train_gender=x_train_gender/255.0
x_test_gender=x_test_gender/255.0
agemodel = Sequential()
agemodel.add(Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)))
agemodel.add(MaxPooling2D((2,2)))
agemodel.add(Conv2D(64, (3,3), activation='relu'))
agemodel.add(MaxPooling2D((2,2)))
agemodel.add(Conv2D(128, (3,3), activation='relu'))
agemodel.add(MaxPooling2D((2,2)))
agemodel.add(Flatten())
agemodel.add(Dense(64, activation='relu'))
agemodel.add(Dropout(0.5))
agemodel.add(Dense(1, activation='relu'))

agemodel.compile(loss='mean_squared_error',
             optimizer=optimizers.Adam(lr=0.0001))
genmodel = Sequential()
genmodel.add(Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)))
genmodel.add(MaxPooling2D((2,2)))
genmodel.add(Conv2D(64, (3,3), activation='relu'))
genmodel.add(MaxPooling2D((2,2)))
genmodel.add(Conv2D(128, (3,3), activation='relu'))
genmodel.add(MaxPooling2D((2,2)))
genmodel.add(Flatten())
genmodel.add(Dense(64, activation='relu'))
genmodel.add(Dropout(0.5))
genmodel.add(Dense(1, activation='sigmoid'))

genmodel.compile(loss='binary_crossentropy',
             optimizer=optimizers.Adam(lr=0.0001),
             metrics=['accuracy'])

x=genmodel.fit(x_train_gender,y_train_gender,epochs=50)

genmodel.save('genmodel.h5')
y=agemodel.fit(x_train_age,y_train_age,epochs=50)
agemodel.save('agemodel.h5')

input_image_path='C:/Users/santh/Desktop/project4/UTKFace/58_1_0_20170117133115674.jpg.chip.jpg'

input_image = cv2.imread(input_image_path)

input_image_resize = cv2.resize(input_image, (64,64))

input_image_scaled = input_image_resize/255

image_reshaped = np.reshape(input_image_scaled, [1,64,64,3])
input_prediction_gender = genmodel.predict(image_reshaped)
input_prediction_age=agemodel.predict(image_reshaped)
print(input_prediction_age,input_prediction_gender)



    
