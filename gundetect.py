import os
import cv2
import numpy as np
import keras
from keras.utils import to_categorical
import sklearn.model_selection
"""
NUM_OF_IMGS = 60

data=[]
label = []
categories = ['positive', 'negative']
path = 'G:\Desktop\Gun Detection'
for a in categories:
    fullpath = os.path.join(path, a)
    newfullpath = os.listdir(fullpath)


    
    for v in newfullpath:
        fp = os.path.join(fullpath, v)
        nfp = cv2.imread(fp)
        nfp = cv2.cvtColor(nfp, cv2.COLOR_BGR2GRAY)
        nfp = cv2.resize(nfp, (25,25))
        data.append(nfp)
        if a=='positive':
            label.append(1)
        if a=='negative':
            label.append(0)

data= np.array(data)
data=data/255
data=np.reshape(data, (NUM_OF_IMGS ,25,25,1))
print(data[0].shape)
label = np.array(label)
label = to_categorical(label)
print(label)
trainimgs, testimgs, trainlabels, testlabels = sklearn.model_selection.train_test_split(data, label, test_size = 0.05)
print(len(trainimgs))
print(len(testimgs))

model = keras.Sequential([
    keras.layers.Conv2D(128, kernel_size=(3,3), input_shape=(25,25,1), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, kernel_size=(3,3), input_shape=(25,25,1), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(32, kernel_size=(3,3), input_shape=(25,25,1), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
    ])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(trainimgs,trainlabels,epochs=20)

model.save('gundetect.h5')
"""
from keras.models import load_model
model = load_model('gundetect.h5')
img = "blueshirt.jpg"
test_data = []

image = cv2.imread(img)
cv2.rectangle(image,(0,0),(60,60), (255,0,0), 2)
image = image[0:60, 0:60]
##cv2.imshow('Screen', image)
##cv2.waitKey()
##cv2.destroyAllWindows()
image[np.where((image==[255,255,255]).all(axis=2))] = [0,0,0]
##cv2.imshow('Screen1', image)
##cv2.waitKey()
##cv2.destroyAllWindows()



gsimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
resimage = cv2.resize(gsimage, (25,25))

test_data.append(resimage)
test_data=np.array(test_data)
test_data=test_data/255
test_data = np.reshape(test_data, (1,25,25,1))

pred = model.predict(test_data)
pred = pred[0]
prediction = np.argmax(pred)
print(pred)


if prediction == 1:
    print('Gun has been detected')
else:
    print('No gun has been detected')


    
