import csv
import cv2
import numpy as np

# import basic scikit learn method for processing data
from sklearn.model_selection import train_test_split
from random import shuffle
import sklearn
import math

# import all necessary keras modules necessary
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers import Lambda, Cropping2D
from math import ceil 

def generator(samples, batch_size=32):
    num_samples = len(samples)
    correction_factor = 0.30
    while 1: 
        shuffle(samples) #shuffling the total images
        for offset in range(0, num_samples, batch_size):
            
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                    for i in range(0,3): 
                        
                        #name = 'data/IMG/'+batch_sample[i].split('/')[-1]
                        path = 'data/IMG/'
                        #print (name)
                        #img = cv2.imread(name)
                        #print (img)
                        center_image = cv2.imread(path + batch_sample[i].split('/')[-1], cv2.COLOR_BGR2RGB) 
                        center_angle = float(batch_sample[3]) 
                        #center_angle = float(batch_sample[i]) 
                        images.append(center_image)
                       
                        if(i==0):
                            angles.append(center_angle)
                        if(i==1):
                            angles.append(center_angle + correction_factor)
                        if(i==2):
                            angles.append(center_angle - correction_factor)
                        
                        images.append(cv2.flip(center_image,1))
                        if(i==0):
                            angles.append(center_angle*-1)
                        if(i==1):
                            angles.append((center_angle*-1 + correction_factor))
                        if(i==2):
                            angles.append((center_angle*-1 - correction_factor))
                        
            X_train = np.array(images)
            y_train = np.array(angles)
            #y_tran = float(y_train)
                        
            yield sklearn.utils.shuffle(X_train, y_train)

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)
                                        
train_samples, validation_samples = train_test_split(samples,test_size=0.30)
                                          
train_generator = generator(train_samples, batch_size = 32)
validation_generator = generator(validation_samples, batch_size = 32)
                                          
                                          
# Using Nvidianet architecture
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((60, 25), (0, 0))))
model.add(Conv2D(24, (5, 5), activation='relu', strides=(2, 2)))
model.add(Conv2D(36, (5, 5), activation='relu', strides=(2, 2)))
model.add(Conv2D(48, (3, 3), activation='relu', strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', strides=(2, 2)))
model.add(Dropout(0.35))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

batch_size = 32
model.fit_generator(train_generator, 
            steps_per_epoch=ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=ceil(len(validation_samples)/batch_size), 
            epochs=12, verbose=1)


# save the model
model.save('model.h5')

print("Model saved")
model.summary()