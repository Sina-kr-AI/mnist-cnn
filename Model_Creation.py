from keras import models,layers,optimizers,losses,datasets
import json

#Model

conv_model=models.Sequential()
conv_model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
conv_model.add(layers.MaxPooling2D((2,2)))
conv_model.add(layers.Conv2D(64,(3,3),activation='relu'))
conv_model.add(layers.MaxPooling2D((2,2)))
conv_model.add(layers.Conv2D(64,(3,3),activation='relu'))
conv_model.add(layers.Flatten())
conv_model.add(layers.Dense(64,activation='relu'))
conv_model.add(layers.Dense(10,activation='softmax'))

#Saving Model

model_config=conv_model.get_config()
with open('conv_model','w') as f:
    json.dump(model_config,f,indent=4)