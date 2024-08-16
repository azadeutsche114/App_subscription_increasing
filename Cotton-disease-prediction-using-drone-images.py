#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Tools / IDE

I used Jupyter NoteBook (Google Colab) for model training. used spyder for model deployment on the local system. To use Jupyter NoteBook and Spyder, just install anaconda.
Software Requirments

    Python == 3.7.7
    TensorFlow == 2.1.0
    Keras == 2.4.3
    NumPy == 1.18.5
    Flask == 1.1.2

Install above packages using below command in anaconda prompt


# In[ ]:


## Project: Cotton Plant Disease Prediction & Get Cure AI App - IAIP
 
#import libraries
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
 
keras.__version__
 
train_data_path = "/content/drive/My Drive/My ML Project /DL Project/CNN/cotton plant disease prediction/data/train"
validation_data_path = "/content/drive/My Drive/My ML Project /DL Project/CNN/cotton plant disease prediction/data/val"
 
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


# In[ ]:


pip install tensorflow==2.1.0
pip install Keras==2.4.3
pip install numpy==1.18.5
pip install flask==1.1.2
Training 🌿Cotton Plant Disease Prediction & Get Cure AI App


# In[ ]:


# this is the augmentation configuration we will use for training
# It generate more images using below parameters
training_datagen = ImageDataGenerator(rescale=1./255,
                                      rotation_range=40,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                      fill_mode='nearest')
 
# this is a generator that will read pictures found in
# at train_data_path, and indefinitely generate
# batches of augmented image data
training_data = training_datagen.flow_from_directory(train_data_path, # this is the target directory
                                      target_size=(150, 150), # all images will be resized to 150x150
                                      batch_size=32,
                                      class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels
 
training_data.class_indices
 
 


# In[ ]:



# this is the augmentation configuration we will use for validation:
# only rescaling
valid_datagen = ImageDataGenerator(rescale=1./255)

# this is a similar generator, for validation data
valid_data = valid_datagen.flow_from_directory(validation_data_path,
                                 target_size=(150,150),
                                 batch_size=32,
                                 class_mode='binary')

images = [training_data[0][0][0] for i in range(5)]
plotImages(images)

model_path = '/content/drive/My Drive/My ML Project /DL Project/CNN/cotton plant disease prediction/v3_red_cott_dis.h5'
checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#Building cnn model
cnn_model = keras.models.Sequential([
                                   keras.layers.Conv2D(filters=32, kernel_size=3, input_shape=[150, 150, 3]),
                                   keras.layers.MaxPooling2D(pool_size=(2,2)),
                                   keras.layers.Conv2D(filters=64, kernel_size=3),
                                   keras.layers.MaxPooling2D(pool_size=(2,2)),
                                   keras.layers.Conv2D(filters=128, kernel_size=3),
                                   keras.layers.MaxPooling2D(pool_size=(2,2)),                                    
                                   keras.layers.Conv2D(filters=256, kernel_size=3),
                                   keras.layers.MaxPooling2D(pool_size=(2,2)),

                                   keras.layers.Dropout(0.5),                                                                        
                                   keras.layers.Flatten(), # neural network beulding
                                   keras.layers.Dense(units=128, activation='relu'), # input layers
                                   keras.layers.Dropout(0.1),                                    
                                   keras.layers.Dense(units=256, activation='relu'),                                    
                                   keras.layers.Dropout(0.25),                                    
                                   keras.layers.Dense(units=4, activation='softmax') # output layer
])


# In[ ]:


# compile cnn model
cnn_model.compile(optimizer = Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
 
# train cnn model
history = cnn_model.fit(training_data, 
                          epochs=500, 
                          verbose=1, 
                          validation_data= valid_data,
                          callbacks=callbacks_list) # time start 16.06


# In[ ]:


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
 
history.history
Deploy 🌿Cotton Plant Disease Prediction & Get Cure AI App on Local System

Open Spyder and create a new project then create folders and files according to below hierarchy of the project.
app.py


# In[ ]:



#Import necessary libraries
from flask import Flask, render_template, request
 
import numpy as np
import os
 
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
 
#load model
model =load_model("model/v3_pred_cott_dis.h5")
 
print('@@ Model loaded')
 


# In[ ]:


def pred_cot_dieas(cott_plant):
  test_image = load_img(cott_plant, target_size = (150, 150)) # load image 
  print("@@ Got Image for prediction")
   
  test_image = img_to_array(test_image)/255 # convert image to np array and normalize
  test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
   
  result = model.predict(test_image).round(3) # predict diseased palnt or not
  print('@@ Raw result = ', result)
   
  pred = np.argmax(result) # get the index of max value
 
  if pred == 0:
    return "Healthy Cotton Plant", 'healthy_plant_leaf.html' # if index 0 burned leaf
  elif pred == 1:
      return 'Diseased Cotton Plant', 'disease_plant.html' # # if index 1
  elif pred == 2:
      return 'Healthy Cotton Plant', 'healthy_plant.html'  # if index 2  fresh leaf
  else:
    return "Healthy Cotton Plant", 'healthy_plant.html' # if index 3
 


# In[ ]:


#------------>>pred_cot_dieas<<--end
     
 
# Create flask instance
app = Flask(__name__)
 
# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
        return render_template('index.html')
     
  
# get input image from client then predict class and render respective .html page for solution
@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] # fet input
        filename = file.filename        
        print("@@ Input posted = ", filename)
         
        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)
 
        print("@@ Predicting class......")
        pred, output_page = pred_cot_dieas(cott_plant=file_path)
               
        return render_template(output_page, pred_output = pred, user_image = file_path)
     
# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False) 

