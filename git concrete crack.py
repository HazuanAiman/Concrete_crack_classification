# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 10:32:30 2022

@author: User
"""
#https://data.mendeley.com/datasets/5y9wdsg2zt/2
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import pathlib

file_path = r"C:\Users\User\.keras\datasets\Concrete Crack Images for Classification"
data_dir = pathlib.Path(file_path)

#%%
SEED = 12345
BATCH_SIZE = 8
IMG_SIZE = (224, 224)

train_dataset = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.3, subset='training', seed=SEED, batch_size=BATCH_SIZE, image_size=IMG_SIZE)
val_dataset = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.3, subset='validation', seed=SEED, batch_size=BATCH_SIZE, image_size=IMG_SIZE)

#%%
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take(val_batches // 5)
validation_dataset = val_dataset.skip(val_batches // 5)

#%%
AUTOTUNE = tf.data.AUTOTUNE

train_dataset_pf = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset_pf = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset_pf = test_dataset.prefetch(buffer_size=AUTOTUNE)

#%%
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])

#%%
#show example of applied image augmentation
for image, labels in train_dataset.take(1):
  plt.figure(figsize=(10, 10))
  first_image = image[0]
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
    plt.imshow(augmented_image[0] / 255)
    plt.axis('off')

#%%
#Use the method provided in the pretrained model object to rescale input
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

#Create the base model by calling out MobileNetV2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

#Freeze the model and show model summary
base_model.trainable = False
base_model.summary()

#%%
#Add classification layer using global average pooling
global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
class_names = train_dataset.class_names
output_dense = tf.keras.layers.Dense(len(class_names), activation='softmax') #when using softmax, logits in loss compiler must be set to False

#%%
#Use functional API to create the entire model (input pipeline + NN)
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_avg_pool(x)
outputs = output_dense(x)

model = tf.keras.Model(inputs,outputs)
model.summary()

#%%
#Compile model
adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
loss = tf.keras.losses.SparseCategoricalCrossentropy()

model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])

#%%
EPOCHS = 10
import datetime
log_path = r'X:\Users\User\Tensorflow Deep Learning\Tensorboard\concretecrack_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)
history = model.fit(train_dataset_pf, validation_data=validation_dataset_pf, epochs=EPOCHS, callbacks=[tb_callback])

#%%
#Now perform fine tuning, the goal is to train the higher level convolution layers. This is to allow high level features to be adapted to our own dataset
#1. Unfreeze base_model
base_model.trainable = True

#2. Freeze first n number of layers (leave the behind layers to be unfrozen so that they will be trained)
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

rmsprop = tf.keras.optimizers.RMSprop(learning_rate=0.00001)        
model.compile(optimizer=rmsprop, loss=loss, metrics=['accuracy'])
model.summary()

#%%
fine_tune_epoch = 10
total_epoch = EPOCHS + fine_tune_epoch

#Continue from previous checkpoint
history_fine = model.fit(train_dataset_pf, validation_data=validation_dataset_pf, epochs=total_epoch,initial_epoch=history.epoch[-1], callbacks=[tb_callback])

#%%
#Evaluate with test dataset
test_loss,test_accuracy = model.evaluate(test_dataset_pf)

print('------------------------Test Result----------------------------')
print(f'Loss = {test_loss}')
print(f'Accuracy = {test_accuracy}')

#%%
#Deploy model to make prediction
image_batch, label_batch = test_dataset_pf.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch)
class_predictions = np.argmax(predictions, axis=1)

#%%
#Plot the predictions
plt.figure(figsize=(10,10))

for i in range(4):
    axs = plt.subplot(2, 2, i+1)
    plt.imshow(image_batch[i].astype('uint8'))
    current_prediction = class_names[class_predictions[i]]
    current_label = class_names[label_batch[i]]
    plt.title(f"Prediction: {current_prediction}, Actual: {current_label}")
    plt.axis('off')
    
save_path = r"X:\Users\User\Tensorflow Deep Learning\github\graphs image"
plt.savefig(os.path.join(save_path,"concrete.png"),bbox_inches='tight')
plt.show()