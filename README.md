# Image Classificaton of Concrete Crack using Transfer Learning
## Problem Statement
The aim of this project is to create a convolutional neural network (CNN) model that can identify between a good image of concrete or a concrete with cracks. The model is trained using 40000 which half of it consists of good condition concretes while the other half are cracked concretes.

Dataset credit: [source](https://data.mendeley.com/datasets/5y9wdsg2zt/2)

## Methodology
#### IDE and Library
This project is made using Spyder. The main libraries used in this project are Numpy, Tensorflow Keras and Matplotlib.
#### Model Pipeline
Transfer learning is used to make the model of this project. Firstly, the input layer for this model is coloured images with dimension of 224x224. Next, the input pipeline is finished using a buffered prefetching which is the AUTOTUNE method.

For feature extractor a pretrained model of MobileNetV2 is used. More detail about this module can be obtained [here](https://www.tensorflow.org/api_docs/python/tf/keras/applications).

Lastly, global average pooling and dense layer are used as the classifier. The output is a softmax signals which is used to identify the predicted binary class.

## Results
The model is trained using the train dataset and evaluated using the test dataset. The test result are as show below:

<p align="center">
  <img width="800" src="https://github.com/HazuanAiman/Concrete_crack_classification/blob/main/images/concrete%20crack%20result.PNG">
</p>
<br>
<br>

Figure below shows example of the prediction and actual results of the images.
<p align="center">
  <img src="https://github.com/HazuanAiman/Concrete_crack_classification/blob/main/images/concrete.png">
</p>
  
