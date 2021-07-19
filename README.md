## Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This pipeline uses Convolutional Neural Network (CNN) to classify traffic signs. The model used is Lnet. The input to the model is the German Traffic Sign Dataset. The model should understand these signs and return a prediction. The whole pipeline can be found in the python jupter script 'Traffic_Sign_Classifier.ipynb'.  




### The Pipeline Steps:-
* It Loads the data set
* It visualize the data set
* Then it Design, train and test the Lnet model architecture
* Then it uses the model to make predictions on new images
* Analyze the softmax probabilities of the new images

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset

Download the [data set](https://s3.amazonaws.com/video.udacity-data.com/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip). 

### Summary and Results

The training set accuracy was 97 while the validation accuracy is  %. The EPOCH number was chosen to be 70. I believe it was enough until the model accuracy converged. 
