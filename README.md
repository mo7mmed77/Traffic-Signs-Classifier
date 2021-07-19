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

The training set accuracy was 100% while the validation accuracy is  98.8%. The EPOCH number was chosen to be 70. I believe it was enough until the model accuracy converged. 

#### Model Convergance
![Alt text](/plots/Model_accuracy.png?raw=true "Title")


#### Model Prediction Accuracy 
The model was given 10 test images to test its accuracy. It was able to identify 9 of the images correctly, which technically means 90% prediction accuracy. However, when the number of testing images increasing we might see a better represenation of the prediction. 
