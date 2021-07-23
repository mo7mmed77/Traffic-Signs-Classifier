# **Traffic Sign Recognition** 

---
## Overview
The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./plots/Histogram_of_classes.png "Visualization"
[image2]: ./plots/dataset.png "DataSet"
[image3]: ./plots/Original_images.png "Original"
[image4]: ./plots/Preprocessed.png "Preprocessed"
[image5]: ./plots/Model_accuracy.png "Accuracy"

[image6]: ./plots/Test_images.png "Test Images"
[image7]: ./test-images/Keep.png "Caution"


### Data Set Summary & Exploration

#### 1. Basic summary of the data set. 


I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43. 

####  Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.
![alt text][image2]
This is a bar chart showing how the data images represented in each class of the traffic sign. For some classes the number of image are high which will result in high accuracy prediction for that class. 

![alt text][image1]

### Design and Test a Model Architecture

#### Processing of the Image Data
The 'pre_process_images' function takes the images then convert it to gray scale then normalize it, to be ready for the training step. The normalization step is done to make the training more efficient and faster. The conversion to gray scale was done to eliminate the effect of colouring on the classification and make it solely about the shape and the curves of the traffic sign. 

As a last step, I normalized the image data because it will be more effecient for the training process and takes less time. 

Here is an example of an original image and an augmented image:


![alt text][image3]
![alt text][image4]


#### Final model architecture 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 GrayScale image   							| 
| Convolution 3x3     	| 1x1 stride, Valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    | 1x1 stride, valid padding     									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Flatten	      	|  outputs 5x5x64 				|
| Fully connected		|outputs 1600    									|
| RELU					|												|
| Dropout					|												|
| Fully connected		|outputs 43    									|
|						|												|
 
Max pooling layers are  used to reduce layer dimensions to a minimal size.

#### Model Training.

The model was trained with the following hyper-parameters:

* Optimizer: Adam
* Batch Size: 128
* Epochs: 70
* Learning Rate: 1E-3

These values are mostly typical except for the Ephochs I chose this number based on many training tests and it seems to be the optimal number in case of convergance within minimum time needed. 

The Final model results were:-
* training set accuracy of 99.4%
* validation set accuracy of 100% 
* testing set accuracy of 94.2%

The number of Epochs plays a big role in increasing the accuracy of the model. Moreover, the learning rate was suffeicient enough to reach this validation set accuracy. 

![alt text][image5]


### Test a Model on New Images

#### Testing the model with 10 images of Traffic Signs

Here are the 10 traffic signs that i downloaded some of them from the internet. 

![alt text][image6] 

This number of images might not be enough for giving a valid result of the accurcay of the model, however it should give a good idea. 

#### Model Prediction

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 


