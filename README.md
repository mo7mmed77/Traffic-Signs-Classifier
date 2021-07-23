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
[image7]: ./plots/Pred_images.png "Pred"


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

![alt text][image7] 


As it can be seen 8 out of 10 predictions were correct, which means the model has 80% accuracy. The test set accuracy was 94%.

#### Softmax Probability 

The top Five softmax probabilites for each prediction was:- 

'
Image: test-images\caution.jpg
Probabilities:
   1.000000 : 18 - General caution
   0.000000 : 11 - Right-of-way at the next intersection
   0.000000 : 26 - Traffic signals
   0.000000 : 27 - Pedestrians
   0.000000 : 34 - Turn left ahead

Image: test-images\Keep_Right.jpeg
Probabilities:
   1.000000 : 38 - Keep right
   0.000000 : 23 - Slippery road
   0.000000 : 20 - Dangerous curve to the right
   0.000000 : 31 - Wild animals crossing
   0.000000 : 29 - Bicycles crossing

Image: test-images\No_Entry.jpeg
Probabilities:
   0.999998 : 17 - No entry
   0.000001 : 35 - Ahead only
   0.000000 : 32 - End of all speed and passing limits
   0.000000 : 12 - Priority road
   0.000000 : 33 - Turn right ahead

Image: test-images\Pedestrian.jpeg
Probabilities:
   0.626803 : 0 - Speed limit (20km/h)
   0.373165 : 1 - Speed limit (30km/h)
   0.000026 : 18 - General caution
   0.000005 : 4 - Speed limit (70km/h)
   0.000000 : 27 - Pedestrians

Image: test-images\right_of_way.jpg
Probabilities:
   1.000000 : 11 - Right-of-way at the next intersection
   0.000000 : 18 - General caution
   0.000000 : 27 - Pedestrians
   0.000000 : 30 - Beware of ice/snow
   0.000000 : 28 - Children crossing

Image: test-images\Road Work.jpeg
Probabilities:
   1.000000 : 25 - Road work
   0.000000 : 20 - Dangerous curve to the right
   0.000000 : 36 - Go straight or right
   0.000000 : 29 - Bicycles crossing
   0.000000 : 22 - Bumpy road

Image: test-images\soeed_limit_50Km_h.jfif
Probabilities:
   0.999941 : 2 - Speed limit (50km/h)
   0.000059 : 1 - Speed limit (30km/h)
   0.000000 : 3 - Speed limit (60km/h)
   0.000000 : 5 - Speed limit (80km/h)
   0.000000 : 31 - Wild animals crossing

Image: test-images\Stop.jpeg
Probabilities:
   0.649651 : 3 - Speed limit (60km/h)
   0.190684 : 14 - Stop
   0.091150 : 2 - Speed limit (50km/h)
   0.030635 : 1 - Speed limit (30km/h)
   0.015675 : 13 - Yield

Image: test-images\Stop_sign.jpeg
Probabilities:
   0.702330 : 14 - Stop
   0.245311 : 18 - General caution
   0.033062 : 1 - Speed limit (30km/h)
   0.010140 : 15 - No vehicles
   0.002763 : 35 - Ahead only

Image: test-images\Yield.jpeg
Probabilities:
   1.000000 : 13 - Yield
   0.000000 : 35 - Ahead only
   0.000000 : 15 - No vehicles
   0.000000 : 9 - No passing
   0.000000 : 33 - Turn right ahead
'
