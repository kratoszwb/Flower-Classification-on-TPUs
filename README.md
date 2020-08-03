# Flower Classification with Neural Network

In this program we're classifying 104 types of flowers based on their images drawn from five different public datasets.

## Abstract

Computer Vision is an trending research filed of artificial intelligence, which aims to train computers to interpret and understand the visual world. Through training, machines is able to accurately identify and classify objects. In auto pivoting of intelligence automotive and fancy apps in our smart phones, a lot of applications related with objective detection and cognization are affecting our normal life.  The development of deep learning with Convolutional Neural Network(CNN) improving the computer vision research to a higher level. CNN is a category of Neural Networks that is mostly applied to visual imagery analysis. I will illustrate how diverse models based on CNN help us with recognizing and classifying more than 100 kinds of flowers in a fine grained level with the accuracy of nearly 100%. And this challenge is more efficient solved on the basis of TPUs.

## Exploratory Data Analysis

As we could see from the graph below, the numbers of different kinds of flowers differ a lot so that we may face the problem of class imbalance. In order to get rid of overfitting, we use the data augmentation to deal with this problem. Based on all the images already exist, random rotation, shear, zoom, shift, and more advanced approach can be done on the minority dataset by data augumentation. Besides, we use external database including other pictures of flowers to increase the training set. The process of data augmentation is in the file named 'Helper_Functions.py'.

![alt](https://github.com/kratoszwb/Flower-Classification-on-TPUs/blob/master/image/EDA_BarChart.png)

## Analysis Process

Following Convolutional Neural Network are planned to be used:

-	VGG-16
-	Xception
-	DenseNet
-	EfficientNet
-	InceptionV3
-	ResNet
-	MobileNetV2
-	InceptionResnetV2

All the models are built in file named 'Models.py'. We use one model each time and all the other models are commented out. We collect results from
all the models for ensembling. The process of ensembling is in the file named 'Ensemble.py'.

On the basis of pre-trained models(with weights of ImgaeNet), we adopt transfer learning to solve the flower classfication problem.

We customize a loss function named cos-layer which is defined in file named Models.py'.

Here is an example of the result of validation process.

![alt](https://github.com/kratoszwb/Flower-Classification-on-TPUs/blob/master/image/Validation.png)
