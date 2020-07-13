# Topic

In this competition we're classifying 104 types of flowers based on their images drawn from five different public datasets.
Some classes are very narrow, containing only a particular sub-type of flower (e.g. pink primroses) while other classes contain many sub-types (e.g. wild roses).
![](images/EDA_BarChart.png)
# Method

We use external database including other pictures of flowers to increase the training set. On the basis of pre-trained models(with weights of ImgaeNet),
we adopt transfer learning to solve theflower classfication problem. Also, we use several ways to do data augmentation in order to increase the training set and 
solve the problems of class imbalance. 

We customize a loss function named cos-layer which is defined in file named 'inceptionv3-md1.ipynb'.

# Model

Following Convolutional Neural Network are planned to be used:

-	VGG-16
-	Xception
-	DenseNet
-	EfficientNet
-	InceptionV3
-	ResNet
-	MobileNetV2
-	InceptionResnetV2

All the models are built in file named 'inceptionv3-md1.ipynb'. We use one model each time and all the other models are commented out. We collect results from
all the models for ensembling. The process of ensembling is in the file named 'submissions-ensembling.ipynb'.

