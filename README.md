# Food-Classification

## About
<br>Food 101 is a labelled data set with 101 different food classes. Each food class contains 1000 images. Using the data provided,  a Machine Learning Model that can classify 3 classes in Food 101 dataset is trained.

**Classes:** (Apple_pie, Baby_back_ribs, Baklava)
<br>**Epoches:** 100
<br>**Batch_size:** 64

## Table of Contents
+ [About](#about)
+ [Exploratory Data Analysise](#exploratory-data-analysis)
+ [Data Augmentation](#data-augmentation)
+ [Model](#model)
+ [Training](#training)
+ [Results Evaluation](#results-evaluation)
+ [Conclusion](#conclusion)
+ [Reference](#reference)


Images are split to train and test set with 750 and 250 images per class respectively. 

<img src="https://github.com/rileykwok/Food-Classification/blob/master/img/files.PNG" width="300">

## Exploratory Data Analysis

Let's preview some of the images.

<img src = "https://github.com/rileykwok/Food-Classification/blob/master/img/EDA.PNG" width="1000">

As shown here, the quality of the images are not very good: 
- different background
- dissimilar lightings
- wrong labels 
   - empty plate in the first pic of the baklava
   - looks more like a pizza than an apple pie for the first apple pie pic
   - missing apple pie in the last apple pie pic
   - strange colours of the last baby pork ribs pic

The size of the images are also inconsistent as shown in the height against width plot shown below:

<img src = "https://github.com/rileykwok/Food-Classification/blob/master/img/picsize.PNG" width="400">


## Data Augmentation

Since the data set for each class is relatively small to train a good neural network, an image data generator from Keras is used for image tranformation to expand the dataset and to reduce the overfitting problem.

```python
train_datagen = ImageDataGenerator(featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 rotation_range=10,
                 width_shift_range=0.05,
                 height_shift_range=0.05,
                 shear_range=0.1,
                 zoom_range=0.2,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=True,
                 vertical_flip=False,
                 rescale=1/255)
#rescale to [0-1], add zoom range of 0.2x, width and height shift and horizontal flip

train_generator = train_datagen.flow_from_directory(
        "../input/food102/food101/food101/train",
        target_size=(224,224),                     # resize images to (224,224) to increase the training speed and efficiency
        batch_size=64)

test_datagen = ImageDataGenerator(rescale=1./255)  # rescale to [0-1] for testing set
test_generator = test_datagen.flow_from_directory(
        "../input/food102/food101/food101/test",
        target_size=(224,224),
        batch_size=64)
```
Check the images from data generator. As shown, the images are slightly distorted and rotated. This shall enable the model to learn the important features of the images and produce a more robust model.

<img src = "https://github.com/rileykwok/Food-Classification/blob/master/img/datagen.PNG" width="1000">

## Model
To create a convolution neural network to classfied the images, Keras Sequencial model is used.


Convol
**Batch normalisation:** Tested with batch normalisation layers and removed all dropout layers. It results in faster training and higher learning rates, but it caused more overfitting (large diffence between train and test accuracy) than dropout, thus batch normalisation has not been used in this case.
**Optimizers:** Adam final accuracy slightly out-performs RMSProp and also converge to minima faster than RMSProp as it's similar to RMSProp + Momentum.
**Initializers:** 


## Training

<img src = "https://github.com/rileykwok/Food-Classification/blob/master/img/history.PNG" width="1000">


## Results Evaluation

The confusion matrix of 750 test images:

As shown, most of the wrong prediction are between apple pie and baklava. This can be explained by that fact that both of these food types have similar texture and colour, as both are made from pastry.

Now, let's examine in more detail how the model performs and evaluate those 'wrong-est' predictions.
To determine 'how wrong' the model predicts each images, the wrongly predicted images are sorted by the difference between the *probability of predicted label* and the *probability of the true class label*

<img src = "https://github.com/rileykwok/Food-Classification/blob/master/img/wrongpredictions.PNG" width="1000">


## Conclusion



## Reference


Keras Intializers: 
https://towardsdatascience.com/hyper-parameters-in-action-part-ii-weight-initializers-35aee1a28404
https://becominghuman.ai/priming-neural-networks-with-an-appropriate-initializer-7b163990ead
Optimizers:
http://ruder.io/optimizing-gradient-descent/index.html#visualizationofalgorithms
Batch Normalization:
https://www.dlology.com/blog/one-simple-trick-to-train-keras-model-faster-with-batch-normalization/
https://arxiv.org/abs/1801.05134
https://medium.com/deeper-learning/glossary-of-deep-learning-batch-normalisation-8266dcd2fa82

