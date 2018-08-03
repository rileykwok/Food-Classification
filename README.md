# Food-Classification

## Table of Contents
+ [About](#about)
+ [Exploratory Data Analysise](#exploratory-data-analysis)
+ [Data Augmentation](#data-augmentation)
+ [Model](#model)
+ [Training](#training)
+ [Results Evaluation](#results-evaluation)
+ [Conclusion](#conclusion)
+ [Reference](#reference)

## About

Food 101 is a labelled data set with 101 different food classes. Each food class contains 1000 images. Using the data provided,  a deep learning model built on Keras/TensorFlow is trained to classify 3 classes in Food 101 dataset.

**Classes:** (Apple_pie, Baby_back_ribs, Baklava)
<br>**Epoches:** 100
<br>**Batch_size:** 64

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
   - looks like an ice cream than an apple pie in the last apple pie pic
   - strange colours of the last baby pork ribs pic

The size of the images are also inconsistent as shown in the height against width plot shown below, but all the images have at least one side with 512 pixels, so we dont have to worry about extremely small images that is pixelated.

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
        target_size=(224,224),       # resize images to (224,224) to increase the training speed and efficiency
        batch_size=64)

test_datagen = ImageDataGenerator(rescale=1./255)    # rescale to [0-1] for testing set

test_generator = test_datagen.flow_from_directory(
        "../input/food102/food101/food101/test",
        target_size=(224,224),
        batch_size=64)
```
Check the images from data generator. As shown, the images are slightly distorted and rotated. This shall enable the model to learn the important features of the images and produce a more robust model.

<img src = "https://github.com/rileykwok/Food-Classification/blob/master/img/datagen.PNG" width="1000">

## Model
To create a convolution neural network to classfied the images, Keras Sequencial model is used.

```python
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5), strides = 2, padding = 'Same', activation ='relu', input_shape = (224,224,3), kernel_initializer='he_normal'))
model.add(Conv2D(filters = 32, kernel_size = (5,5), strides = 2, padding = 'Same', activation ='relu',kernel_initializer='he_normal'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu',kernel_initializer='he_normal'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu',kernel_initializer='he_normal'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(filters = 128, kernel_size = (2,2),padding = 'Same', activation ='relu',kernel_initializer='he_normal'))
model.add(Conv2D(filters = 128, kernel_size = (2,2),padding = 'Same', activation ='relu',kernel_initializer='he_normal'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(filters = 256, kernel_size = (2,2),padding = 'Same', activation ='relu',kernel_initializer='he_normal'))
model.add(Conv2D(filters = 256, kernel_size = (2,2),padding = 'Same', activation ='relu',kernel_initializer='he_normal'))
model.add(GlobalAveragePooling2D())
model.add(Dense(512, activation = "relu",kernel_initializer='he_normal'))
model.add(Dropout(0.2))
model.add(Dense(3, activation = "softmax",kernel_initializer='he_normal',kernel_regularizer=l2()))

#callbacks
checkpointer = ModelCheckpoint(filepath='model.hdf5', verbose=1, save_best_only=True, save_weights_only=True)
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=20, mode='auto')
reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, mode='auto')

model.compile(optimizer = 'Adam' , loss = "categorical_crossentropy", metrics=["accuracy"])

history = model.fit_generator(train_generator,steps_per_epoch=2250/64,
                              validation_data=test_generator,validation_steps=750/64, 
                              epochs=100, callbacks=[checkpointer, reduceLR, earlystopping])
```


**Batch normalisation:** Tested with batch normalisation layers and removed all dropout layers. It results in faster training and higher learning rates, but it caused more overfitting (large diffence between train and test accuracy) than dropout, thus batch normalisation has not been used in this case.

**Optimizers:** *Adam* final accuracy slightly out-performs *RMSProp* and also converge to minima faster as it's similar to *RMSProp + Momentum*.

**Activation Function:** 
*ReLu* activation used at convolution layers to produce a sparse matrix, which requires less computational power then sigmoid or tanh which produce dense matrix. Also, it reduced the likelihood of vanishing gradients. When a>0, the gradient has a constant value, so it results in faster learning than sigmoids as gradients becomes increasingly small as the absolute value of x increases. 
*Softmax* activation used at the last layer to assign the probability of each class.

**Initializers:** Kernal weights are initialized using *He normal* initializers which helps to attain a global minimum of the cost function faster and more efficiently.The weights differ in range depending on the size of the previous layer of neurons and this is a good inializer to be used with *ReLu* activation function.

**Regularization:** *L2 regularization* is implemented aim to decrease the complexity of the model and minimise overfitting by penalising weights with large magnitudes. 


## Training

<img src = "https://github.com/rileykwok/Food-Classification/blob/master/img/history.PNG" width="1000">


Model accuracy increased over each epoch, overfitting started at around 40 epochs. The model achieved validation accuracy of **78.9%** with a 0.49 cross entropy validation loss.

## Results Evaluation

Preview some predictions from the model:

<img src = "https://github.com/rileykwok/Food-Classification/blob/master/img/prediction.PNG" width="1000">

The confusion matrix of 750 test images:

<img src = "https://github.com/rileykwok/Food-Classification/blob/master/img/cm.PNG" width="500">

As shown in the confusion matrix, most of the wrong prediction are between apple pie and baklava. To visualise the model performance for each class, ROC curve is plotted on the true positive rate against false positive rate. As anticipated, the ROC curve for the baby pork rib class has the best performance with an AUC score of 0.99. The performance of the apple pie and baklava are not as good, this might be explained by that fact that both of these food types have similar texture and colour, as both are made from pastry and the model finds it harder to classify between them.

<img src = "https://github.com/rileykwok/Food-Classification/blob/master/img/roc.PNG" width="600">

Now, let's examine in more detail how the model performs and evaluate those 'wrong-est' predictions.
To determine 'how wrong' the model predicts each images, the wrongly predicted images are sorted by the `difference between the *probability of predicted label* and the *probability of the true class label*`

<img src = "https://github.com/rileykwok/Food-Classification/blob/master/img/wrongpredictions.PNG" width="1000">


## Conclusion

With the given data sets for 3 classes of food: apple pie, baby pork ribs and baklavas, the model final accuracy reached 78.9%  with cross entropy validation loss of 0.49. The main cause of error is due to the similarity between baklavas and applie pie as they both exhibit alike texture and colours.


## Reference

Food-101 Dataset
- [Food-101 – Mining Discriminative Components with Random Forests](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)

Weight Intializers: 
- [Hyper-parameters in Action! Part II — Weight Initializers](https://towardsdatascience.com/hyper-parameters-in-action-part-ii-weight-initializers-35aee1a28404)
- [Priming neural networks with an appropriate initializer.](https://becominghuman.ai/priming-neural-networks-with-an-appropriate-initializer-7b163990ead)

Optimizers:
- [An overview of gradient descent optimization algorithms](http://ruder.io/optimizing-gradient-descent/index.html#visualizationofalgorithms)

Batch Normalization:
- [Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift](https://arxiv.org/abs/1801.05134)
- [Glossary of Deep Learning: Batch Normalisation](https://medium.com/deeper-learning/glossary-of-deep-learning-batch-normalisation-8266dcd2fa82)

Activation Functions:
- [Understanding Activation Functions in Neural Networks](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0)

Grad CAM:
- [Grad-CAM: Why did you say that? Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/pdf/1610.02391v1.pdf)
- [Deep Learning with Python Book by Francois Chollet](http://www.deeplearningitalia.com/wp-content/uploads/2017/12/Dropbox_Chollet.pdf)

Regularization: 
- [L1 and L2 Regularization Methods](https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c)
