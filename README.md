# Food-Classification
**About**
<br>Food 101 is a labelled data set with 101 different food classes. Each food class contains 1000 images. Using the data provided,  a Machine Learning Model that can classify 3 classes in Food 101 dataset is trained.

**Classes:** (Apple_pie, Baby_back_ribs, Baklava)
<br>**Epoches:** 100
<br>**Batch_size:** 64

Images are split to train and test set with 750 and 250 images per class respectively.  
![img](files)

## Exploratory Data Analysis

Let's preview some of the images.
![img](EDA)
As shown here, the quality of the images are not very good: 
- with different background (noise)
- different lightings
- and even wrong labels (e.g. empty plate in the first pic of the baklava, missing apple pie in the last apple pie pic).

The size of the images are also inconsistent as shown in the height against width plot shown below:
![img](picsize)



## Data Augmentation

Since the data set for each class is relatively small to train a good neural network, an image data generator from Keras is used for image tranformation to expand the dataset and to reduce the overfitting problem.

```python
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True, rotation_range=10) 
#rescale to [0-1], add zoom range of 0.2x and horizontal flip
train_generator = train_datagen.flow_from_directory(
        "../input/food102/food101/food101/train",
        target_size=(224,224),
        batch_size=64)
```


![img](datagen)

![img](wrongpredictions)
