from keras.models import Sequential
from keras.layers import Dense
from keras.applications import vgg16

#importing vgg16 
model=vgg16.VGG16()

#checking the summary and type of model
model.summary()
type(model)

#changing model into a Sequential model and removing the last layer
model1=Sequential()
i=0
for layer in model.layers:
    i=i+1
    
for layer in model.layers:
    if i>2:
        model1.add(layer)
    i=i-1
    
#disabling trainablity of layers so that their wheight don't change
for layer in model1.layers:
    layer.trainable=False
    
#adding layers in model

model1.add(Dense(units = 128, activation = 'relu'))
model1.add(Dense(units = 1, activation = 'sigmoid'))




model1.summary()

# Compiling the CNN
model1.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (224,224),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'binary')

model1.fit_generator(training_set,
                     samples_per_epoch = 6283,
                     nb_epoch = 7,
                     validation_data = test_set,
                     nb_val_samples = 1626)


