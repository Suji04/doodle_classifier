from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

# conv layer 1
model.add(Convolution2D(filters = 32,
                        kernel_size = (3,3),
                        activation = "relu",
                        input_shape = (28,28,1)))

# conv layer 2
model.add(Convolution2D(filters = 32,
                        kernel_size = (3,3),
                        activation = "relu"))

# pooling layer 1
model.add(MaxPooling2D(pool_size=(2,2)))

# deactivating some elements
model.add(Dropout(.20))

# conv layer 3
model.add(Convolution2D(filters = 64,
                        kernel_size = (3,3),
                        activation = "relu"))
                        

# conv layer 4
model.add(Convolution2D(filters = 64,
                        kernel_size = (3,3),
                        activation = "relu"))
                       

# pooling layer 2
model.add(MaxPooling2D(pool_size=(2,2)))

# deactivating some elements
model.add(Dropout(.1))

# flatten layer
model.add(Flatten())

# fully connected layer
model.add(Dense(256, activation="relu"))
model.add(Dense(output_dim=5, activation="softmax"))

model.compile(optimizer = "adam", loss="categorical_crossentropy", metrics=["accuracy"])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'data/train_set',
        target_size=(28, 28),
        color_mode="grayscale",
        batch_size=128,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'data/test_set',
        target_size=(28, 28),
        color_mode="grayscale",
        batch_size=128,
        class_mode='categorical') 

model.fit_generator(
        training_set,
        steps_per_epoch=12000,
        epochs=5,
        validation_data=test_set,
        validation_steps=3000)

model.save("model_new.h5")
model_json = model.to_json()
with open("model_new.json", "w") as json_file:
    json_file.write(model_json)

