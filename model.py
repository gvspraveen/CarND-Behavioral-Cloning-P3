from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import numpy as np

from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D


from helper import parse_driving_log, preproces_datasets, read_image

default_batch_size = 128
def batch_generater(np_training_paths, np_training_angles, batch_size=default_batch_size):
    num_samples = len(np_training_paths)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_training_paths = np_training_paths[offset:offset+batch_size]
            batch_training_angles = np_training_angles[offset:offset+batch_size]
            batch_training_images = []
            for training_path in batch_training_paths:
                img = read_image(training_path)
                batch_training_images.append(img)
            yield shuffle(np.array(batch_training_images), batch_training_angles)


# ==================================================================
#
# Training code starts here
#
# ==================================================================

all_samples, all_angles = parse_driving_log()
train_samples, validation_samples, training_angles, validation_angles = train_test_split(all_samples, all_angles, test_size=0.35)
test_samples, validation_samples, test_angles, validation_angles = train_test_split(validation_samples, validation_angles, test_size=0.35)

train_samples, training_angles = shuffle(train_samples, training_angles)
test_samples,  test_angles = shuffle(test_samples, test_angles)
validation_samples, validation_angles = shuffle(validation_samples, validation_angles)

np_training_paths = np.array(train_samples)
np_training_angles = np.array(training_angles)
np_validation_paths = np.array(validation_samples)
np_validation_angles = np.array(validation_angles)
np_test_paths = np.array(test_samples)
np_test_angles = np.array(test_angles)

print("Training shape: {}, {}".format(np_training_paths.shape, np_training_angles.shape))
print("Validation shape: {}, {}".format(np_validation_paths.shape, np_validation_angles.shape))
print("Test shape: {}, {}".format(np_test_paths.shape, np_test_angles.shape))

# Pre process data sets.
# This step does following.
# 1. Evaluate distribution of data and remove samples with too many images (to avoid overfitting). This is because track has too many zero angle driving scenarios
# 2. Generate a fake data for angles with very few samples. I use image transformation techniques to generate fake data from existing data.
np_training_paths, np_training_angles = preproces_datasets(np_training_paths, np_training_angles)
np_training_paths, np_training_angles = shuffle(np_training_paths, np_training_angles)

train_generator = batch_generater(np_training_paths, np_training_angles)
validation_generator = batch_generater(np_validation_paths, np_validation_angles)
test_generator = batch_generater(np_test_paths, np_test_angles)

def resize_image(image):
    from keras.backend import tf
    return tf.image.resize_images(image, (66, 200))

def generate_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

    # Using NVDIA model as suggested https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    # This is one of the hints provided in project requirements

    # NVDIA model uses 66*200*3 image
    model.add(Lambda(resize_image))

    # Normalize
    model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(66,200,3)))

    # Layer1 - 5X5 kernel, stride length = 2, channels = 24, activation = relu
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))

    # Layer2 - 5X5 kernel, stride length = 2, channels = 36, activation = relu
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))

    # Layer3 - 5X5 kernel, stride length = 2, channels = 48, activation = relu
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))

    # Layer4 - 3×3 kernel size, non-strided, channels=64
    model.add(Convolution2D(64, 3, 3, activation="relu"))

    # Layer5 - 3×3 kernel size, non-strided, channels=64
    model.add(Convolution2D(64, 3, 3, activation="relu"))

    # Layer6 - flatten to 1164 neurons
    model.add(Flatten())

    # Add a drop out to prevent overfitting
    model.add(Dropout(0.60))

    # Layer7 - fully connected layer
    model.add(Dense(100))
    # Layer8 - fully connected layer
    model.add(Dense(50))
    # Layer9 - fully connected layer
    model.add(Dense(10))
    # Layer10 - fully connected layer
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    return model


learning_rate=0.0001
epochs = 3
model = generate_model()
model.fit_generator(train_generator,
                    samples_per_epoch=len(np_training_paths),
                    validation_data=validation_generator,
                    nb_val_samples=len(np_validation_paths),
                    nb_epoch=epochs)

model.save('model.h5')
print ("Model saved")
print("========================================================")

test_loss = model.evaluate_generator(test_generator, int(len(np_test_paths)/default_batch_size))
print ("Test Loss : {:.4f}".format(test_loss))



