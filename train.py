import tensorflow as tf
import keras.preprocessing.image

def buildLeNetModel(imageWidth: int, imageHeight: int, imageChannel: int):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(6, (3,3), activation='relu', input_shape=(imageWidth, imageHeight, imageChannel)))
    model.add(keras.layers.AveragePooling2D())

    model.add(keras.layers.Conv2D(16, (3,3), activation='relu'))
    model.add(keras.layers.AveragePooling2D())

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(120, activation='relu'))
    model.add(keras.layers.Dense(84, activation='relu'))
    model.add(keras.layers.Dense(7, activation='softmax'))

    return model

if __name__ == '__main__':
    # Solve Problem
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    # Data Preprocessing
    train_data = keras.preprocessing.image.ImageDataGenerator(rotation_range=15, rescale=1./255, shear_range=0.1, zoom_range=0.2, width_shift_range=0.1, height_shift_range=0.1)\
        .flow_from_directory("./training/", target_size=(32, 32), color_mode='grayscale', class_mode='categorical', batch_size=16)
    valid_data = keras.preprocessing.image.ImageDataGenerator(rotation_range=15, rescale=1./255, shear_range=0.1, zoom_range=0.2, width_shift_range=0.1, height_shift_range=0.1)\
        .flow_from_directory("./valid/", target_size=(32, 32), color_mode='grayscale', class_mode='categorical', batch_size=16)

    # Model Preparation
    # Resize all input data into 32x32
    LeNetModel = buildLeNetModel(32, 32, 1)
    LeNetModel.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    LeNetModel.summary()

    # Training
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    callbacks = [EarlyStopping(patience=10), ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)]

    epochs = 100
    history = LeNetModel.fit_generator(
        train_data, 
        epochs=epochs,
        validation_data=valid_data, 
        callbacks=callbacks
    )
    LeNetModel.save('LeNet.h5')
    
    # RMSprop
    # Epoch=20 0.8738 0.7214 with 128x128 without LR Reducing (Highest at Epoch18 0.8636 0.8411)
    # Epoch=10 0.8569 0.8311 with 128x128 with LR Reducing (Highest at Epoch 9 0.8519 0.8700)
