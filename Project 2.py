import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

image_height = 500
image_width = 500
channels = 3
input_shape = (image_height, image_width, channels)

test_dir = "/Users/williamsparenberg/Documents/GitHub/Project-2/Data/test/"
train_dir = "/Users/williamsparenberg/Documents/GitHub/Project-2/Data/train/"
valid_dir = "/Users/williamsparenberg/Documents/GitHub/Project-2/Data/valid/"

augment_train_data = ImageDataGenerator(
    rescale=1./255,   
    zoom_range=.2,
    shear_range=.2,
    horizontal_flip=True
)

train_generator = augment_train_data.flow_from_directory(
    directory=train_dir,
    target_size=(image_height, image_width), 
    batch_size=32,
    class_mode='categorical'
)

augment_valid_data = ImageDataGenerator(
    rescale=1./255    
)

valid_generator = augment_valid_data.flow_from_directory(
    directory=valid_dir,
    target_size=(image_height, image_width),
    batch_size=32,
    class_mode='categorical'
)

model = Sequential([
    Conv2D(
        filters=32, 
        kernel_size=(3, 3), 
        strides=(1, 1), 
        activation='relu', 
        input_shape=input_shape
    ),
    MaxPooling2D(pool_size=(2, 2)), 
    Conv2D(
        filters=64, 
        kernel_size=(3, 3),  
        activation='relu'
    ),
    MaxPooling2D(pool_size=(2, 2)), 
    Conv2D(
        filters=128,             
        kernel_size=(3, 3),  
        activation='relu'
    ),
    MaxPooling2D(pool_size=(2, 2)), 
    Flatten(), 
    Dense(
        units = 256, 
        activation = 'relu'
    ),
    Dropout(
        rate=0.75
    ),
    Dense(
        units= 3, 
        activation='softmax'
    )
])

model.compile(
    optimizer = Adam(0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(
    monitor='val_loss',     
    patience=5,                           
    restore_best_weights=True
    )

history = model.fit(
    train_generator,
    epochs=40,
    validation_data=valid_generator,
    callbacks=[early_stopping]
)

model.summary()

model.save("best.model")

