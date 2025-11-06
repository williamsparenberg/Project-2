import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

image_height = 500
image_width = 500
channels = 3
input_shape = (image_height, image_width, channels)

train_dir = "/Users/williamsparenberg/Documents/GitHub/Project-2/Data/train/"
valid_dir = "/Users/williamsparenberg/Documents/GitHub/Project-2/Data/valid/"

#Data Prepping
augment_train_data = ImageDataGenerator(
    rescale = 1./255,   
    zoom_range = 0.2,
    shear_range = 0.2,
    horizontal_flip = True
)

train_generator = augment_train_data.flow_from_directory(
    directory = train_dir,
    target_size = (image_height, image_width), 
    batch_size = 32,
    class_mode = 'categorical'
)

augment_valid_data = ImageDataGenerator(
    rescale = 1./255    
)

valid_generator = augment_valid_data.flow_from_directory(
    directory = valid_dir,
    target_size = (image_height, image_width),
    batch_size = 32,
    class_mode = 'categorical'
)

#Model Creation
model = Sequential([
    Conv2D(
        filters = 32, 
        kernel_size = (3, 3), 
        strides = (1, 1), 
        activation = 'relu', 
        input_shape = input_shape
    ),
    MaxPooling2D(pool_size = (2, 2)), 
    Conv2D(
        filters = 64, 
        kernel_size = (3, 3),  
        activation = 'relu'
    ),
    MaxPooling2D(pool_size = (2, 2)), 
    Conv2D(
        filters = 128,             
        kernel_size = (3, 3),  
        activation = 'relu'
    ),
    MaxPooling2D(pool_size = (2, 2)), 
    Flatten(), 
    Dense(
        units = 256, 
        activation = 'relu'
    ),
    Dropout(
        rate = 0.75
    ),
    Dense(
        units = 3, 
        activation = 'softmax'
    )
])

model.compile(
    optimizer = Adam(0.0005),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

early_stopping = EarlyStopping(
    monitor = 'val_loss',     
    patience = 5,                           
    restore_best_weights = True
    )

#Hyperparameter analysis
def plot_training_metrics(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))
    
    plt.figure(figsize = (12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label = 'Training Accuracy')
    plt.plot(epochs_range, val_acc, label = 'Validation Accuracy')
    plt.legend(loc = 'lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label = 'Training Loss')
    plt.plot(epochs_range, val_loss, label = 'Validation Loss')
    plt.legend(loc = 'upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.show()
    
history = model.fit(
    train_generator,
    epochs = 40,
    validation_data = valid_generator,
    callbacks=[early_stopping]
)

model.summary()
model.save("best.model.keras")

plot_training_metrics(history)

