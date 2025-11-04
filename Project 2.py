from tensorflow import tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from keras.preprocessing import image_dataset_from_directory

image_height = 500
image_width = 500
channels = 3
input_shape = (image_height, image_width, channels)

test_dir = "/Users/williamsparenberg/Documents/GitHub/Project-2/Data/test/"
train_dir = "/Users/williamsparenberg/Documents/GitHub/Project-2/Data/train/"
valid_dir = "/Users/williamsparenberg/Documents/GitHub/Project-2/Data/valid/"

augment_train_data = ImageDataGenerator(
    train_dir,
    rescale = 1./255,
    zoom_range = .2,
    shear_range = .2
    )

train_generator = image_dataset_from_directory(
    train_dir, 
    image_size = (image_height, image_width), 
    label_mode = 'categorical',
    batch_size = 32,
    color_mode =  'rgb'
    )

augment_valid_data = ImageDataGenerator(
    valid_dir,
    rescale = 1./255,
    zoom_range = .2,
    shear_range = .2
    )

valid_generator = image_dataset_from_directory(
    valid_dir, 
    image_size = (image_height, image_width), 
    label_mode = 'categorical',
    batch_size = 32,
    color_mode =  'rgb'
    )

model = Sequential([
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation='relu',
        input_shape=input_shape