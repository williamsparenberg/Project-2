import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

model_path = "best.model.keras"
model = load_model(model_path)

class_labels = ['crack', 'missing-head', 'paint-off']
base_data_dir = "/Users/williamsparenberg/Documents/GitHub/Project-2/"
test_dir = os.path.join(base_data_dir, "Data/test")
image_height = 500
image_width = 500

test_images = {
    'crack': 'Data/test/crack/test_crack.jpg',
    'missing-head': 'Data/test/missing-head/test_missinghead.jpg',
    'paint-off': 'Data/test/paint-off/test_paintoff.jpg'
    }

for image_key, relative_path in test_images.items():
    
    full_image_path = os.path.join(base_data_dir, relative_path)
    print(f"\n--- Testing Image: {image_key} ---")

    img = image.load_img(full_image_path, target_size=(image_height, image_width))
    img_array = image.img_to_array(img)
    
    img_batch = np.expand_dims(img_array, axis=0)
    norm_img = img_batch / 255.0

    predict = model.predict(norm_img)
    
    predicted_class_index = np.argmax(predict[0])
    
    predicted_class = class_labels[predicted_class_index]
    confidence_score = predict[0][predicted_class_index] * 100

    print(f"Model predicted: **{predicted_class}** with **{confidence_score:.2f}%** confidence.")

    plt.figure()
    plt.imshow(img)
    plt.title(f"Actual: {image_key} | Predicted: {predicted_class} ({confidence_score:.2f}%)")
    plt.axis('off')
    plt.show()

