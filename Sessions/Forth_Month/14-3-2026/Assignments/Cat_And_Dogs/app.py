import gradio as gr
import tensorflow 
from tensorflow.keras.models import  load_model
import numpy as np
from PIL import Image

model = tensorflow.keras.models.load_model('model.h5')

IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

def predict_image(img):
    
    img_resized = img.resize(IMAGE_SIZE)
    img_array = tensorflow.keras.preprocessing.image.img_to_array(img_resized)
    
  
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

   
    prediction = model.predict(img_array, verbose=0)[0]

    
    if len(prediction) == 1:
        prob_dog = float(prediction[0])
        prob_cat = 1 - prob_dog
    else: # Categorical (Multi-class)
        prob_cat = float(prediction[1])
        prob_dog = float(prediction[0])

    return {'Cat 🐱': prob_cat, 'Dog 🐶': prob_dog}


interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload Pet Image"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="🐾 Dog vs. Cat Classifier",
    description=f"""
    Upload an image of a dog or cat, and the AI will classify it!

    **Model Performance:**
    - Architecture: Custom CNN with BatchNormalization
    - Training Dataset: 25,000 images
    """
)

if __name__ == "__main__":
    interface.launch()