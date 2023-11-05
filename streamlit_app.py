# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
import h5py
from tensorflow import keras
from keras.preprocessing import image
import numpy as np
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

crop_labels = {
        0: 'Mango',
        1: 'Guava',
        2: 'Apple',
        3: 'Tomato plant',
        4: 'Cauliflower'
    }
fruit_labels = {
        0: 'Mango',
        1: 'Guava',
        2: 'Apple' 
    }
vegetable_labels = {
        0: 'Tomato plant',
        1: 'Cauliflower'
    }
mango_diseases = {
        0: 'Alternaria',
        1: 'Anthracnose',
        2: 'Black Mould Rot',
        3: 'Healthy Mango',
        4: 'Stem end Rot'
    }
guava_diseases = {
        0: 'Root', 
        1: 'Guava Scab', 
        2: 'Phytopthora'
    }
apple_diseases = {
        0: 'Blotch',
        1: 'Healthy Apple',
        2: 'Apple Scab',
        3: 'Rot'
    }
tomato_diseases = {
        0: 'Bacterial spot',
        1: 'Early blight',
        2: 'Healthy Tomato',
        3: 'Late blight',
        4: 'Leaf mold',
        5: 'Leaf spot',
        6: 'Spider mites'
    }
cauliflower_diseases = {
        0: 'Healthy Cauliflower',
        1: 'Downy Mildew',
        2: 'Black Rot',
        3: 'Bacterial Soft Rot'
    }
image_dims = {
        'Mango': (82, 82),
        'Guava': (256, 256),
        'Apple': (256, 256),
        'Tomato plant': (256, 256),
        'Cauliflower': (300, 300)
    }

disease_files = {
    'Alternaria': 'disease_descriptions/Alternaria.txt',
    'Anthracnose': 'disease_descriptions/Anthracnose.txt',
    'Black Mould Rot': 'disease_descriptions/Black Mould Rot.txt',
    'Stem end Rot': 'disease_descriptions/Stem End Rot.txt',
    'Bacterial Soft Rot': 'disease_descriptions/Bacterial Soft Rot.txt',
    'Black Rot': 'disease_descriptions/Black Rot.txt',
    'Downy Mildew': 'disease_descriptions/Downy Mildew.txt',
    'Root': 'disease_descriptions/Root Rot.txt',
    'Guava Scab': 'disease_descriptions/Scab.txt',
    'Phytopthora': 'disease_descriptions/Phytophthora.txt',
    'Bacterial spot': 'disease_descriptions/Bacterial Spot.txt',
    'Early blight': 'disease_descriptions/Early Blight.txt',
    'Late blight': 'disease_descriptions/Late Blight.txt',
    'Leaf mold': 'disease_descriptions/Leaf Mold.txt',
    'Leaf spot': 'disease_descriptions/Leaf Spot.txt',
    'Spider mites': 'disease_descriptions/Spider Mites.txt',
    'Blotch': 'disease_descriptions/Apple Blotch.txt',
    'Rot': 'disease_descriptions/Apple Rot.txt',
    'Apple Scab': 'disease_descriptions/Apple Scab.txt',
    'Healthy Apple': 'disease_descriptions/Healthy Apple.txt',
    'Healthy Mango': 'disease_descriptions/Healthy Mango.txt',
    'Healthy Tomato': 'disease_descriptions/Healthy Tomato.txt',
    'Healthy Cauliflower': 'disease_descriptions/Healthy Cauliflower.txt'

    }


# Load crop detection models
mango_model = keras.models.load_model('mango_disease_model81.h5')
guava_model = keras.models.load_model('guava_disease_model81.h5')
apple_model = keras.models.load_model('apple_disease_model85.h5')
tomato_model = keras.models.load_model('tomatoplant_disease_model90.h5')
cauliflower_model = keras.models.load_model('cauliflower_disease_model85.h5')
    
crop_models = {'Mango': mango_model, 'Guava': guava_model, 'Apple': apple_model, 'Tomato plant': tomato_model, 'Cauliflower': cauliflower_model}
disease_labels = {'Mango': mango_diseases, 'Guava': guava_diseases, 'Apple': apple_diseases, 'Tomato plant': tomato_diseases, 'Cauliflower': cauliflower_diseases}



def predict_image(image_path, model, disease_label, image_dim):
    img = image.load_img(image_path, target_size=image_dim)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale to match the training data

    # Make the prediction
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    

    # Define class labels based on your mapping
    

    predicted_class = disease_label[class_index]
    confidence = np.max(prediction)
    return predicted_class, confidence

def run():
    st.set_page_config(
        page_title="Crop Guardian",
        page_icon="🍅 ",
    )

    st.header("Welcome to Crop Guardian.")

    selected_crop_type = st.selectbox("Select a crop type:", ['Fruits', 'Vegetables'])
    
    if selected_crop_type == 'Fruits':
        crop_options = list(fruit_labels.values())
    else:
        crop_options = list(vegetable_labels.values())

    selected_crop = st.selectbox("Select a crop to analyze for possible disease:", crop_options)
    selected_crop_model = crop_models[selected_crop]
    selected_disease_label = disease_labels[selected_crop]
    selected_image_dim = image_dims[selected_crop]
    predicted_class = ""
    confidence = 0.0
    
   
    uploaded_image = st.file_uploader("Next, upload an image of your chosen crop.", type=["jpg", "png", "jpeg"])
    run_model_button = st.button("Run Model")
    predicted_class_container = st.empty()
    predicted_confidence_container = st.empty()
    description_container = st.empty()
    image_container = st.empty()
    
# Save and display the uploaded image, if possible
    if uploaded_image is not None and run_model_button:
        image_container.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        predicted_class, confidence = predict_image(uploaded_image, selected_crop_model, selected_disease_label, selected_image_dim)
        predicted_class_container.write(f"Predicted Class: {predicted_class}")
        predicted_confidence_container.write(f"Confidence: {confidence:.2f}")
        with open(disease_files[predicted_class], "r") as file:
            file_contents = file.read()
        description_container.write(file_contents, language="text")


# Display the contents of the text file with syntax highlighting
            
    
# Load crop detection models
    mango_model = keras.models.load_model('mango_disease_model81.h5')
    guava_model = keras.models.load_model('guava_disease_model81.h5')
    apple_model = keras.models.load_model('apple_disease_model85.h5')
    tomato_model = keras.models.load_model('tomatoplant_disease_model90.h5')
    cauliflower_model = keras.models.load_model('cauliflower_disease_model85.h5')
    
    
    st.markdown(
        """
       
    """
    )


if __name__ == "__main__":
    run()
