import os
# This line must be before the 'import tensorflow' line to take effect.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# --- 2. Page Configuration ---
# Set the page configuration. This must be the first Streamlit command.
st.set_page_config(
    page_title="Disaster Prediction",
    page_icon="🌪️",
    layout="centered"
)


# --- 3. Model Loading ---
# Use st.cache_resource to load the model only once and store it in cache.
# This prevents the model from being reloaded every time a user interacts with the app.
@st.cache_resource
def load_keras_model():
    """
    Loads the pre-trained Keras model from a file.
    The @st.cache_resource decorator ensures the model is loaded only once.
    """
    try:
        # Replace with the actual path to your model if it's not in the same directory
        model = load_model("best_combined_model.keras")
        print("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        print(f"Error loading model: {e}")
        return None


# --- 4. Image Preprocessing Function ---
def preprocess_image(image):
    """
    Preprocesses the uploaded image to match the model's input requirements.
    - Resizes the image to the required dimensions.
    - Converts it to a NumPy array.
    - Handles transparency channels (alpha).
    - Normalizes pixel values.
    - Adds a batch dimension.
    """
    # CRITICAL: These dimensions must match the 'img_height' and 'img_width' from your training script.
    img_height = 224
    img_width = 224

    # Resize the image using PIL
    image = image.resize((img_height, img_width))

    # Convert image to a numpy array
    image_array = np.array(image)

    # If the image has 4 channels (like PNG with transparency), remove the alpha channel
    if image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]

    # Normalize pixel values from [0, 255] to [0.0, 1.0], if your model was trained this way.
    image_array = image_array / 255.0

    # The model expects a batch of images as input.
    # Expand dimensions from (height, width, channels) to (1, height, width, channels)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# --- 5. Main Application UI ---

# Load the model using the cached function
model = load_keras_model()

# Set the title of the Streamlit app
st.title("Disaster Type Prediction 🌪️🌊🔥")

# Add some introductory text
st.write(
    "Upload an image of a natural disaster, and the AI will try to predict its type. "
    "This model is a combination of a custom CNN and MobileNetV2 for feature extraction."
)

# Create a file uploader widget that accepts common image formats
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

# Main logic: Proceed only if the model is loaded and a file has been uploaded
if model is not None and uploaded_file is not None:
    # Open the uploaded file as an image
    image = Image.open(uploaded_file)
    
    # Display the uploaded image on the page
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Add a button to trigger the prediction
    if st.button("Predict Disaster Type"):
        # Show a spinner while the model is making a prediction
        with st.spinner("Analyzing the image..."):
            
            # Preprocess the image to prepare it for the model
            processed_image = preprocess_image(image)

            # Make a prediction
            prediction = model.predict(processed_image)

            # Get the index of the class with the highest probability
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            
            # Get the confidence score (the highest probability)
            confidence = np.max(prediction) * 100

            # !IMPORTANT!: You MUST replace these with your actual class names
            # The order must match the output of your model's softmax layer.
            # Example: If your training data folders were 00-Cyclone, 01-Earthquake, etc.
            class_names = ['Cyclone', 'Earthquake', 'Flood', 'Wildfire']

            # Check if the predicted index is valid for your class_names list
            if predicted_class_index < len(class_names):
                predicted_class_name = class_names[predicted_class_index]

                # Display the prediction result in a green box (success)
                st.success(f"**Prediction:** {predicted_class_name}")
                
                # Display the confidence score in a blue box (info)
                st.info(f"**Confidence:** {confidence:.2f}%")
            else:
                st.error("Prediction index is out of range. Please check your model's output classes and the `class_names` list in the code.")

# Handle cases where the model or file is not ready
else:
    if model is None:
        st.error("The model could not be loaded. Please ensure 'best_combined_model.keras' is in the correct directory.")
    else:
        st.info("Please upload an image to get a prediction.")

# --- 6. Footer ---
st.markdown("---")
st.markdown("Developed with Streamlit and TensorFlow.")