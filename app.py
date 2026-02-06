import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image


st.title(" AI Image Recognizer")
st.markdown("""
The app uses the ResNet50 model to identify objects in images.
Upload an image to see what the AI ​​thinks it is!
""")

@st.cache_resource
def load_model():
    return ResNet50(weights='imagenet')

model = load_model()

uploaded_file = st.file_uploader("Drag and drop an image here (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    #Open the image and fix orientation automatically
    img = Image.open(uploaded_file)
    img = ImageOps.exif_transpose(img)
    
    # Display the uploaded image in the app
    st.image(img, caption='Uploaded image', use_column_width=True)
    
    with st.spinner('Analyzing the image..'):
        # Pre-processing
        # Resize to 224x224 as required by ResNet50
        img_resized = img.resize((224, 224))
        x = image.img_to_array(img_resized)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        #Model Prediction
        preds = model.predict(x)
        results = decode_predictions(preds, top=3)[0]
        
        #Display the results
        st.subheader("Results:")
        for i, (id, label, prob) in enumerate(results):
            clean_label = label.replace('_', ' ').capitalize()
            st.write(f"**{clean_label}**")
            st.progress(float(prob))