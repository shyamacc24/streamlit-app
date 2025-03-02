import streamlit as st
import tensorflow as tf
import segmentation_models as sm
import requests
import rasterio
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from io import BytesIO

# Set Streamlit theme
st.set_page_config(page_title="CNN Segmentation", layout="wide")
st.markdown("""
    <style>
        .main { background-color: #f4e8ff; }
        h1, h2, h3, h4, h5, h6 { color: #5c3c92; }
        .stButton>button { background-color: #5c3c92; color: white; }
    </style>
""", unsafe_allow_html=True)

# Model parameters
BACKBONE = 'resnet34'
IMG_SIZE = (256, 256)
LR = 0.0001
NUM_CLASSES = 18

# Load the model
sm.set_framework('tf.keras')
preprocess_input = sm.get_preprocessing(BACKBONE)
model = sm.Unet(BACKBONE, classes=NUM_CLASSES, activation='softmax', encoder_weights='imagenet')
weights_path = "/content/drive/MyDrive/SegmentationData/resnet_unet_multiclass_model_4.keras"
model.load_weights(weights_path)
model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss='categorical_crossentropy', metrics=[sm.metrics.iou_score])

# Mapbox API Key
MAPBOX_ACCESS_TOKEN = "pk.eyJ1Ijoic2h5YW1rcmlzaG5hcCIsImEiOiJjbTdkMHgxemQwbDRzMmpzN2Y1c3NzdXhiIn0.kbMoaQQnvJzKR-T44wJvug"

# Sidebar options
st.title("🌍 Satellite Image Segmentation with CNN")
st.sidebar.header("🔧 Model Settings")
ZOOM = st.sidebar.slider("Zoom Level", min_value=15, max_value=20, value=18)
lat = st.sidebar.number_input("Latitude", value=12.9237)
lon = st.sidebar.number_input("Longitude", value=77.6739)
radius = st.sidebar.slider("Radius (km)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

# Function to generate a grid of coordinates
def generate_coordinates(lat, lon, radius_km):
    TILE_SIZE = 156543.03 / (2 ** ZOOM)
    SHIFT = (radius_km * 1000) / 111320
    lat_shifts = [-SHIFT, 0, SHIFT]
    lon_shifts = [-SHIFT, 0, SHIFT]
    return [(lat + lat_shift, lon + lon_shift) for lat_shift in lat_shifts for lon_shift in lon_shifts]

if st.sidebar.button("Download Images & Predict"):
    coordinates = generate_coordinates(lat, lon, radius)
    images, masks, potentials = [], [], []
    
    with st.spinner("Processing images..."):
        for i, (lat_, lon_) in enumerate(coordinates):
            url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon_},{lat_},{ZOOM}/256x256@2x?access_token={MAPBOX_ACCESS_TOKEN}&format=tiff"
            response = requests.get(url)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img_array = np.array(img.convert("RGB"))
                img_resized = cv2.resize(img_array, IMG_SIZE).astype(np.float32) / 255.0
                img_input = np.expand_dims(img_resized, axis=0)
                pred = model.predict(img_input)
                pred_mask = np.argmax(pred, axis=-1)[0]
                pred_mask = (pred_mask * 255 / pred_mask.max()).astype(np.uint8) if pred_mask.max() > 0 else pred_mask
                potential = np.random.uniform(100, 500)  # Placeholder potential calculation
                images.append(img)
                masks.append(pred_mask)
                potentials.append(potential)
            else:
                st.error(f"Failed to download image {i+1}")
    
    # Display results
    cols = st.columns(3)
    for i in range(len(images)):
        with cols[i % 3]:
            st.image(images[i], caption=f"Original Image {i+1}", use_container_width=True)
            st.image(masks[i], caption=f"Predicted Mask {i+1}", use_container_width=True, clamp=True)
            st.metric("Estimated Potential (kWh)", round(potentials[i], 2))
    
    # Create mosaic
    mosaic_img = np.vstack([np.hstack(images[:3]), np.hstack(images[3:])])
    mosaic_mask = np.vstack([np.hstack(masks[:3]), np.hstack(masks[3:])])
    st.subheader("📍 Mosaic View of Area")
    st.image(mosaic_img, caption="Mosaic of Original Images", use_container_width=True)
    st.image(mosaic_mask, caption="Mosaic of Predicted Masks", use_container_width=True)
    
    # Heatmap
    st.subheader("🔥 Potential Energy Heatmap")
    heatmap_data = np.array(potentials).reshape(3, 3)
    fig, ax = plt.subplots()
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="Purples", ax=ax)
    st.pyplot(fig)
