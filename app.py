import streamlit as st
import folium
from streamlit_folium import folium_static
import requests
import numpy as np
import cv2
import rasterio
import tensorflow as tf
import segmentation_models as sm
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from io import BytesIO
import pandas as pd
import os
import gdown


# ---- CONFIGURATION ----
st.set_page_config(page_title="Solar Energy & Rooftop Segmentation", layout="wide")
MAPBOX_ACCESS_TOKEN = "pk.eyJ1Ijoic2h5YW1rcmlzaG5hcCIsImEiOiJjbTdkMHgxemQwbDRzMmpzN2Y1c3NzdXhiIn0.kbMoaQQnvJzKR-T44wJvug"

BACKBONE = 'resnet34'
IMG_SIZE = (256, 256)
NUM_CLASSES = 18
LR = 0.0001

# ---- LOAD MODEL ----
sm.set_framework('tf.keras')
preprocess_input = sm.get_preprocessing(BACKBONE)
model = sm.Unet(BACKBONE, classes=NUM_CLASSES, activation='softmax', encoder_weights='imagenet')
weights_path =  "https://drive.google.com/file/d/1-6QNahgN4MrXWzHwiPtIU_CfZ_2iot2b/view?usp=drive_link"

model.load_weights(weights_path)
model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss='categorical_crossentropy', metrics=[sm.metrics.iou_score])

# ---- SOLAR LOCATION PLANNING ----
st.title("Solar Location Planning & Rooftop Segmentation")

st.subheader("India's Solar Potential Heatmap")
st.image("https://drive.google.com/file/d/1ufjmjvKGIjF1qRwXcNfo9kRGLmEdEO-t/view?usp=drive_link", caption="Solar Energy Potential Across India", use_container_width=True)

# Fetch solar irradiance data
# NASA POWER API function
def get_daily_solar_irradiance(lat, lon, year=2023):
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": f"{year}0101",
        "end": f"{year}1231",
        "format": "JSON"
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        irradiance_values = data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
        df = pd.DataFrame(irradiance_values.items(), columns=["Date", "Solar_Irradiance (kWh/m²/day)"])
        avg_daily_irradiance = df["Solar_Irradiance (kWh/m²/day)"].mean()
        annual_irradiance = avg_daily_irradiance * 365
        return avg_daily_irradiance, annual_irradiance
    else:
        return None, None

# Step 2: Select a City for Analysis

st.subheader("Choose a City for Analysis")
cities = {"Jaipur": (26.9124, 75.7873), "Ahmedabad": (23.0225, 72.5714), "Chennai": (13.0827, 80.2707)}

st.markdown("""
    <style>
    /* Change text color to white */
    label[data-testid="stWidgetLabel"] {
        color: white !important;
    }
    
    /* Change background color to purple and text color to white */
    div[data-testid="stSelectbox"] {
        background-color: purple !important;
        border-radius: 5px;
        padding: 5px;
    }
    
    /* Ensure dropdown items also have a white text color */
    div[data-testid="stSelectbox"] * {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

selected_city = st.selectbox("Select a City", ["Jaipur", "Ahmedabad", "Chennai"])



lat, lon = cities[selected_city]

st.write(f"**Selected Coordinates:** {lat}, {lon}")

# Display location on map
m = folium.Map(location=[lat, lon], zoom_start=12)
folium.Marker([lat, lon], tooltip=selected_city).add_to(m)
folium_static(m)

# Step 3: Fetch Solar Irradiance Data
st.subheader("Solar Irradiance Data")
avg_daily, annual_irradiance = get_daily_solar_irradiance(lat, lon)

if avg_daily is not None:
    st.markdown(f"""
        <div style="
            border: 3px solid black; /* Black border */
            border-radius: 10px;
            padding: 15px;
            background-color: #FFF3F3; /* Light red background */
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        ">
            <p style="color: black; font-size: 18px; font-weight: bold; text-align: center;">
                🌞 <b>Average Daily Solar Irradiance:</b> {avg_daily:.2f} kWh/m²/day
            </p>
            <p style="color: black; font-size: 18px; font-weight: bold; text-align: center;">
                ⚡ <b>Annual Solar Irradiance:</b> {annual_irradiance:.2f} kWh/m²/year
            </p>
        </div>
    """, unsafe_allow_html=True)
else:
    st.error("Error fetching solar data")


# ---- SOLAR PANEL INSTALLATION ----
st.markdown("""
    <style>
    /* Change background color of input boxes to purple and text color to white */
    div[data-baseweb="input"] {
        background-color: purple !important;
        color: white !important;
        border-radius: 5px;
        padding: 5px;
    }

    /* Ensure input text is white */
    input {
        color: white !important;
    }

    /* Change displayed numbers (inside input boxes and slider) to purple */
    div[data-testid="stNumberInput"] input {
        color: white !important;
        font-weight: bold;
    }

    /* Change displayed text inside slider to purple */
    div[data-testid="stSlider"] span {
        color: purple !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Rooftop Segmentation & Energy Potential")
st.write("Select a location to analyze rooftop solar potential.")

lat = st.number_input("Enter Latitude", min_value=-90.0, max_value=90.0, value=37.74, step=0.0001)
lon = st.number_input("Enter Longitude", min_value=-180.0, max_value=180.0, value=-122.4194, step=0.0001)
zoom = st.slider("Zoom Level", min_value=15, max_value=20, value=18)



st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

if st.button("Fetch Satellite Images & Analyze"):
    img_url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon},{lat},{zoom}/256x256@2x?access_token={MAPBOX_ACCESS_TOKEN}&format=tiff"
    response = requests.get(img_url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img_array = np.array(img)
        
        # Display original image
        st.image(img_array, caption="Satellite Image", use_container_width=True)
        
        # Predict segmentation mask
        img_resized = cv2.resize(img_array, IMG_SIZE).astype(np.float32) / 255.0
        img_input = np.expand_dims(img_resized, axis=0)
        pred = model.predict(img_input)
        pred_mask = np.argmax(pred, axis=-1)[0]
        pred_mask = (pred_mask * 255 / pred_mask.max()).astype(np.uint8) if pred_mask.max() > 0 else pred_mask
        
        # Display segmented mask
        st.image(pred_mask, caption="Segmented Rooftop Mask", use_container_width=True, clamp=True)
        
        # Business Insights
        
        # Calculate energy potential and carbon savings
        total_area = np.count_nonzero(pred_mask) * (1 / (IMG_SIZE[0] * IMG_SIZE[1]))  # Normalize area
        energy_potential = total_area * avg_daily * 365  # kWh/year estimate
        carbon_savings = energy_potential * 0.85  # Approximate conversion factor
        energy_cost_per_kWh = 0.12  # Example cost in $ per kWh (adjust based on fuel source)
        annual_savings = energy_potential * energy_cost_per_kWh

        # Display results in a styled box
        st.markdown(f"""
            <div style="
                border: 4px solid #4A90E2; 
                border-radius: 15px; 
                padding: 20px;
                background-color: #F3F6FB;
                text-align: center;
                box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.1);
            ">
                <h2 style="color: #4A90E2;">🌍 Key Performance Indicators (KPIs)</h2>
                <p style="font-size: 18px; color: black; font-weight: bold;">
                    ⚡ <b>Estimated Energy Production:</b> {energy_potential:.2f} kWh/year
                </p>
                <p style="font-size: 18px; color: black; font-weight: bold;">
                    🍃 <b>Carbon Reduction:</b> {carbon_savings:.2f} kg CO₂/year
                </p>
                <p style="font-size: 18px; color: black; font-weight: bold;">
                    💰 <b>Annual Savings:</b> ${annual_savings:.2f}/year
                </p>
            </div>
        """, unsafe_allow_html=True)

    else:
        st.error("Failed to fetch satellite image")
# ---- CUSTOM STYLING ----
st.markdown("""
    <style>
        .stApp {
            background-color: #f3e8ff; /* Light Purple Background */
            color: black; /* Sets text color to black */
        }
        h1, h2, h3, h4, h5, h6, p, div, span {
            color: black !important; /* Ensures text remains black */
        }
    </style>
""", unsafe_allow_html=True)


