import streamlit as st
import joblib
import pandas as pd
import base64

# Function to encode local image to base64 for CSS background
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(bin_file):
    try:
        bin_str = get_base64_of_bin_file(bin_file)
        page_bg_img = f'''
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{bin_str}");
            background-size: cover;
            background-attachment: fixed;
        }}
        .stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp label, .stApp span, .stApp div {{
            color: white !important;
        }}
        .main-container {{
            background-color: rgba(0, 0, 0, 0.7);
            padding: 30px;
            border-radius: 15px;
        }}
        .header-box {{
            background-color: rgba(255, 255, 255, 0.15);
            border: 2px solid white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 25px;
            text-align: center;
        }}
        input, .stNumberInput div div input {{
            color: black !important;
        }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except:
        st.warning("Background image not found. Proceeding with default background.")

set_png_as_page_bg('/content/vibrant-purple-iris-flower.jpg')

st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Load model and define explicit mapping
model = joblib.load("rf_pipeline.pkl")
species_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

st.title("🌸 Iris Flower Classifier")

st.markdown("""
<div class="header-box">
    <h2>Intelligence Prediction Hub</h2>
    <p>Capable of identifying: Setosa, Versicolor, and Virginica.</p>
</div>
""", unsafe_allow_html=True)

st.subheader("📏 Input Measurements")
col1, col2 = st.columns(2)
with col1:
    sl = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1)
    sw = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5)
with col2:
    pl = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)
    pw = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)

if st.button("Predict Species"):
    feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    query = pd.DataFrame([[sl, sw, pl, pw]], columns=feature_names)

    prediction_idx = int(model.predict(query)[0])
    result = species_map.get(prediction_idx, "Unknown")

    st.markdown("--- ")
    st.success(f"### Results: {result}")
    st.write(f"**Label Mapping Detected:** {species_map}")
    st.info(f"Current Predicted Class Index: {prediction_idx}")

st.markdown('</div>', unsafe_allow_html=True)
