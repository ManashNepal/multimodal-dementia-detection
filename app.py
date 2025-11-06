import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import nibabel as nib
from scipy.ndimage import zoom
import joblib
import os

# --- SET PAGE CONFIG ---
st.set_page_config(page_title="Dementia Detection", page_icon= ":brain:" , layout="wide")

# --- DEFINE FILE PATHS ---
MODEL_PATH = os.path.join('models', 'dementia_model.h5')
SCALER_PATH = os.path.join('models', 'scaler.pkl')
CONFUSION_MATRIX_PATH = os.path.join('models', 'confusion_matrix.jpeg') 

# --- LOAD MODEL AND SCALER ---
@st.cache_resource
def load_model():
    """Loads the saved Keras .h5 model."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model from {MODEL_PATH}: {e}")
        return None

@st.cache_resource
def load_scaler():
    """Loads the saved Joblib scaler."""
    try:
        scaler = joblib.load(SCALER_PATH)
        return scaler
    except Exception as e:
        st.error(f"Error loading scaler from {SCALER_PATH}: {e}")
        return None

model = load_model()
scaler = load_scaler()

# --- IMAGE PREPROCESSING FUNCTION ---
IMG_SIZE = 128
def process_scan(path):
    """Loads a 3D MRI scan, resizes, and normalizes it."""
    try:
        img = nib.load(path)
        img_data = img.get_fdata()
        
        original_shape = img_data.shape
        zoom_factors = (
            IMG_SIZE / original_shape[0],
            IMG_SIZE / original_shape[1],
            IMG_SIZE / original_shape[2]
        )
        
        img_resized = zoom(img_data, zoom_factors, order=1, prefilter=False)
        img_normalized = (img_resized - np.min(img_resized)) / (np.max(img_resized) - np.min(img_resized))
        img_final = img_normalized[..., np.newaxis]
        
        return img_final.astype(np.float32)
    except Exception as e:
        st.error(f"Error processing MRI scan: {e}")
        return None

# --- STREAMLIT APP LAYOUT ---

st.title(":brain: Multimodal Dementia Detection")
st.write("This app predicts the stage of dementia using both clinical data and a 3D MRI scan.")

# --- Sidebar for Inputs ---
st.sidebar.title("Patient Information")
st.sidebar.write("Please provide the patient's clinical data and MRI scan.")

age = st.sidebar.slider("Age", 18, 100, 70)
sex_option = st.sidebar.selectbox("Sex", ["Male", "Female"])
educ = st.sidebar.number_input("Education Level (1-5)", 1, 5, 3)
ses = st.sidebar.number_input("Socioeconomic Status (1-5)", 1, 5, 2)
mmse = st.sidebar.slider("MMSE Score", 0, 30, 28)

uploaded_file = st.sidebar.file_uploader("Upload MRI Scan (.nii file)", type=["nii"])
st.sidebar.subheader("Clinical Dementia Rating (CDR) Scale")
st.sidebar.markdown("""
This model is trained to predict the CDR stage, which is the "answer" from the dataset:
* **CDR = 0.0**: Non-Demented (0)
* **CDR = 0.5**: Very Mild Dementia (1)
* **CDR = 1.0**: Mild Dementia (2)
* **CDR = 2.0**: Moderate Dementia (3)
""")


# --- Main Page for Outputs ---
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Patient Inputs")
    st.markdown(f"**Age:** {age}")
    st.markdown(f"**Sex:** {sex_option}")
    st.markdown(f"**Education:** {educ}")
    st.markdown(f"**SES:** {ses}")
    st.markdown(f"**MMSE:** {mmse}")
    
    if uploaded_file:
        st.markdown(f"**MRI Scan:** `{uploaded_file.name}`")
    else:
        st.markdown("**MRI Scan:** `No file uploaded`")

with col2:
    st.subheader("Prediction")

    predict_button = st.button("Run Prediction")

    if predict_button:
        if uploaded_file is None or model is None or scaler is None:
            st.warning("Please upload an MRI scan and ensure model/scaler are loaded.")
        else:
            with st.spinner("Analyzing patient data... This may take a moment."):
                try:
                    # Process Tabular Data
                    sex = 0 if sex_option == "Male" else 1
                    tabular_data_df = pd.DataFrame([[age, educ, ses, mmse]], columns=['Age', 'Educ', 'SES', 'MMSE'])
                    tabular_data_scaled = scaler.transform(tabular_data_df)
                    
                    tabular_input = np.array([[
                        tabular_data_scaled[0, 0], # Scaled Age
                        sex,                       # Encoded Sex
                        tabular_data_scaled[0, 1], # Scaled Educ
                        tabular_data_scaled[0, 2], # Scaled SES
                        tabular_data_scaled[0, 3]  # Scaled MMSE
                    ]], dtype=np.float32)

                    # Process Image Data
                    temp_file_path = "temp_scan.nii"
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    image_input = process_scan(temp_file_path)
                    image_input_batch = np.expand_dims(image_input, axis=0)
                    os.remove(temp_file_path)

                    # Run Prediction
                    inputs = {
                        'image_input': image_input_batch,
                        'tabular_input': tabular_input
                    }
                    preds_prob = model.predict(inputs)[0]
                    
                    prediction_index = np.argmax(preds_prob)
                    class_names = ['Non-Demented (0)', 'Very Mild (1)', 'Mild (2)', 'Moderate (3)']
                    prediction_class = class_names[prediction_index]
                    confidence = preds_prob[prediction_index]

                    st.success(f"**Prediction:** {prediction_class}")
                    st.metric(label="Confidence", value=f"{confidence*100:.2f}%")

                    # Display Probabilities
                    st.write("**Prediction Probabilities:**")
                    prob_df = pd.DataFrame(preds_prob, index=class_names, columns=["Probability"])
                    st.bar_chart(prob_df)

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

# --- About Section ---
with st.expander(":information_source: About This Model & Performance"):
    st.write("""
    This model is a multimodal 3D-CNN and Feed-Forward Network (FNN) trained on the OASIS-1 dataset.
    - The **3D-CNN** analyzes structural patterns from the 3D MRI scan.
    - The **FNN** analyzes clinical features (Age, Sex, MMSE, etc.).
    - Both feature streams are combined to make a final prediction.
    """)
    st.write("It was trained on the OASIS-1 dataset and achieved the following performance on the unseen test set:")
    
    st.metric("Final Test Accuracy", "77.27%")
    
    try:
        st.image(CONFUSION_MATRIX_PATH, caption="Model Confusion Matrix")
    except:
        st.warning(f"Confusion Matrix image not found at {CONFUSION_MATRIX_PATH}")
    st.write("""
    **Analysis:** The model is highly effective at identifying 'Non-Demented' patients but struggles with the 'Very Mild' dementia stage.
    """)