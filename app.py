
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO


WINDOW_SIZE = 100
STEP_SIZE = 50


st.set_page_config(page_title="Human Activity Recognition", page_icon="🏃", layout="centered")
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("https://images.pexels.com/photos/1552252/pexels-photo-1552252.jpeg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

[data-testid="stHeader"], [data-testid="stToolbar"] {{
    background-color: rgba(0, 0, 0, 0);
}}

h1, h2, h3, h4, h5, h6, p, div, label, span {{
    color: white !important;
    font-weight: 500;
}}

section.main > div {{
    background-color: rgba(0, 0, 0, 0.6);
    padding: 20px;
    border-radius: 12px;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_with_attention.keras")

@st.cache_resource
def load_encoder():
    return joblib.load("label_encoder.pkl")

model = load_model()
label_encoder = load_encoder()


st.title("🏃 Human Activity Recognition")
st.caption("Using Accelerometer Data | WISDM Dataset")
st.markdown(
    """
    Upload a **CSV file** with columns: `x`, `y`, and `z` from your accelerometer.  
    The model will segment your data using a sliding window and classify the activity being performed:
    
    - 🚶 Walking  
    - 🏃 Jogging  
    - 🪑 Sitting  
    - 🧍 Standing  
    - ⬆️ Upstairs  
    - ⬇️ Downstairs
    """
)

st.divider()


uploaded_file = st.file_uploader("📁 Upload your accelerometer CSV file", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        if not {'x', 'y', 'z'}.issubset(df.columns):
            st.error("❌ CSV must contain columns: x, y, z")
            st.stop()

        if len(df) < WINDOW_SIZE:
            st.warning("⚠️ At least 100 rows are required for prediction.")
            st.stop()

        st.success("✅ File loaded successfully!")
        with st.expander("📄 View Uploaded Data"):
            st.dataframe(df.head(10)) 


        
        st.info("🔄 Segmenting Data...")
        segments = []
        indices = []

        for start in range(0, len(df) - WINDOW_SIZE, STEP_SIZE):
            end = start + WINDOW_SIZE
            x, y, z = df['x'].values[start:end], df['y'].values[start:end], df['z'].values[start:end]
            if len(x) == WINDOW_SIZE:
                segments.append(np.array([x, y, z]).T)
                indices.append(start)

        if not segments:
            st.warning("⚠️ Not enough data for segmentation.")
            st.stop()

        X_input = np.array(segments)
        preds = model.predict(X_input)
        pred_classes = np.argmax(preds, axis=1)
        pred_labels = label_encoder.inverse_transform(pred_classes)
        pred_probs = np.max(preds, axis=1)

        
        result_df = pd.DataFrame({
            "Start_Index": indices,
            "Predicted_Activity": pred_labels,
            "Confidence": np.round(pred_probs, 3)
        })
        st.subheader("🧠 Predicted Activities")
        st.dataframe(result_df)

        
        st.subheader("🔎 High Confidence Segments")
        threshold = st.slider("📊 Confidence Threshold", 0.80, 1.0, 0.95, 0.01)
        high_conf_df = result_df[result_df['Confidence'] >= threshold]
        st.dataframe(high_conf_df if not high_conf_df.empty else "No high confidence predictions found.")

        
        st.subheader("📊 Activity Distribution (Filtered)")
        if not high_conf_df.empty:
            st.bar_chart(high_conf_df["Predicted_Activity"].value_counts())
        else:
            st.warning("⚠️ No data above the selected confidence threshold.")

        
        st.subheader("🕒 Activity Timeline (Filtered)")
        if not high_conf_df.empty:
            plt.figure(figsize=(12, 3))
            sns.lineplot(x=high_conf_df["Start_Index"], y=high_conf_df["Predicted_Activity"], marker='o')
            plt.xticks(rotation=45)
            plt.xlabel("Start Index of Window")
            plt.ylabel("Activity")
            plt.grid(True)
            st.pyplot(plt)
        else:
            st.warning("⚠️ No activity timeline to show.")

        
        st.subheader("📥 Download Predictions")
        csv_buffer = BytesIO()
        result_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download as CSV",
            data=csv_buffer.getvalue(),
            file_name="predicted_activities.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.exception(f"❌ Error: {e}")

else:
    st.info("📌 Awaiting CSV file upload...")


st.markdown("---")
st.markdown("👩‍💻 Built with ❤️ by Tiasha")

