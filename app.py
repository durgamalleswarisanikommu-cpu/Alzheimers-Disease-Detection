import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Alzheimer's MRI AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f5f9ff 0%, #eef4fb 100%);
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 20px 10px 10px 10px;
    }

    .main-header h1 {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .main-header p {
        font-size: 18px;
        color: #5f6b7a;
    }

    /* Cards */
    .card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0px 6px 20px rgba(0,0,0,0.08);
        margin-bottom: 20px;
        border: 1px solid #e6edf5;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 15px;
    }

    /* Prediction */
    .prediction-card {
        background: linear-gradient(135deg, #eef7ff, #ffffff);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        border: 1px solid #dcecff;
        box-shadow: 0px 8px 25px rgba(0,0,0,0.08);
    }

    .prediction-label {
        font-size: 17px;
        color: #667085;
        margin-bottom: 8px;
    }

    .prediction-result {
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .confidence {
        font-size: 20px;
        font-weight: 600;
    }

    /* Info boxes */
    .info-box {
        background: #f8fafc;
        padding: 18px;
        border-radius: 14px;
        border-left: 5px solid #4f8cff;
        margin-bottom: 12px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #7a8699;
        padding: 25px;
        font-size: 14px;
    }

    /* Hide Streamlit menu */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("alzheimer_model_oldd.keras")


try:
    model = load_model()
except Exception as e:
    st.error("❌ Unable to load the trained model.")
    st.exception(e)
    st.stop()


# =========================================================
# CLASS INFORMATION
# =========================================================

CLASS_NAMES = [
    "Non-Demented",
    "Very Mild Demented",
    "Mild Demented",
    "Moderate Demented"
]

CLASS_DESCRIPTIONS = {
    "Non-Demented":
        "The model classified the uploaded MRI into the non-demented category.",

    "Very Mild Demented":
        "The model classified the uploaded MRI into the very mild dementia category.",

    "Mild Demented":
        "The model classified the uploaded MRI into the mild dementia category.",

    "Moderate Demented":
        "The model classified the uploaded MRI into the moderate dementia category."
}


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🧠 Alzheimer's AI")

    st.markdown("---")

    st.markdown("### 📌 About")

    st.write(
        """
        This application uses a deep learning model to classify
        Alzheimer's MRI images into four categories.
        """
    )

    st.markdown("---")

    st.markdown("### 🤖 Model")

    st.write("**Architecture:** EfficientNetB0")
    st.write("**Input Size:** 224 × 224")
    st.write("**Classes:** 4")
    st.write("**Framework:** TensorFlow / Keras")

    st.markdown("---")

    st.markdown("### 📂 Supported Files")

    st.write("""
    - JPG
    - JPEG
    - PNG
    """)

    st.markdown("---")

    st.info(
        "⚠️ This application is intended for academic and "
        "research purposes only."
    )


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="main-header">

<h1>🧠 Alzheimer's MRI Classification</h1>

<p>
AI-powered classification of brain MRI images using Deep Learning
</p>

</div>
""", unsafe_allow_html=True)


st.markdown("---")


# =========================================================
# INTRODUCTION
# =========================================================

st.markdown("""
<div class="card">

<div class="section-title">
🔬 AI-Powered MRI Analysis
</div>

<p>
Upload a brain MRI image and the trained EfficientNetB0 model
will analyze the image and predict one of four classification
categories.
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# UPLOAD SECTION
# =========================================================

col1, col2 = st.columns([1.2, 1])

with col1:

    st.markdown("""
    <div class="card">

    <div class="section-title">
    📤 Upload MRI Image
    </div>

    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an MRI image",
        type=["jpg", "jpeg", "png"],
        help="Upload a brain MRI image in JPG, JPEG or PNG format."
    )


# =========================================================
# PROCESS IMAGE
# =========================================================

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        with col1:

            st.image(
                image,
                caption="Uploaded MRI Image",
                use_container_width=True
            )

            st.success("✅ MRI image uploaded successfully")

            st.caption(
                f"File: {uploaded_file.name} | "
                f"Original Size: {image.size[0]} × {image.size[1]}"
            )


        # =================================================
        # PREPROCESSING
        # =================================================

        resized_image = image.resize((224, 224))

        image_array = np.array(resized_image).astype(np.float32)

        image_array = np.expand_dims(image_array, axis=0)


        # =================================================
        # PREDICTION
        # =================================================

        with st.spinner("🔍 AI is analyzing the MRI image..."):

            predictions = model.predict(
                image_array,
                verbose=0
            )

        probabilities = predictions[0]

        predicted_index = int(np.argmax(probabilities))

        predicted_class = CLASS_NAMES[predicted_index]

        confidence = float(probabilities[predicted_index]) * 100


        # =================================================
        # RESULT SECTION
        # =================================================

        with col2:

            st.markdown("""
            <div class="section-title">
            🎯 Prediction Result
            </div>
            """, unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class="prediction-card">

                <div class="prediction-label">
                Predicted Classification
                </div>

                <div class="prediction-result">
                {predicted_class}
                </div>

                <div class="confidence">
                Confidence: {confidence:.2f}%
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            st.markdown("<br>", unsafe_allow_html=True)


            # Interpretation

            st.markdown(
                f"""
                <div class="info-box">

                <b>📋 Interpretation</b>

                <br><br>

                {CLASS_DESCRIPTIONS[predicted_class]}

                </div>
                """,
                unsafe_allow_html=True
            )


            # Confidence indicator

            if confidence >= 80:

                st.success(
                    f"High model confidence: {confidence:.2f}%"
                )

            elif confidence >= 60:

                st.warning(
                    f"Moderate model confidence: {confidence:.2f}%"
                )

            else:

                st.info(
                    f"Low model confidence: {confidence:.2f}%"
                )


        # =================================================
        # PROBABILITY DISTRIBUTION
        # =================================================

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="card">

        <div class="section-title">
        📊 Prediction Probability Distribution
        </div>

        </div>
        """, unsafe_allow_html=True)


        probability_columns = st.columns(4)

        for i, class_name in enumerate(CLASS_NAMES):

            probability = float(probabilities[i]) * 100

            with probability_columns[i]:

                st.metric(
                    label=class_name,
                    value=f"{probability:.2f}%"
                )

                st.progress(
                    min(probability / 100, 1.0)
                )


        # =================================================
        # MODEL INFORMATION
        # =================================================

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="card">

        <div class="section-title">
        🤖 Model Information
        </div>

        </div>
        """, unsafe_allow_html=True)


        info1, info2, info3, info4 = st.columns(4)

        with info1:
            st.metric(
                "Architecture",
                "EfficientNetB0"
            )

        with info2:
            st.metric(
                "Input",
                "224 × 224"
            )

        with info3:
            st.metric(
                "Output Classes",
                "4"
            )

        with info4:
            st.metric(
                "Framework",
                "TensorFlow"
            )


        # =================================================
        # CLASS DETAILS
        # =================================================

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="card">

        <div class="section-title">
        📚 Classification Categories
        </div>

        </div>
        """, unsafe_allow_html=True)


        for class_name in CLASS_NAMES:

            st.markdown(
                f"""
                <div class="info-box">
                <b>{class_name}</b>
                <br>
                {CLASS_DESCRIPTIONS[class_name]}
                </div>
                """,
                unsafe_allow_html=True
            )


    except Exception as e:

        st.error("❌ Error while processing the MRI image.")

        st.exception(e)


else:

    # =====================================================
    # EMPTY STATE
    # =====================================================

    with col2:

        st.markdown("""
        <div class="card">

        <div class="section-title">
        🔎 How It Works
        </div>

        <p>1️⃣ Upload an MRI image</p>

        <p>2️⃣ Image is resized to 224 × 224 pixels</p>

        <p>3️⃣ EfficientNetB0 analyzes the image</p>

        <p>4️⃣ The model calculates probabilities</p>

        <p>5️⃣ The highest probability class is displayed</p>

        </div>
        """, unsafe_allow_html=True)


# =========================================================
# DISCLAIMER
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

st.warning(
    """
    ⚠️ **Important Medical Disclaimer**

    This application is an academic/research prototype.
    The prediction generated by the model should not be considered
    a medical diagnosis and should not replace evaluation by a
    qualified healthcare professional.
    """
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">

🧠 Alzheimer's MRI Classification System
<br>
Built using Python • TensorFlow • EfficientNetB0 • Streamlit

</div>
""", unsafe_allow_html=True)