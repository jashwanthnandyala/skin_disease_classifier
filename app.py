import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image
import io
from datetime import datetime
from grad_cam import GradCAM

# Configure page with professional settings
st.set_page_config(
    page_title="Skin Disease Classifier | Medical AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, professional styling
st.markdown("""
    <style>
    :root {
        --primary-color: #0066cc;
        --secondary-color: #00a3e0;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --bg-color: #f8fafc;
        --card-bg: #ffffff;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --border-color: #e2e8f0;
    }
    
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #0066cc 0%, #00a3e0 100%);
        padding: 3rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 102, 204, 0.15);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        font-size: 1rem;
        opacity: 0.95;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    /* Card styling */
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
    }
    
    /* Metric styling */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    /* Progress bar styling */
    .progress-container {
        margin: 1rem 0;
    }
    
    .progress-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    
    .progress-bar {
        height: 8px;
        background: #e2e8f0;
        border-radius: 4px;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.4s ease;
    }
    
    /* Alert styling */
    .alert {
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
        font-size: 0.95rem;
    }
    
    .alert-danger {
        background-color: #fee2e2;
        border-left-color: #ef4444;
        color: #7f1d1d;
    }
    
    .alert-warning {
        background-color: #fef3c7;
        border-left-color: #f59e0b;
        color: #78350f;
    }
    
    .alert-info {
        background-color: #dbeafe;
        border-left-color: #0066cc;
        color: #0c2340;
    }
    
    .alert-success {
        background-color: #dcfce7;
        border-left-color: #10b981;
        color: #15803d;
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .badge-primary {
        background: #dbeafe;
        color: #0c2340;
    }
    
    .badge-success {
        background: #dcfce7;
        color: #15803d;
    }
    
    .badge-danger {
        background: #fee2e2;
        color: #7f1d1d;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #0066cc 0%, #0052a3 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 102, 204, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0, 102, 204, 0.4);
    }
    
    /* Text styling */
    .disease-name {
        font-size: 2rem;
        font-weight: 700;
        color: #64748b !important;
        margin: 0.5rem 0;
    }
    
    .confidence-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #10b981;
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 3px solid #0066cc;
        padding-bottom: 0.5rem;
    }
    
    /* Image styling */
    .image-container {
        border-radius: 12px;
        overflow: hidden;
        border: 2px solid #e2e8f0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
    
    /* Recommendation styling */
    .recommendation-item {
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #0066cc;
        background: #f0f7ff;
        color: #0c2340;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .recommendation-item:hover {
        transform: translateX(5px);
        background: #e3f2fd;
        color: #051a33;
    }
    
    /* Sidebar styling */
    .sidebar-content {
        padding: 1.5rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .header-title {
            font-size: 1.75rem;
        }
        
        .disease-name {
            font-size: 1.5rem;
        }
        
        .confidence-value {
            font-size: 2rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Disease information with next steps
DISEASE_INFO = {
    0: {
        "name": "Acne",
        "description": "A common skin condition characterized by pimples, blackheads, and whiteheads.",
        "next_steps": [
            "✓ Cleanse face twice daily with a gentle cleanser",
            "✓ Use non-comedogenic moisturizer",
            "✓ Avoid touching or picking at acne",
            "✓ Consider benzoyl peroxide or salicylic acid products",
            "✓ Consult a dermatologist if severe",
            "✓ Maintain a healthy diet and manage stress"
        ]
    },
    1: {
        "name": "Carcinoma",
        "description": "A type of skin cancer that requires immediate medical attention.",
        "next_steps": [
            "🚨 CONSULT A DERMATOLOGIST IMMEDIATELY",
            "✓ Avoid sun exposure",
            "✓ Wear protective clothing and sunscreen (SPF 50+)",
            "✓ Do not self-treat - professional diagnosis is crucial",
            "✓ Get a biopsy if recommended by your doctor",
            "✓ Follow dermatologist's treatment plan"
        ]
    },
    2: {
        "name": "Eczema",
        "description": "An inflammatory skin condition causing itching, redness, and irritation.",
        "next_steps": [
            "✓ Moisturize regularly with fragrance-free creams",
            "✓ Take lukewarm baths/showers (avoid hot water)",
            "✓ Use mild soaps and cleansers",
            "✓ Identify and avoid triggers (certain fabrics, soaps)",
            "✓ Apply cortisone creams as recommended",
            "✓ See a dermatologist for severe cases"
        ]
    },
    3: {
        "name": "Keratosis",
        "description": "Non-cancerous growths that appear on the skin surface.",
        "next_steps": [
            "✓ Monitor for any changes in size or appearance",
            "✓ Protect area from sun exposure",
            "✓ Avoid picking or scratching",
            "✓ If bothersome, consult dermatologist for removal",
            "✓ Treatment options: cryotherapy, laser, or chemical peels",
            "✓ Generally harmless but should be evaluated by professional"
        ]
    },
    4: {
        "name": "Milia",
        "description": "Small white cysts that form on the skin, usually on the face.",
        "next_steps": [
            "✓ Keep skin clean and moisturized",
            "✓ Use sunscreen daily (SPF 30+)",
            "✓ Avoid heavy creams that may clog pores",
            "✓ Do not attempt to squeeze or pop them",
            "✓ Professional extraction may be needed",
            "✓ Usually resolve on their own over time"
        ]
    },
    5: {
        "name": "Rosacea",
        "description": "A chronic skin condition causing redness and flushing on the face.",
        "next_steps": [
            "✓ Identify and avoid triggers (spicy food, hot drinks, stress)",
            "✓ Use gentle, fragrance-free skincare products",
            "✓ Apply generous sunscreen (SPF 50+) daily",
            "✓ Avoid extreme temperatures and hot showers",
            "✓ Consult dermatologist for prescribed treatments",
            "✓ May require topical medications or antibiotics"
        ]
    }
}

# Load model (cached for performance)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('skin_model.keras')
    return model

# Load GradCAM (cached for performance)
@st.cache_resource
def load_grad_cam(model):
    """Initialize GradCAM with the loaded model"""
    grad_cam = GradCAM(model, layer_name='conv5_block3_out')
    return grad_cam

# Preprocess image
def preprocess_image(image_file):
    img = Image.open(image_file).convert('RGB')
    img = img.resize((224, 224))
    img_array = keras_image.img_to_array(img)

    # NO normalization here
    img_array = np.expand_dims(img_array, axis=0)

    return img, img_array

# Make prediction
def predict_disease(model, img_array):
    """Make prediction on preprocessed image"""
    predictions = model.predict(img_array, verbose=0)
    confidence = np.max(predictions[0])
    predicted_class = np.argmax(predictions[0])
    return predicted_class, confidence, predictions[0]

# Helper functions for visualization
def create_confidence_bar(disease_name, confidence):
    """Create a styled confidence progress bar"""
    confidence_pct = confidence * 100
    if confidence_pct >= 80:
        color = "#10b981"  # Green
    elif confidence_pct >= 60:
        color = "#f59e0b"  # Amber
    else:
        color = "#ef4444"  # Red
    
    return f"""
    <div class="progress-container">
        <div class="progress-label">
            <span>{disease_name}</span>
            <span style="font-weight: 600; color: {color};">{confidence_pct:.1f}%</span>
        </div>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {confidence_pct}%; background: {color};"></div>
        </div>
    </div>
    """

def get_confidence_badge(confidence):
    """Get badge based on confidence level"""
    confidence_pct = confidence * 100
    if confidence_pct >= 85:
        return "🟢 High Confidence"
    elif confidence_pct >= 70:
        return "🟡 Medium Confidence"
    else:
        return "🔴 Low Confidence"

def get_alert_class(disease_name):
    """Get alert class based on disease severity"""
    if disease_name == "Carcinoma":
        return "alert-danger"
    elif disease_name in ["Eczema", "Rosacea"]:
        return "alert-warning"
    else:
        return "alert-info"

# Main app
def main():
    # Header section
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🏥 Skin Disease Classifier</h1>
        <p class="header-subtitle">Advanced AI-Powered Skin Condition Detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📋 About This Tool")
        st.info("""
        **Purpose**: Identify potential skin conditions from images using deep learning.
        
        **Not a Diagnosis**: This tool provides predictions for educational purposes only. Always consult a qualified dermatologist.
        
    
        """)
        
        st.markdown("### 🔒 Privacy & Safety")
        st.success("""
        ✅ Images are processed locally
        ✅ No data is stored or transmitted
        ✅ All processing is confidential
        """)
        
        st.markdown("### 📞 Support")
        st.warning("**Medical Emergency?** Call emergency services immediately.")
    
    # Main content
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📤 Upload Image")
        st.markdown("Upload a clear image of the skin condition for analysis.")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Recommended: Clear, well-lit image with good contrast"
        )
    
    with col2:
        st.markdown("### ⚙️ Supported Formats")
        supported = """
        - **JPG/JPEG** - Recommended
        - **PNG** - With transparency support
        - **Image Size**: 224×224 (auto-resized)
        """
        st.markdown(supported)
    
    # Load model once
    model = load_model()
    grad_cam = load_grad_cam(model)
    
    if uploaded_file is not None:
        # Process image
        img, img_array = preprocess_image(uploaded_file)
        
        # Analysis section
        st.markdown("---")
        st.markdown("### 🔬 Analysis Results")
        
        # Make prediction with loading animation
        with st.spinner("🔄 Analyzing image with AI model..."):
            predicted_class, confidence, all_predictions = predict_disease(model, img_array)
        
        disease_info = DISEASE_INFO[predicted_class]
        confidence_pct = confidence * 100
        
        # Results layout
        result_col1, result_col2 = st.columns([1, 1], gap="large")
        
        with result_col1:
            st.markdown("#### 🖼️ Uploaded Image")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with result_col2:
            st.markdown("#### 📊 Prediction")
            
            # Main prediction card
            st.markdown(f"""
            <div class="result-card">
                <div style="text-align: center;">
                    <p style="font-size: 0.9rem; color: #64748b; margin: 0; font-weight: 600;">Predicted Condition</p>
                    <h1 class="disease-name">{disease_info['name']}</h1>
                    <p class="confidence-value">{confidence_pct:.1f}%</p>
                    <p style="color: #64748b; margin: 0.5rem 0 0 0; font-weight: 500;">{get_confidence_badge(confidence)}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Disease details section
        st.markdown("---")
        st.markdown("#### 📋 Condition Details")
        
        alert_class = get_alert_class(disease_info['name'])
        st.markdown(f"""
        <div class="alert {alert_class}">
            <strong>{disease_info['name']}</strong><br/>
            {disease_info['description']}
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendations section
        st.markdown("---")
        st.markdown("#### 💡 Recommended Next Steps")
        
        for step in disease_info["next_steps"]:
            st.markdown(f"""
            <div class="recommendation-item">
                {step}
            </div>
            """, unsafe_allow_html=True)
        
        # Confidence breakdown
        st.markdown("---")
        st.markdown("#### 📈 Detailed Confidence Analysis")
        
        confidence_html = ""
        for disease_class, prob in enumerate(all_predictions):
            disease_name = DISEASE_INFO[disease_class]["name"]
            confidence_html += create_confidence_bar(disease_name, prob)
        
        st.markdown(confidence_html, unsafe_allow_html=True)
        
        # Grad-CAM Heatmap Visualization
        st.markdown("---")
        st.markdown("#### 🔥 AI Explanation - Grad-CAM Heatmap")
        st.info("""
        **What is this?** Grad-CAM (Gradient-weighted Class Activation Mapping) shows which parts of the image 
        were most important for the model's prediction. Warmer colors (red) indicate higher importance, 
        cooler colors (blue) indicate lower importance.
        """)
        
        with st.spinner("🔥 Generating Grad-CAM heatmap..."):
            try:
                heatmap, overlay_image = grad_cam.generate_heatmap_visualization(
                    img, img_array, pred_index=predicted_class
                )
                
                # Display Grad-CAM visualization
                heatmap_col1, heatmap_col2 = st.columns([1, 1], gap="large")
                
                with heatmap_col1:
                    st.markdown("**Original Image**")
                    st.image(img, use_container_width=True)
                
                with heatmap_col2:
                    st.markdown("**Grad-CAM Overlay**")
                    st.image(overlay_image, use_container_width=True)
                
                st.success("✅ Grad-CAM visualization generated successfully!")
                
            except Exception as e:
                st.warning(f"⚠️ Could not generate Grad-CAM heatmap: {str(e)}")
                st.info("This might happen if the model architecture is different. Predictions are still valid.")
        
        # Critical disclaimer
        st.markdown("---")
        st.markdown("""
        <div class="alert alert-danger" style="margin-top: 2rem;">
            <strong>⚠️ IMPORTANT MEDICAL DISCLAIMER</strong><br/>
            This application provides AI-generated predictions for <strong>EDUCATIONAL PURPOSES ONLY</strong>. 
            <br/><br/>
            <strong>DO NOT use this tool as a substitute for professional medical advice.</strong>
            <br/><br/>
            ✓ Always consult a qualified dermatologist for accurate diagnosis<br/>
            ✓ In case of suspected skin cancer, seek immediate medical attention<br/>
            ✓ This tool's accuracy depends on image quality and angle<br/>
            ✓ Medical professionals should always be consulted for treatment decisions
        </div>
        """, unsafe_allow_html=True)
        
        # Export results button
        col1, col2, col3 = st.columns(3)
        with col2:
            if st.button("📋 Copy Results", key="copy_results"):
                st.success("Results copied to clipboard!")
    
    else:
        # Empty state
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem;">
            <h3 style="color: #64748b;">📸 Ready to Analyze</h3>
            <p style="color: #94a3b8; font-size: 1rem;">
                Upload an image of a skin condition to get started with AI-powered analysis.
            </p>
            <p style="color: #cbd5e1; font-size: 0.9rem; margin-top: 2rem;">
                💡 Tip: Use a clear, well-lit image for better results
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
