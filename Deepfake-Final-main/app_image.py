import streamlit as st
from PIL import Image
import numpy as np
from inference import predict
import time
from fpdf import FPDF

# --- Page Configuration ---
st.set_page_config(page_title="Authentic AI - Deep Fake Detection", page_icon="🔍", layout="wide")

# --- Custom CSS for Styling ---
st.markdown(
    """
    <style>
        @media screen and (max-width: 768px) {
            .title {
                font-size: 32px;
            }
            .subheader {
                font-size: 20px;
            }
            .section {
                font-size: 16px;
                line-height: 1.5;
            }
        }
        .title {
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            color: #ffffff;
            margin-bottom: 30px;
        }
        .subheader {
            font-size: 24px;
            text-align: center;
            color: #b3b3b3;
            margin-bottom: 40px;
        }
        .section {
            margin-bottom: 30px;
            font-family: 'Georgia', serif;
            font-size: 18px;
            color: white;
            text-align: justify;
            line-height: 1.4;
        }
        .section h2 {
            font-size: 20px;
            color: #ffcc00;
            text-align: justify;
            margin-bottom: 5px;
        }
        .section p {
            margin-bottom: 5px;
        }
        .bold-text {
            font-weight: bold;
            font-size: 18px;
            color: #00ffaa;
            text-align: justify;
        }
        .testimonial-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 20px;
        }
        .testimonial-card {
            background-color: #222;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
            text-align: center;
            margin: 10px;
            width: 30%;
            min-width: 250px;
        }
        .testimonial-card:hover {
            transform: scale(1.05);
            box-shadow: 0px 0px 20px rgba(255, 255, 255, 0.5);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Page Content ---
st.markdown('<h1 class="title">🔍 Authentic AI - Deep Fake Detection</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="subheader">Detect AI-generated images with advanced artificial intelligence technology</h3>', unsafe_allow_html=True)

# About Deepfake Misuse and Solution
st.markdown("""
    <div class='section'>
        <h2>🛑 About Deep Fake Misuse and Solution</h2>
        <p>Deep Fake technology is increasingly being exploited for misinformation, financial fraud, identity theft, and malicious propaganda. The ability to create hyper-realistic fake images and videos poses serious challenges to digital trust and security.</p>
        <br>
        <h2>🛑 The Growing Threat of Deepfakes</h2>
        <p>Deep Fake technology is widely used to manipulate media, spread disinformation, and create synthetic identities. Its rapid advancement demands robust countermeasures.</p>
        <br>
        <h2>🔬 Our AI-Powered Solution</h2>
        <p class="bold-text">Authentic AI</p> uses cutting-edge machine learning algorithms to detect deep fakes with <b>high accuracy</b>. Our model analyzes <b>image metadata, texture inconsistencies, and digital fingerprints</b> to differentiate real from manipulated media. We aim to enhance digital security for journalists, law enforcement, and the general public.
    </div>
    """, unsafe_allow_html=True)

# Testimonials Section
st.markdown("""
    <h3 style='text-align:center; color:#ffcc00;'>⭐ What People Say ⭐</h3>
    <div class='testimonial-container'>
        <div class='testimonial-card'>
            <p>"Authentic AI is a revolutionary tool for detecting fake images! Highly accurate and easy to use."</p>
            <span class='testimonial-author'>- Dr. Emily Carter, AI Researcher</span>
        </div>
        <div class='testimonial-card'>
            <p>"A must-have tool for fact-checkers and journalists. It saves hours of manual verification!"</p>
            <span class='testimonial-author'>- Mark Thompson, Investigative Journalist</span>
        </div>
        <div class='testimonial-card'>
            <p>"This AI-driven approach to deepfake detection is groundbreaking. Highly recommended!"</p>
            <span class='testimonial-author'>- Sarah Lee, Media Analyst</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# File uploader
uploaded_file = st.file_uploader("📷 Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], key="file_uploader")

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("")

    # Classification process with progress bar
    with st.spinner("Analyzing image..."):
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1)
        prediction = predict(image)
        progress_bar.empty()
    
    # Display prediction results
    confidence_score = float(prediction[0][0])  # Assuming model output is probability
    formatted_confidence = confidence_score * 100 if confidence_score > 0.5 else (1 - confidence_score) * 100
    
    st.subheader("🧐 Prediction Result")
    if confidence_score > 0.5:
        st.success(f"✅ The image is likely **REAL** (Confidence: {formatted_confidence:.2f}%)")
    else:
        st.error(f"🚨 The image is likely **FAKE** (Confidence: {formatted_confidence:.2f}%)")
    
    # Generate PDF Report
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Deepfake Detection Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Image: {uploaded_file.name}", ln=True)
    pdf.cell(200, 10, txt=f"Prediction: {'REAL' if confidence_score > 0.5 else 'FAKE'}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {formatted_confidence:.2f}%", ln=True)
    pdf.cell(200, 10, txt="Analysis conducted using AI-powered deepfake detection.", ln=True)
    pdf_output = "deepfake_report.pdf"
    pdf.output(pdf_output)
    
    st.download_button(label="📄 Download Report (PDF)", data=open(pdf_output, "rb"), file_name=pdf_output, mime="application/pdf")
    
    st.write("\n🔬 This tool uses AI to analyze image authenticity. Use results with caution!")

# Footer
st.markdown("""
    <hr>
    <p style='text-align:center; color:gray;'>
        Made with ❤️ by Team Authentic AI | Hackathon 2025
    </p>
    """, unsafe_allow_html=True)
