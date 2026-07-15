import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import base64
from PIL import Image

# Resolve absolute path for the tab icon
script_dir = os.path.dirname(os.path.abspath(__file__))
icon_filename = "icon-app.png"
icon_path = os.path.join(script_dir, icon_filename)

# Check and open local image as icon, fallback to emoji if there's any loading issue
if os.path.exists(icon_path):
    try:
        app_icon = Image.open(icon_path)
    except Exception:
        app_icon = "🌺"
else:
    app_icon = "🌺"

# Set page configuration
st.set_page_config(
    page_title="Prediksi Jenis Bunga Iris",
    page_icon=app_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to convert local image to base64 for embedding in HTML headers
def get_image_base64(filepath):
    try:
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except Exception:
        pass
    return None

# Custom CSS for modern simple blue minimalist theme with Mobile-Friendly Media Queries
st.markdown("""
    <style>
    /* Clean modern sans-serif typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Override background and text colors globally for light-minimalist style */
    .stApp, .main, [data-testid="stHeader"] {
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    
    /* Sidebar styling: subtle light blue-gray background */
    [data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Set custom text colors for general markdown and labels */
    p, span, label, div {
        color: #1e293b;
    }
    
    /* Header colors */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    
    /* Custom Title Style (inline rendering for robust wrapping) */
    .main-title {
        color: #1e40af !important;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        line-height: 1.25;
    }
    .title-icon {
        width: 55px;
        height: auto;
        vertical-align: middle;
        margin-right: 12px;
        border-radius: 6px;
        display: inline-block;
    }
    .main-subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Custom Result Badges */
    .result-card {
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        margin-top: 1rem;
        font-weight: 600;
        font-size: 1.3rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Iris Species Specific Result Styles */
    .iris-setosa {
        background-color: #eff6ff;
        border: 1px solid #3b82f6;
        color: #1e40af;
    }
    .iris-versicolor {
        background-color: #faf5ff;
        border: 1px solid #a855f7;
        color: #6b21a8;
    }
    .iris-virginica {
        background-color: #fffbeb;
        border: 1px solid #f59e0b;
        color: #b45309;
    }
    
    /* Modern minimalist sliders styling in sidebar */
    .stSlider [data-testid="stWidgetLabel"] p {
        font-weight: 500 !important;
        color: #475569 !important;
    }
    
    /* Style the native Streamlit container to look like our minimalist cards */
    [data-testid="stElementContainer"] div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #e2e8f0 !important;
        background-color: #ffffff !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
        border-radius: 8px !important;
    }
    
    /* Mobile optimization rules (screen sizes smaller than 768px) */
    @media (max-width: 768px) {
        .main-title {
            font-size: 1.5rem !important;
            text-align: center !important;
        }
        .title-icon {
            width: 42px !important;
            margin-right: 8px !important;
        }
        .main-subtitle {
            font-size: 0.85rem !important;
            text-align: center;
            margin-bottom: 1rem !important;
        }
        .result-card {
            padding: 1rem !important;
            font-size: 1.1rem !important;
        }
        .result-card h2 {
            font-size: 1.25rem !important;
            margin-top: 0.25rem !important;
        }
        /* Push main content container down to clear the floating top header bar on mobile */
        .block-container {
            padding-top: 3.5rem !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
    }
    
    /* Hide default streamlit decoration */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Helper function to load model safely
@st.cache_resource
def load_pickle_model(filepath):
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model, None
    except Exception as e:
        return None, str(e)

# Initialize Session State values for Sliders (needed for dynamic preset loading)
if 'sepal_l' not in st.session_state:
    st.session_state.sepal_l = 5.8
if 'sepal_w' not in st.session_state:
    st.session_state.sepal_w = 3.0
if 'petal_l' not in st.session_state:
    st.session_state.petal_l = 3.8
if 'petal_w' not in st.session_state:
    st.session_state.petal_w = 1.2
if 'prev_mode' not in st.session_state:
    st.session_state.prev_mode = "Atur Sendiri Secara Manual"

# Sidebar Inputs (Minimalist Settings Panel)
st.sidebar.markdown("<h2 style='font-size: 1.3rem; color: #1e40af !important; margin-bottom: 0.5rem;'>⚙️ Pilihan Input</h2>", unsafe_allow_html=True)

input_mode = st.sidebar.selectbox(
    "Metode Pengisian Ukuran",
    ["Atur Sendiri Secara Manual", "Contoh Bunga Setosa", "Contoh Bunga Versicolor", "Contoh Bunga Virginica"]
)

# If input mode select box changes, update slider values in session state
if input_mode != st.session_state.prev_mode:
    st.session_state.prev_mode = input_mode
    if input_mode == "Contoh Bunga Setosa":
        st.session_state.sepal_l = 5.0
        st.session_state.sepal_w = 3.4
        st.session_state.petal_l = 1.5
        st.session_state.petal_w = 0.3
    elif input_mode == "Contoh Bunga Versicolor":
        st.session_state.sepal_l = 5.9
        st.session_state.sepal_w = 2.8
        st.session_state.petal_l = 4.3
        st.session_state.petal_w = 1.3
    elif input_mode == "Contoh Bunga Virginica":
        st.session_state.sepal_l = 6.6
        st.session_state.sepal_w = 3.0
        st.session_state.petal_l = 5.6
        st.session_state.petal_w = 2.0

st.sidebar.markdown("<hr style='border: 0; border-top: 1px solid #e2e8f0; margin: 1rem 0;'>", unsafe_allow_html=True)
st.sidebar.markdown("<h4 style='font-size: 0.95rem; color: #0f172a; margin-bottom: 1rem;'>Geser Parameter Kelopak:</h4>", unsafe_allow_html=True)

with st.sidebar:
    sepal_length = st.slider("Panjang Kelopak Luar (cm)", 4.3, 7.9, key='sepal_l', step=0.1)
    sepal_width = st.slider("Lebar Kelopak Luar (cm)", 2.0, 4.4, key='sepal_w', step=0.1)
    st.markdown("<hr style='border: 0; border-top: 1px solid #e2e8f0; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    petal_length = st.slider("Panjang Kelopak Dalam (cm)", 1.0, 6.9, key='petal_l', step=0.1)
    petal_width = st.slider("Lebar Kelopak Dalam (cm)", 0.1, 2.5, key='petal_w', step=0.1)

# Embed visual Iris Diagram Guide in the sidebar with absolute path resolution
image_filename = "iris_guide.png"
image_path = os.path.join(script_dir, image_filename)

if os.path.exists(image_path):
    st.sidebar.image(image_path, width='stretch')
else:
    st.sidebar.warning("Gambar panduan tidak ditemukan di sistem.")

# Sepal and Petal text explanation under the guide image
st.sidebar.markdown("""
    <div style='margin-top: 0.75rem;'>
        <p style='font-size: 0.8rem; color: #475569; margin-bottom: 0.5rem; line-height: 1.4;'>
            <strong>🌸 Kelopak Luar (Sepal):</strong> Bagian pelindung bunga terluar yang menopang kelopak mahkota saat bunga mekar.
        </p>
        <p style='font-size: 0.8rem; color: #475569; line-height: 1.4;'>
            <strong>🌺 Kelopak Dalam (Petal):</strong> Mahkota bunga dalam yang berwarna indah dan mencolok.
        </p>
    </div>
""", unsafe_allow_html=True)

# GitHub Profile Credit Section at the bottom of the sidebar (with inline SVG Github icon)
st.sidebar.markdown("<hr style='border: 0; border-top: 1px solid #e2e8f0; margin: 1rem 0;'>", unsafe_allow_html=True)
st.sidebar.markdown("""
    <div style='text-align: center; font-size: 0.8rem; color: #64748b; display: flex; justify-content: center; align-items: center; gap: 5px;'>
        Dibuat oleh <a href='https://github.com/khamalputra' target='_blank' style='color: #1e40af; text-decoration: none; font-weight: 600;'>Ade Khamelia Putra</a>
        <svg height="14" width="14" viewBox="0 0 16 16" style="fill: #64748b; vertical-align: middle; display: inline-block;"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
    </div>
""", unsafe_allow_html=True)

# Main Title Section
# Embed the custom app icon as a Base64 image in the header title
icon_base64 = get_image_base64(icon_path)
if icon_base64:
    st.markdown(f"""
        <h1 class='main-title'>
            <img src="data:image/png;base64,{icon_base64}" class="title-icon"/>Analisis Jenis Bunga Iris
        </h1>
    """, unsafe_allow_html=True)
else:
    st.markdown("<h1 class='main-title'>🌺 Analisis Jenis Bunga Iris</h1>", unsafe_allow_html=True)

st.markdown("<p class='main-subtitle'>Hasil prediksi dan perbandingan dimensi kelopak ditampilkan secara real-time di bawah ini.</p>", unsafe_allow_html=True)

# Mobile-friendly tips banner (Streamlit sidebar is collapsed by default on mobile)
st.markdown("""
    <div style='background-color: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 0.75rem 1rem; font-size: 0.85rem; color: #1e40af; margin-bottom: 1.5rem;'>
        <strong>💡 Petunjuk HP (Mobile):</strong> Tekan tombol panah kecil <strong>&gt;&gt;</strong> di pojok kiri atas layar untuk membuka panel pengatur kelopak bunga.
    </div>
""", unsafe_allow_html=True)

# Load model using absolute path resolution
model_filename = "best_svm_model.pkl"
model_path = os.path.join(script_dir, model_filename)
model, err = load_pickle_model(model_path)

if err:
    st.error(f"Gagal memuat sistem prediksi. Silakan hubungi admin.")
    st.info(f"Detail Kesalahan: {err}")
else:
    # Layout columns (Main Page)
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("<h3 style='font-size: 1.1rem; margin-bottom: 1rem;'>Hasil Prediksi</h3>", unsafe_allow_html=True)
        
        # Predict
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)[0]
        
        # Species styling cards
        if prediction == 0:
            st.markdown("""
                <div class="result-card iris-setosa">
                    <span style='font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; color: #3b82f6;'>Jenis Hasil Analisis</span>
                    <h2 style='margin: 0.5rem 0 0 0; color: #1e40af !important;'>Iris Setosa</h2>
                </div>
            """, unsafe_allow_html=True)
        elif prediction == 1:
            st.markdown("""
                <div class="result-card iris-versicolor">
                    <span style='font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; color: #a855f7;'>Jenis Hasil Analisis</span>
                    <h2 style='margin: 0.5rem 0 0 0; color: #6b21a8 !important;'>Iris Versicolor</h2>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="result-card iris-virginica">
                    <span style='font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; color: #f59e0b;'>Jenis Hasil Analisis</span>
                    <h2 style='margin: 0.5rem 0 0 0; color: #b45309 !important;'>Iris Virginica</h2>
                </div>
            """, unsafe_allow_html=True)
            
        # Info box
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
            <div style='background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; font-size: 0.85rem; color: #475569;'>
                <strong>Keterangan:</strong> Hasil analisis diperoleh secara instan dengan memproses ukuran kelopak luar (sepal) dan kelopak dalam (petal) yang Anda tentukan di panel kiri (sidebar).
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("<h3 style='font-size: 1.1rem; margin-bottom: 1rem;'>Grafik Dimensi Kelopak (cm)</h3>", unsafe_allow_html=True)
        
        # Prepare data for plotting with clean, shorter names
        chart_data = pd.DataFrame({
            "Ukuran (cm)": [sepal_length, sepal_width, petal_length, petal_width]
        }, index=[
            "Kelopak Luar (P)",
            "Kelopak Luar (L)",
            "Kelopak Dalam (P)",
            "Kelopak Dalam (L)"
        ])
        
        # Plotting clean horizontal minimalist bar chart
        with st.container(border=True):
            st.bar_chart(chart_data, color="#3b82f6", horizontal=True)

    # Full width visual reference gallery at the bottom
    st.markdown("<hr style='border: 0; border-top: 1px solid #e2e8f0; margin: 2rem 0;'>", unsafe_allow_html=True)
    
    # Embed custom app icon in the bottom section header (PNG supports transparency)
    if icon_base64:
        st.markdown(f"""
            <h3 style='font-size: 1.2rem; margin-bottom: 1rem; display: flex; align-items: center;'>
                <img src="data:image/png;base64,{icon_base64}" width="28" style="vertical-align: middle; margin-right: 8px; border-radius: 3px;"/>
                Mengenal Spesies Bunga Iris
            </h3>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='font-size: 1.2rem; margin-bottom: 1rem;'>🌸 Mengenal Spesies Bunga Iris</h3>", unsafe_allow_html=True)
        
    with st.container(border=True):
        st.image("https://miro.medium.com/v2/resize:fit:1400/1*ZK9_HrpP_lhSzTq9xVJUQw.png", caption="Perbandingan Tiga Spesies: Iris Setosa, Iris Versicolor, dan Iris Virginica", width='stretch')
