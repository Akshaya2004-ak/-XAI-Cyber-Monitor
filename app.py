import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from utils.feature_extraction import extract_features
from utils.traffic_analyzer import TrafficAnalyzer
import warnings
warnings.filterwarnings('ignore')

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="XAI Cyber Monitor",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CSS STYLING ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #60a5fa 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 15px 30px rgba(30, 58, 138, 0.3);
        animation: gradient-flow 4s ease infinite;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
        animation: shine 2.5s infinite;
    }
    
    @keyframes shine {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    @keyframes gradient-flow {
        0%, 100% { background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #60a5fa 100%); }
        50% { background: linear-gradient(135deg, #60a5fa 0%, #1e3a8a 50%, #3b82f6 100%); }
    }
    
    .main-header h1 {
        font-family: 'Roboto', sans-serif;
        font-weight: 700;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        font-family: 'Roboto', sans-serif;
        font-weight: 400;
        font-size: 1.2rem;
        opacity: 0.85;
        position: relative;
        z-index: 1;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #f8fafc 0%, #ffffff 100%);
        padding: 1.8rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #3b82f6, #1e3a8a);
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.12);
    }
    
    .threat-detected {
        background: linear-gradient(145deg, #fef2f2 0%, #fee2e2 100%);
        border: 1px solid #fecaca;
        box-shadow: 0 6px 20px rgba(239, 68, 68, 0.12);
    }
    
    .threat-detected::before {
        background: linear-gradient(90deg, #ef4444, #dc2626);
    }
    
    .safe-traffic {
        background: linear-gradient(145deg, #ecfdf5 0%, #d1fae5 100%);
        border: 1px solid #a7f3d0;
        box-shadow: 0 6px 20px rgba(34, 197, 94, 0.12);
    }
    
    .safe-traffic::before {
        background: linear-gradient(90deg, #10b981, #059669);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1e3a8a 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.8rem 2.5rem;
        font-weight: 500;
        font-family: 'Roboto', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 3px 12px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
    }
    
    .stSelectbox > div > div {
        background: linear-gradient(145deg, #f8fafc 0%, #ffffff 100%);
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
    }
    
    .stTextArea > div > div > textarea {
        background: linear-gradient(145deg, #f8fafc 0%, #ffffff 100%);
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
        font-family: 'Roboto', sans-serif;
        color: #1f2937;
    }
    
    .stTextArea > div > div > textarea::placeholder {
        color: #6b7280;
        opacity: 0.8;
    }
    
    @media (prefers-color-scheme: dark) {
        .stTextArea > div > div > textarea {
            background: linear-gradient(145deg, #1f2937 0%, #374151 100%) !important;
            border: 1px solid #4b5563 !important;
            color: #f3f4f6 !important;
            box-shadow: 0 3px 12px rgba(0,0,0,0.2) !important;
        }
        
        .stTextArea > div > div > textarea::placeholder {
            color: #9ca3af !important;
            opacity: 0.7 !important;
        }
    }
    
    .stApp[data-theme="dark"] .stTextArea > div > div > textarea,
    html[data-theme="dark"] .stTextArea > div > div > textarea {
        background: linear-gradient(145deg, #1f2937 0%, #374151 100%) !important;
        border: 1px solid #4b5563 !important;
        color: #f3f4f6 !important;
        box-shadow: 0 3px 12px rgba(0,0,0,0.2) !important;
    }
    
    .stApp[data-theme="dark"] .stTextArea > div > div > textarea::placeholder,
    html[data-theme="dark"] .stTextArea > div > div > textarea::placeholder {
        color: #9ca3af !important;
        opacity: 0.7 !important;
    }
    
    @media (prefers-color-scheme: dark) {
        .stTextArea > div > div > textarea:focus {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3) !important;
        }
    }
    
    .stApp[data-theme="dark"] .stTextArea > div > div > textarea:focus,
    html[data-theme="dark"] .stTextArea > div > div > textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3) !important;
    }
    
    .upload-section {
        background: linear-gradient(145deg, #f8fafc 0%, #ffffff 100%);
        padding: 2.5rem;
        border-radius: 15px;
        border: 2px dashed #3b82f6;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .upload-section::before {
        content: '📤';
        font-size: 3.5rem;
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        opacity: 0.08;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translate(-50%, -50%) translateY(0px); }
        50% { transform: translate(-50%, -50%) translateY(-12px); }
    }
    
    .upload-section:hover {
        border-color: #1e3a8a;
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.15);
    }
    
    .sidebar .stSelectbox > div > div {
        background: linear-gradient(145deg, #3b82f6 0%, #1e3a8a 100%);
        color: white;
        border-radius: 10px;
    }
    
    .analysis-section {
        background: linear-gradient(145deg, #f8fafc 0%, #ffffff 100%);
        padding: 2.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
    }
    
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6, #1e3a8a);
        border-radius: 8px;
    }
    
    .footer {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 3rem;
        box-shadow: 0 -8px 25px rgba(59, 130, 246, 0.2);
    }
    
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
        padding: 1.2rem;
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #3b82f6 0%, #1e3a8a 100%);
        color: white;
        padding: 1.8rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .sidebar-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
        animation: shine 3s infinite;
    }
    
    .sidebar-header h2 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 700;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    
    .sidebar-header .subtitle {
        margin: 0.5rem 0 0 0;
        font-size: 0.95rem;
        opacity: 0.85;
        position: relative;
        z-index: 1;
    }
    
    .analysis-selector {
        background: transparent;
        border: none;
        border-radius: 0;
        padding: 1rem 0;
        margin: 1rem 0;
        box-shadow: none;
        position: relative;
    }

    .selector-header {
        background: linear-gradient(135deg, #3b82f6 0%, #1e3a8a 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.2);
    }

    .selector-header h3 {
        margin: 0;
        font-size: 1.3rem;
        font-weight: 600;
    }
    
    div[data-testid="stRadio"] > div {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
        box-shadow: none !important;
    }
    
    div[data-testid="stRadio"] label {
        display: flex !important;
        align-items: center !important;
        background: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 1.2rem !important;
        margin: 0.5rem 0 !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        color: #1f2937 !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08) !important;
    }
    
    @media (prefers-color-scheme: dark) {
        div[data-testid="stRadio"] label {
            background: #1f2937 !important;
            border: 2px solid #4b5563 !important;
            color: #f3f4f6 !important;
            box-shadow: 0 3px 12px rgba(0,0,0,0.2) !important;
        }
    }
    
    .stApp[data-theme="dark"] div[data-testid="stRadio"] label,
    html[data-theme="dark"] div[data-testid="stRadio"] label {
        background: #1f2937 !important;
        border: 2px solid #4b5563 !important;
        color: #f3f4f6 !important;
        box-shadow: 0 3px 12px rgba(0,0,0,0.2) !important;
    }
    
    div[data-testid="stRadio"] label:hover {
        background: linear-gradient(135deg, #3b82f6 0%, #1e3a8a 100%) !important;
        color: white !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
        border-color: #3b82f6 !important;
    }
    
    div[data-testid="stRadio"] label > div:first-child {
        background: transparent !important;
        border: none !important;
        margin-right: 0.9rem !important;
    }
    
    .mode-description {
        background: linear-gradient(145deg, #eff6ff 0%, #dbeafe 100%);
        border-left: 4px solid #3b82f6;
        border-radius: 12px;
        padding: 1.8rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.08);
    }
    
    @media (prefers-color-scheme: dark) {
        .mode-description {
            background: linear-gradient(145deg, #1f2937 0%, #374151 100%);
            color: #f3f4f6;
        }
        .mode-description .title {
            color: #f3f4f6 !important;
        }
        .mode-description .desc {
            color: #d1d5db !important;
        }
    }
    
    .stApp[data-theme="dark"] .mode-description,
    html[data-theme="dark"] .mode-description {
        background: linear-gradient(145deg, #1f2937 0%, #374151 100%);
        color: #f3f4f6;
    }
    
    .stApp[data-theme="dark"] .mode-description .title,
    html[data-theme="dark"] .mode-description .title {
        color: #f3f4f6 !important;
    }
    
    .stApp[data-theme="dark"] .mode-description .desc,
    html[data-theme="dark"] .mode-description .desc {
        color: #d1d5db !important;
    }
    
    .mode-description .icon {
        font-size: 1.8rem;
        margin-bottom: 0.9rem;
        display: block;
    }
    
    .mode-description .title {
        font-weight: 700;
        color: #1f2937;
        font-size: 1.2rem;
        margin-bottom: 0.9rem;
    }
    
    .mode-description .desc {
        font-size: 0.95rem;
        color: #4b5563;
        line-height: 1.6;
    }
    
    .sidebar-stats {
        background: linear-gradient(145deg, #f8fafc 0%, #ffffff 100%);
        border-radius: 15px;
        padding: 1.8rem;
        margin: 1.5rem 0;
        border: 2px solid #e2e8f0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        position: relative;
        overflow: hidden;
    }
    
    @media (prefers-color-scheme: dark) {
        .sidebar-stats {
            background: linear-gradient(145deg, #1f2937 0%, #374151 100%);
            border: 2px solid #4b5563;
        }
        .stats-header, .stat-label, .stat-value {
            color: #f3f4f6 !important;
        }
        .stat-item {
            border-bottom: 1px solid #4b5563 !important;
        }
    }
    
    .stApp[data-theme="dark"] .sidebar-stats,
    html[data-theme="dark"] .sidebar-stats {
        background: linear-gradient(145deg, #1f2937 0%, #374151 100%);
        border: 2px solid #4b5563;
    }
    
    .stApp[data-theme="dark"] .stats-header,
    .stApp[data-theme="dark"] .stat-label,
    .stApp[data-theme="dark"] .stat-value,
    html[data-theme="dark"] .stats-header,
    html[data-theme="dark"] .stat-label,
    html[data-theme="dark"] .stat-value {
        color: #f3f4f6 !important;
    }
    
    .stApp[data-theme="dark"] .stat-item,
    html[data-theme="dark"] .stat-item {
        border-bottom: 1px solid #4b5563 !important;
    }
    
    .sidebar-stats::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        width: 100%; 
        height: 3px;
        background: linear-gradient(90deg, #10b981, #059669);
        border-radius: 15px 15px 0 0;
        z-index: 1; 
    }
    
    .stats-header {
        text-align: center;
        margin-bottom: 1.5rem;
        color: #1f2937;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    .stat-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.9rem 0;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .stat-item:last-child {
        border-bottom: none;
    }
    
    .stat-label {
        font-size: 0.95rem;
        color: #4b5563;
        font-weight: 500;
    }
    
    .stat-value {
        font-weight: 700;
        color: #1f2937;
        font-size: 0.95rem;
    }
    
    div[data-testid="stRadio"] > div:first-child {
        display: none !important;
    }
    
    div[data-testid="stRadio"] {
        margin-top: -1rem !important;
    }
    
    .selector-header {
        margin-bottom: 0rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ========== FEATURE NAMES ==========
FEATURE_NAMES = [
    'Token_Count', 'Token_Length_Sum', 'Avg_Token_Length', 'Max_Token_Length', 'URL_Length',
    'Special_Chars', 'Encoded_Chars', 'Numeric_Chars', 'Query_Length',
    'SQL_Keyword_Count', 'SQL_Operator_Count', 'SQL_Function_Count', 'SQL_Comment_Pattern',
    'SQL_Quote_Pattern', 'SQL_Equals_Pattern', 'SQL_Union_Pattern', 'SQL_OR_Injection',
    'XSS_Keyword_Count', 'XSS_Tag_Count', 'XSS_Event_Handler', 'XSS_JS_Protocol',
    'XSS_Encoded_Script', 'XSS_HTML_Entities', 'Query_Entropy', 'Path_Entropy',
    'URL_Entropy', 'Param_Count', 'Avg_Param_Length', 'Suspicious_Param_Chars',
    'Directory_Traversal', 'File_Inclusion', 'Command_Injection'
]

# ========== INITIALIZE SESSION STATE ==========
if 'model' not in st.session_state:
    st.session_state.model = joblib.load("model/rf_model.pkl")
    st.session_state.explainer = shap.Explainer(st.session_state.model)
    st.session_state.traffic_analyzer = TrafficAnalyzer()

# ========== XAI VISUALIZATION FUNCTIONS ==========
def show_shap_explanation(features):
    try:
        shap_values = st.session_state.explainer(np.array([features]))
        if len(shap_values.shape) == 3:
            shap_values = shap_values[..., 1]
        
        df = pd.DataFrame({
            'Feature': FEATURE_NAMES,
            'SHAP Value': shap_values.values[0],
            'Color': ['#ef4444' if x > 0 else '#3b82f6' for x in shap_values.values[0]]
        }).sort_values('SHAP Value', key=abs, ascending=False)
        
        fig = px.bar(
            df.head(15),
            x='SHAP Value',
            y='Feature',
            color='Color',
            color_discrete_map='identity',
            orientation='h',
            title='SHAP Feature Impact (Positive = More Malicious)'
        )
        fig.update_layout(
            height=600,
            showlegend=False,
            margin=dict(l=150)
        )
        return fig
    except Exception as e:
        st.error(f"SHAP explanation error: {str(e)}")
        return px.bar()

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2>⚙️ Command Center</h2>
        <p class="subtitle">ML-driven security with explainable AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    analysis_options = [
        {
            "value": "Real-time URL Analysis",
            "icon": "⚡",
            "title": "Instant URL Scanner",
            "description": "Analyze URLs in real-time to detect threats and malicious patterns"
        },
        {
            "value": "Batch File Analysis", 
            "icon": "📦",
            "title": "Bulk File Processor",
            "description": "Process multiple URLs from CSV files for comprehensive threat analysis"
        },
        {
            "value": "Model Performance",
            "icon": "📊", 
            "title": "Model Analytics",
            "description": "Review detailed metrics and performance analytics for the AI model"
        },
        {
            "value": "Traffic Insights",
            "icon": "🔗",
            "title": "Network Monitor",
            "description": "Gain insights into network traffic patterns and threat trends"
        }
    ]
    
    st.markdown("""
    <div class="analysis-selector">
        <div class="selector-header" style="margin-bottom: -1rem;">
            <h3>🎯 Select Analysis Mode</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)

    analysis_mode = st.radio(
        "analysis_mode_selector",
        options=[opt["value"] for opt in analysis_options],
        format_func=lambda x: next(opt["icon"] + "  " + opt["title"] for opt in analysis_options if opt["value"] == x),
        label_visibility="hidden"
    )
    
    selected_option = next(opt for opt in analysis_options if opt["value"] == analysis_mode)
    st.markdown(f"""
    <div class="mode-description">
        <div class="icon">{selected_option["icon"]}</div>
        <div class="title">{selected_option["title"]}</div>
        <div class="desc">{selected_option["description"]}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-stats">
        <div class="stats-header">📈 System Overview</div>
        <div class="stat-item">
            <span class="stat-label">🔒 Active Models</span>
            <span class="stat-value">1</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">⚡ System Status</span>
            <span class="stat-value" style="color: #10b981;">Online</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">🛡️ Security Level</span>
            <span class="stat-value" style="color: #3b82f6;">High</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">🚀 Version</span>
            <span class="stat-value">v2.1.0</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">📡 Last Update</span>
            <span class="stat-value" style="color: #10b981;">Active</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========== MAIN CONTENT ==========
st.markdown("""
<div class="main-header">
    <h1>🔒 XAI Cyber Monitor</h1>
    <p>Machine learning-driven network security system with explainable AI for real-time threat detection and traffic analysis</p>
</div>
""", unsafe_allow_html=True)

if analysis_mode == "Real-time URL Analysis":
    st.header("⚡ Instant URL Scanner")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        url_input = st.text_area(
            "Enter URLs (one per line):",
            height=150,
            placeholder="http://example.com/search?q=test\nhttp://malicious.com?q=<script>alert('xss')</script>"
        )
        
        if st.button("⚡ Scan URLs", type="primary"):
            if url_input:
                urls = [url.strip() for url in url_input.split('\n') if url.strip()]
                results = []
                
                progress_bar = st.progress(0)
                for i, url in enumerate(urls):
                    try:
                        features = extract_features(url)
                        if features is None or len(features) != len(FEATURE_NAMES):
                            st.error(f"Feature extraction failed for: {url}")
                            continue
                            
                        pred = st.session_state.model.predict([features])[0]
                        proba = st.session_state.model.predict_proba([features])[0]
                        traffic_type = st.session_state.traffic_analyzer.classify_traffic(url)
                        
                        results.append({
                            'URL': url,
                            'Status': 'Malicious' if pred else 'Safe',
                            'Confidence': f"{max(proba)*100:.1f}%",
                            'Traffic_Type': traffic_type,
                            'Risk_Score': proba[1] if len(proba) > 1 else 0
                        })
                        
                        with st.expander(f"Explanation for {url}", expanded=False):
                            shap_fig = show_shap_explanation(features)
                            st.plotly_chart(shap_fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error analyzing {url}: {str(e)}")
                        results.append({
                            'URL': url,
                            'Status': 'Error',
                            'Confidence': '0%',
                            'Traffic_Type': 'Unknown',
                            'Risk_Score': 0
                        })
                    
                    progress_bar.progress((i + 1) / len(urls))
                
                if results:
                    df_results = pd.DataFrame(results)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total URLs", len(urls))
                    with col2:
                        malicious_count = len(df_results[df_results['Status'] == 'Malicious'])
                        st.metric("🔴 Threats Detected", malicious_count)
                    with col3:
                        benign_count = len(df_results[df_results['Status'] == 'Safe'])
                        st.metric("🟢 Safe URLs", benign_count)
                    with col4:
                        st.metric("Threat Rate", f"{malicious_count/len(urls)*100:.1f}%")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.pie(df_results, names='Status', 
                                   title='Threat Distribution',
                                   color_discrete_map={'Malicious': '#ff4444', 'Safe': '#44ff44'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.bar(df_results.groupby('Traffic_Type').size().reset_index(name='Count'),
                                   x='Traffic_Type', y='Count', title='Traffic Classification')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(df_results, use_container_width=True)

elif analysis_mode == "Batch File Analysis":
    st.header("📦 Bulk File Processor")
    
    st.markdown("""
    <div class="upload-section">
        <h3 style="color: #3b82f6; margin-bottom: 1rem;">📤 Upload Your CSV File</h3>
        <p style="color: #4b5563; margin-bottom: 1rem;">Maximum file size: 200MB | Supported format: CSV</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"], help="CSV should contain a 'url' column. Maximum file size: 200MB")
    
    if uploaded_file is not None:
        try:
            file_size = uploaded_file.size / (1024 * 1024)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 File Size", f"{file_size:.2f} MB")
            with col2:
                st.metric("📊 Format", "CSV")
            with col3:
                st.metric("✅ Status", "Ready")
            
            if file_size > 200:
                st.error("⚠️ File size exceeds 200MB limit! Please use a smaller file.")
            else:
                if file_size > 50:
                    chunk_size = 1000
                    chunks = []
                    for chunk in pd.read_csv(uploaded_file, chunksize=chunk_size):
                        chunks.append(chunk)
                    df = pd.concat(chunks, ignore_index=True)
                else:
                    df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ Loaded {len(df):,} URLs successfully!")
                
                if 'url' not in df.columns:
                    st.error("❌ CSV must contain a 'url' column!")
                else:
                    if st.button("🚀 Process File", type="primary"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        results = []
                        total_urls = len(df)
                        batch_size = 100
                        
                        for i, url in enumerate(df['url']):
                            if i % batch_size == 0 or i == total_urls - 1:
                                status_text.text(f"🔄 Processing batch {(i//batch_size)+1} - URL {i+1:,}/{total_urls:,}")
                                progress_bar.progress((i+1)/total_urls)
                            
                            try:
                                features = extract_features(url)
                                pred = st.session_state.model.predict([features])[0]
                                proba = st.session_state.model.predict_proba([features])[0]
                                traffic_type = st.session_state.traffic_analyzer.classify_traffic(url)
                                
                                results.append({
                                    'URL': url,
                                    'Status': 'Malicious' if pred else 'Safe',
                                    'Confidence': f"{max(proba)*100:.1f}%",
                                    'Traffic_Type': traffic_type,
                                    'Risk_Score': proba[1] if len(proba) > 1 else 0
                                })
                                
                            except Exception as e:
                                st.warning(f"Error processing URL {url}: {str(e)}")
                                results.append({
                                    'URL': url,
                                    'Status': 'Error',
                                    'Confidence': '0%',
                                    'Traffic_Type': 'Unknown',
                                    'Risk_Score': 0
                                })
                        
                        status_text.text(f"✅ Processing complete! Displaying results...")
                        progress_bar.progress(1.0)
                        
                        df_results = pd.DataFrame(results)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total URLs", f"{len(df_results):,}")
                        with col2:
                            malicious_count = len(df_results[df_results['Status'] == 'Malicious'])
                            st.metric("🔴 Threats", f"{malicious_count:,}")
                        with col3:
                            benign_count = len(df_results[df_results['Status'] == 'Safe'])
                            st.metric("🟢 Safe", f"{benign_count:,}")
                        with col4:
                            st.metric("Detection Rate", f"{malicious_count/len(df_results)*100:.1f}%")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig = px.histogram(df_results, x='Risk_Score', color='Status',
                                             title='Risk Score Distribution')
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            fig = px.scatter(df_results, x='Risk_Score', y='Confidence', 
                                           color='Status',
                                           title='Risk vs Confidence Analysis')
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.subheader("🔗 Traffic Classification Analysis")
                        traffic_summary = df_results.groupby(['Traffic_Type', 'Status']).size().reset_index(name='Count')
                        fig = px.bar(traffic_summary, x='Traffic_Type', y='Count', 
                                   color='Status', title='Threats by Traffic Type')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.subheader("📊 Detailed Results")
                        st.dataframe(df_results, use_container_width=True)
                        
                        csv = df_results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="security_analysis_results.csv",
                            mime="text/csv"
                        )
                        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")

elif analysis_mode == "Model Performance":
    st.header("📊 Model Analytics")
    
    if st.button("🔄 Generate Analytics Report"):
        test_df = pd.read_csv("sample_http.csv")
        
        X_test = [extract_features(url) for url in test_df["url"]]
        y_true = test_df["label"].values
        y_pred = st.session_state.model.predict(X_test)
        y_prob = st.session_state.model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_true, y_pred) * 100
        precision = precision_score(y_true, y_pred) * 100
        recall = recall_score(y_true, y_pred) * 100
        f1 = f1_score(y_true, y_pred) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{accuracy:.1f}%")
        with col2:
            st.metric("Precision", f"{precision:.1f}%")
        with col3:
            st.metric("Recall", f"{recall:.1f}%")
        with col4:
            st.metric("F1-Score", f"{f1:.1f}%")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cm = confusion_matrix(y_true, y_pred)
            fig = px.imshow(cm, text_auto=True, aspect="auto",
                          title="Confusion Matrix",
                          labels=dict(x="Predicted", y="Actual"))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if len(y_prob.shape) > 1:
                fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1])
                roc_auc = auc(fpr, tpr)
                
                fig = px.area(
                    x=fpr, y=tpr,
                    title=f'ROC Curve (AUC = {roc_auc:.2f})',
                    labels=dict(x='False Positive Rate', y='True Positive Rate')
                )
                fig.add_shape(
                    type='line', line=dict(dash='dash'),
                    x0=0, x1=1, y0=0, y1=1
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🔍 Feature Importance")
        if hasattr(st.session_state.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': FEATURE_NAMES,
                'Importance': st.session_state.model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            importance_df = importance_df.tail(15)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', 
                        orientation='h', title='Top Feature Importance',
                        color='Importance',
                        color_continuous_scale='viridis')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Feature importance not available for this model type.")

elif analysis_mode == "Traffic Insights":
    st.header("🔗 Network Monitor")
    
    if st.button("📊 Generate Traffic Report"):
        sample_urls = [
            "http://example.com/api/data",
            "https://cdn.example.com/image.jpg",
            "http://mail.example.com/inbox",
            "https://video.example.com/stream",
            "http://attacker.com?q=<script>alert('xss')</script>",
            "http://bank.com/transfer?to='; DROP TABLE users;--",
        ]
        
        traffic_data = []
        for url in sample_urls * 20: 
            features = extract_features(url)
            pred = st.session_state.model.predict([features])[0]
            traffic_type = st.session_state.traffic_analyzer.classify_traffic(url)
            
            traffic_data.append({
                'URL': url,
                'Traffic_Type': traffic_type,
                'Threat_Status': 'Malicious' if pred else 'Benign',
                'Timestamp': pd.Timestamp.now() - pd.Timedelta(minutes=np.random.randint(0, 1440))
            })
        
        df_traffic = pd.DataFrame(traffic_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            traffic_counts = df_traffic['Traffic_Type'].value_counts()
            fig = px.pie(values=traffic_counts.values, names=traffic_counts.index,
                        title='Traffic Type Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            threat_by_type = df_traffic.groupby(['Traffic_Type', 'Threat_Status']).size().reset_index(name='Count')
            fig = px.bar(threat_by_type, x='Traffic_Type', y='Count', 
                        color='Threat_Status', title='Threats by Traffic Type')
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("⏰ Temporal Analysis")
        df_traffic['Hour'] = df_traffic['Timestamp'].dt.hour
        hourly_threats = df_traffic[df_traffic['Threat_Status'] == 'Malicious'].groupby('Hour').size().reset_index(name='Threats')
        
        fig = px.line(hourly_threats, x='Hour', y='Threats', 
                     title='Threat Detection by Hour')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📋 Traffic Log")
        st.dataframe(df_traffic.sort_values('Timestamp', ascending=False), use_container_width=True)

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
<div class="footer">
    <h3>🔒 XAI Cyber Monitor</h3>
    <p>Machine learning-driven network security system with explainable AI for real-time threat detection and traffic analysis</p>
    <p>✅ Feature 1: ML-based network traffic classification | ✅ Feature 2: Explainable threat detection system</p>
    <p style="margin-top: 1rem; opacity: 0.85;">Powered by advanced ML for real-time security</p>
</div>
""", unsafe_allow_html=True)