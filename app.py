import streamlit as st
import joblib
import json
import os
import sys
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(
    page_title="ReviewIQ · Sentiment Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = os.path.dirname(os.path.abspath(__file__))
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
sys.path.insert(0, BASE)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace !important;
    background-color: #080c10 !important;
    color: #cdd9e5 !important;
}
#MainMenu, footer { visibility: hidden; }
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="stSidebarHeader"] { display: none !important; }
button[data-testid="baseButton-headerNoPadding"] { display: none !important; }
.block-container { padding: 2rem 2.5rem !important; max-width: 1300px !important; }

section[data-testid="stSidebar"] {
    background-color: #0d1117 !important;
    border-right: 1px solid #1e2d3d !important;
}
section[data-testid="stSidebar"] * {
    font-family: 'JetBrains Mono', monospace !important;
}

div[data-testid="stSelectbox"] > div > div {
    background-color: #111820 !important;
    border: 1px solid #1e2d3d !important;
    color: #00f5d4 !important;
}

.stTextArea textarea {
    background-color: #111820 !important;
    border: 1px solid #1e2d3d !important;
    color: #cdd9e5 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stTextArea textarea:focus {
    border-color: #00f5d4 !important;
    box-shadow: none !important;
}

.stButton > button {
    background: transparent !important;
    border: 1px solid #00f5d4 !important;
    color: #00f5d4 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 2px !important;
    padding: 0.45rem 1.2rem !important;
}
.stButton > button:hover {
    background: rgba(0,245,212,0.08) !important;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    color: #e6edf3 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Artifacts ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    mdl = joblib.load(os.path.join(BASE, "models", "linear_svm_best.pkl"))
    vec = joblib.load(os.path.join(BASE, "data", "processed", "tfidf_vectorizer.pkl"))
    with open(os.path.join(BASE, "models", "label_mapping.json")) as f:
        lm = json.load(f)
    return mdl, vec, lm

model, vectorizer, label_mapping = load_artifacts()
from src.preprocessing import preprocess_text

def predict(text):
    clean  = preprocess_text(text)
    vector = vectorizer.transform([clean])
    pred   = model.predict(vector)[0]
    return label_mapping[str(pred)] if isinstance(pred, int) else pred

# ── Session state ──────────────────────────────────────────────────────────────
if "history"  not in st.session_state: st.session_state.history  = []
if "counts"   not in st.session_state: st.session_state.counts   = {"Positive":0,"Negative":0,"Neutral":0}

# ── Plotly base layout ─────────────────────────────────────────────────────────
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono", color="#4a5568", size=11),
    margin=dict(l=10,r=10,t=30,b=10),
)
GC = dict(gridcolor="#1e2d3d", zeroline=False)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.2rem 0 1rem; border-bottom:1px solid #1e2d3d; margin-bottom:1rem;">
        <span style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:#00f5d4;">Review</span><span style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:#4a5568;">IQ</span>
        <div style="font-size:0.6rem;color:#4a5568;letter-spacing:0.1em;margin-top:0.2rem;">AI SENTIMENT INTELLIGENCE</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.selectbox("PAGE", ["Home","Predict","Dashboard","Model Comparison","About"], label_visibility="collapsed")

    st.markdown("""
    <div style="margin-top:1rem;padding:0.8rem;background:#111820;border:1px solid rgba(0,245,212,0.15);font-size:0.65rem;color:#4a5568;line-height:1.8;">
        MODEL<br>
        <span style="color:#00f5d4;font-size:0.8rem;">Linear SVM</span><br><br>
        ACCURACY<br>
        <span style="color:#00f5d4;font-size:0.8rem;">84.6%</span><br><br>
        FEATURES<br>
        <span style="color:#00f5d4;font-size:0.8rem;">TF-IDF · 5k</span>
    </div>
    """, unsafe_allow_html=True)

    total_preds = sum(st.session_state.counts.values())
    st.markdown(f"""
    <div style="margin-top:1rem;font-size:0.6rem;color:#4a5568;letter-spacing:0.1em;text-transform:uppercase;">
        Session Predictions<br>
        <span style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:#00f5d4;">{total_preds}</span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════════
if page == "Home":
    st.markdown('<h1 style="font-size:2.2rem;margin-bottom:0.2rem;">AI Customer Review Intelligence</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#4a5568;font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:2rem;">End-to-end NLP sentiment analysis · Phase 2 complete</p>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    def mcard(col, val, label, color="#00f5d4"):
        col.markdown(f"""
        <div style="border:1px solid #1e2d3d;background:#0d1117;padding:1.2rem;text-align:center;">
            <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:{color};line-height:1;">{val}</div>
            <div style="font-size:0.6rem;color:#4a5568;letter-spacing:0.12em;text-transform:uppercase;margin-top:0.4rem;">{label}</div>
        </div>""", unsafe_allow_html=True)
    mcard(c1,"84.6%","Model Accuracy")
    mcard(c2,"5","Models Benchmarked","#ffb703")
    mcard(c3,"5k","TF-IDF Features")
    mcard(c4,"3","Sentiment Classes","#4a5568")

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([3,2])

    with left:
        st.markdown('<p style="font-size:0.6rem;color:#4a5568;letter-spacing:0.15em;text-transform:uppercase;border-bottom:1px solid #1e2d3d;padding-bottom:0.4rem;">System Overview</p>', unsafe_allow_html=True)
        st.markdown("""
        <div style="border:1px solid #1e2d3d;background:#0d1117;padding:1.2rem 1.4rem;border-left:3px solid #00f5d4;">
            <div style="font-family:'Syne',sans-serif;font-weight:700;color:#e6edf3;margin-bottom:0.6rem;">What this system does</div>
            <div style="font-size:0.76rem;line-height:1.8;color:#cdd9e5;">
                ReviewIQ classifies customer reviews into
                <span style="color:#00f5d4;">Positive</span>,
                <span style="color:#ff4d6d;">Negative</span>, or
                <span style="color:#ffb703;">Neutral</span> sentiment in real-time.
                Built on a classical ML stack with TF-IDF vectorization and a tuned Linear SVM as its production classifier.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p style="font-size:0.6rem;color:#4a5568;letter-spacing:0.15em;text-transform:uppercase;border-bottom:1px solid #1e2d3d;padding-bottom:0.4rem;margin-top:1.5rem;">Inference Pipeline</p>', unsafe_allow_html=True)
        for num, title, desc in [
            ("01","Raw Text Input","Customer review string"),
            ("02","Preprocessing","Lowercase · Negations · Lemmatize"),
            ("03","TF-IDF Vectorization","Unigrams + Bigrams · 5k features"),
            ("04","Linear SVM Inference","Tuned · LinearSVC"),
            ("05","Sentiment Output","Positive / Negative / Neutral"),
        ]:
            st.markdown(f"""
            <div style="display:flex;gap:1rem;align-items:flex-start;margin-bottom:0.5rem;">
                <div style="font-size:0.6rem;color:#4a5568;padding-top:0.3rem;min-width:16px;">{num}</div>
                <div style="flex:1;padding:0.55rem 0.8rem;border:1px solid #1e2d3d;background:#111820;">
                    <div style="font-size:0.74rem;color:#e6edf3;">{title}</div>
                    <div style="font-size:0.62rem;color:#4a5568;">{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    with right:
        st.markdown('<p style="font-size:0.6rem;color:#4a5568;letter-spacing:0.15em;text-transform:uppercase;border-bottom:1px solid #1e2d3d;padding-bottom:0.4rem;">Tech Stack</p>', unsafe_allow_html=True)
        for k,v in [("ML Framework","scikit-learn"),("Classifier","LinearSVC (tuned)"),("Features","TF-IDF · ngram(1,2)"),("NLP","NLTK · lemmatization"),("Boosting","XGBoost"),("Deployment","Streamlit"),("Environment","Conda · review_ai"),("Version Control","Git · GitHub")]:
            st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:0.4rem 0;border-bottom:1px solid rgba(30,45,61,0.5);font-size:0.72rem;">
                <span style="color:#4a5568;">{k}</span><span style="color:#e6edf3;">{v}</span></div>""", unsafe_allow_html=True)

        st.markdown('<p style="font-size:0.6rem;color:#4a5568;letter-spacing:0.15em;text-transform:uppercase;border-bottom:1px solid #1e2d3d;padding-bottom:0.4rem;margin-top:1.5rem;">Key Findings</p>', unsafe_allow_html=True)
        for color, text in [("#00f5d4","Linear models dominate sparse TF-IDF"),("#ffb703","Neutral class hardest to classify"),("#ffb703","Negations need special handling"),("#00f5d4","SVM outperforms tree-based models"),("#4a5568","Tuning yields ~1% gain over baseline")]:
            st.markdown(f'<div style="font-size:0.72rem;color:{color};padding:0.3rem 0;border-bottom:1px solid rgba(30,45,61,0.4);">▸ {text}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PREDICT
# ═══════════════════════════════════════════════
elif page == "Predict":
    st.markdown('<h1 style="font-size:2.2rem;margin-bottom:0.2rem;">Sentiment Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#4a5568;font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:2rem;">Real-time inference · Linear SVM · TF-IDF</p>', unsafe_allow_html=True)

    col_main, col_side = st.columns([3,2])

    with col_main:
        examples = [
            "Absolutely love this product!",
            "It was okay, nothing special.",
            "Terrible quality, waste of money.",
            "Decent for the price I guess.",
            "I was not happy with this at all.",
        ]
        st.markdown('<p style="font-size:0.6rem;color:#4a5568;letter-spacing:0.15em;text-transform:uppercase;border-bottom:1px solid #1e2d3d;padding-bottom:0.4rem;">Quick Examples</p>', unsafe_allow_html=True)
        ecols = st.columns(5)
        selected = None
        for i,(ec,ex) in enumerate(zip(ecols,examples)):
            with ec:
                if st.button(f"#{i+1}", key=f"ex{i}", help=ex):
                    selected = ex

        st.markdown('<p style="font-size:0.6rem;color:#4a5568;letter-spacing:0.15em;text-transform:uppercase;border-bottom:1px solid #1e2d3d;padding-bottom:0.4rem;margin-top:1rem;">Review Input</p>', unsafe_allow_html=True)
        user_input = st.text_area("input", value=selected or "", placeholder="Enter a customer review...", height=130, label_visibility="collapsed")

        if st.button("▶  Run Inference"):
            text = user_input.strip()
            if not text:
                st.warning("Please enter a review.")
            else:
                sentiment = predict(text)
                s = sentiment.lower()
                st.session_state.counts[sentiment] = st.session_state.counts.get(sentiment,0) + 1
                st.session_state.history.insert(0, {"text":text,"sentiment":sentiment,"time":datetime.now().strftime("%H:%M:%S")})

                color = {"positive":"#00f5d4","negative":"#ff4d6d","neutral":"#ffb703"}.get(s,"#00f5d4")
                glow  = {"positive":"rgba(0,245,212,0.06)","negative":"rgba(255,77,109,0.06)","neutral":"rgba(255,183,3,0.06)"}.get(s,"rgba(0,245,212,0.06)")
                icon  = {"positive":"◈","negative":"✕","neutral":"◌"}.get(s,"◈")

                st.markdown(f"""
                <div style="margin-top:1.5rem;border:1px solid #1e2d3d;padding:2.5rem;text-align:center;background:radial-gradient(ellipse at center,{glow} 0%,transparent 70%);">
                    <div style="font-size:0.6rem;color:#4a5568;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:1rem;">Inference complete · {datetime.now().strftime("%H:%M:%S")}</div>
                    <div style="font-family:'Syne',sans-serif;font-size:3rem;font-weight:800;color:{color};letter-spacing:-0.02em;">{icon} {sentiment.upper()}</div>
                    <div style="font-size:0.6rem;color:#4a5568;letter-spacing:0.12em;text-transform:uppercase;margin-top:0.6rem;">Linear SVM · TF-IDF · ~84.6% accuracy</div>
                </div>
                """, unsafe_allow_html=True)

                clean = preprocess_text(text)
                st.markdown(f"""
                <div style="margin-top:0.8rem;padding:0.8rem;background:#111820;border:1px solid #1e2d3d;font-size:0.7rem;">
                    <span style="color:#4a5568;font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;">Preprocessed tokens →</span><br>
                    <span style="color:#00f5d4;">{clean}</span>
                </div>
                """, unsafe_allow_html=True)

    with col_side:
        st.markdown('<p style="font-size:0.6rem;color:#4a5568;letter-spacing:0.15em;text-transform:uppercase;border-bottom:1px solid #1e2d3d;padding-bottom:0.4rem;">Session Stats</p>', unsafe_allow_html=True)
        total = sum(st.session_state.counts.values())
        for label, color in [("Positive","#00f5d4"),("Negative","#ff4d6d"),("Neutral","#ffb703")]:
            val = st.session_state.counts.get(label,0)
            pct = round(val/total*100) if total > 0 else 0
            st.markdown(f"""
            <div style="margin-bottom:0.8rem;">
                <div style="display:flex;justify-content:space-between;font-size:0.7rem;margin-bottom:0.3rem;">
                    <span style="color:{color};">{label}</span>
                    <span style="color:#4a5568;">{val} · {pct}%</span>
                </div>
                <div style="height:2px;background:#1e2d3d;">
                    <div style="height:2px;width:{pct}%;background:{color};"></div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<p style="font-size:0.6rem;color:#4a5568;letter-spacing:0.15em;text-transform:uppercase;border-bottom:1px solid #1e2d3d;padding-bottom:0.4rem;margin-top:1.5rem;">Prediction History</p>', unsafe_allow_html=True)
        if not st.session_state.history:
            st.markdown('<div style="font-size:0.72rem;color:#4a5568;">No predictions yet.</div>', unsafe_allow_html=True)
        else:
            cmap = {"Positive":"#00f5d4","Negative":"#ff4d6d","Neutral":"#ffb703"}
            for item in st.session_state.history[:12]:
                c = cmap.get(item["sentiment"],"#fff")
                short = item["text"][:40]+"…" if len(item["text"])>40 else item["text"]
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:0.6rem;padding:0.4rem 0;border-bottom:1px solid rgba(30,45,61,0.4);font-size:0.7rem;">
                    <span style="color:{c};font-size:0.55rem;">■</span>
                    <span style="flex:1;color:#cdd9e5;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;">{short}</span>
                    <span style="color:#4a5568;font-size:0.62rem;">{item["time"]}</span>
                </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════
elif page == "Dashboard":
    st.markdown('<h1 style="font-size:2.2rem;margin-bottom:0.2rem;">Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#4a5568;font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:2rem;">Session analytics · Model performance overview</p>', unsafe_allow_html=True)

    total = sum(st.session_state.counts.values())
    pos = st.session_state.counts.get("Positive",0)
    neg = st.session_state.counts.get("Negative",0)
    neu = st.session_state.counts.get("Neutral",0)

    c1,c2,c3,c4,c5 = st.columns(5)
    for col,(val,label,color) in zip([c1,c2,c3,c4,c5],[
        (str(total),"Total Predictions","#00f5d4"),
        (str(pos),"Positive","#00f5d4"),
        (str(neg),"Negative","#ff4d6d"),
        (str(neu),"Neutral","#ffb703"),
        ("84.6%","Best Accuracy","#00f5d4"),
    ]):
        col.markdown(f"""
        <div style="border:1px solid #1e2d3d;background:#0d1117;padding:1rem;text-align:center;">
            <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;color:{color};line-height:1;">{val}</div>
            <div style="font-size:0.58rem;color:#4a5568;letter-spacing:0.12em;text-transform:uppercase;margin-top:0.3rem;">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns(2)

    with cl:
        st.markdown('<p style="font-size:0.6rem;color:#4a5568;letter-spacing:0.15em;text-transform:uppercase;border-bottom:1px solid #1e2d3d;padding-bottom:0.4rem;">Session Sentiment Distribution</p>', unsafe_allow_html=True)
        if total > 0:
            fig = go.Figure(go.Pie(
                labels=["Positive","Negative","Neutral"], values=[pos,neg,neu], hole=0.65,
                marker=dict(colors=["#00f5d4","#ff4d6d","#ffb703"], line=dict(color="#080c10",width=3)),
                textinfo="percent", textfont=dict(family="JetBrains Mono",size=11,color="#cdd9e5"),
            ))
            fig.update_layout(**PL, height=260, showlegend=True, legend=dict(font=dict(color="#4a5568",size=10)))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown('<div style="font-size:0.75rem;color:#4a5568;padding:2rem 0;">Make predictions first.</div>', unsafe_allow_html=True)

    with cr:
        st.markdown('<p style="font-size:0.6rem;color:#4a5568;letter-spacing:0.15em;text-transform:uppercase;border-bottom:1px solid #1e2d3d;padding-bottom:0.4rem;">Model Accuracy Comparison</p>', unsafe_allow_html=True)
        mnames = ["Logistic Reg","Linear SVM","Naive Bayes","Random Forest","XGBoost"]
        maccs  = [76.5, 84.6, 81.8, 77.3, 79.2]
        mcols  = ["#1e2d3d","#00f5d4","#1e2d3d","#1e2d3d","#1e2d3d"]
        fig2 = go.Figure(go.Bar(
            x=maccs, y=mnames, orientation="h",
            marker=dict(color=mcols),
            text=[f"{a}%" for a in maccs], textposition="outside",
            textfont=dict(color="#cdd9e5",size=10,family="JetBrains Mono"),
        ))
        fig2.update_layout(**PL, height=260, xaxis=dict(**GC, range=[60,90]), yaxis=dict(**GC))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<p style="font-size:0.6rem;color:#4a5568;letter-spacing:0.15em;text-transform:uppercase;border-bottom:1px solid #1e2d3d;padding-bottom:0.4rem;">Prediction Timeline</p>', unsafe_allow_html=True)
    if st.session_state.history:
        cmap_num = {"Positive":1,"Negative":-1,"Neutral":0}
        cmap_hex = {"Positive":"#00f5d4","Negative":"#ff4d6d","Neutral":"#ffb703"}
        hist = list(reversed(st.session_state.history))
        fig3 = go.Figure()
        for sent, color in cmap_hex.items():
            idx = [i for i,h in enumerate(hist) if h["sentiment"]==sent]
            fig3.add_trace(go.Scatter(
                x=idx, y=[cmap_num[sent]]*len(idx), mode="markers", name=sent,
                marker=dict(color=color, size=10, symbol="square"),
            ))
        fig3.update_layout(**PL, height=160,
            yaxis=dict(**GC, tickvals=[-1,0,1], ticktext=["Neg","Neu","Pos"]),
            xaxis=dict(**GC),
            showlegend=True, legend=dict(font=dict(color="#4a5568",size=10)))
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.markdown('<div style="font-size:0.75rem;color:#4a5568;">Make predictions to see timeline.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# MODEL COMPARISON
# ═══════════════════════════════════════════════
elif page == "Model Comparison":
    st.markdown('<h1 style="font-size:2.2rem;margin-bottom:0.2rem;">Model Benchmarking</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#4a5568;font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:2rem;">5 models · Same TF-IDF feature space · Stratified evaluation</p>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,(val,label,color) in zip([c1,c2,c3,c4],[
        ("SVM","Best Model","#00f5d4"),
        ("84.6%","Top Accuracy","#00f5d4"),
        ("0.36","Best Neutral F1","#ffb703"),
        ("8","Runs Evaluated","#4a5568"),
    ]):
        col.markdown(f"""
        <div style="border:1px solid #1e2d3d;background:#0d1117;padding:1rem;text-align:center;">
            <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;color:{color};line-height:1;">{val}</div>
            <div style="font-size:0.58rem;color:#4a5568;letter-spacing:0.12em;text-transform:uppercase;margin-top:0.3rem;">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    models_data = [
        ("Logistic Regression",        0.7651, 0.7965, 0.3293, False),
        ("Logistic Regression (Tuned)", 0.7651, 0.7963, 0.3287, False),
        ("Linear SVM",                  0.8305, 0.8362, 0.3459, False),
        ("Linear SVM (Tuned)",          0.8463, 0.8418, 0.3599, True),
        ("Naive Bayes",                 0.8183, 0.7643, 0.0247, False),
        ("Naive Bayes (Tuned)",         0.8196, 0.7672, 0.0343, False),
        ("Random Forest",               0.7727, 0.7965, 0.3293, False),
        ("XGBoost",                     0.7917, 0.7116, 0.0050, False),
    ]

    st.markdown('<p style="font-size:0.6rem;color:#4a5568;letter-spacing:0.15em;text-transform:uppercase;border-bottom:1px solid #1e2d3d;padding-bottom:0.4rem;">Full Benchmark Results</p>', unsafe_allow_html=True)
    rows = ""
    for name, acc, f1w, f1n, best in models_data:
        rc = "color:#00f5d4;" if best else ""
        tag = ' <span style="font-size:0.6rem;border:1px solid #00f5d4;color:#00f5d4;padding:0.1rem 0.4rem;letter-spacing:0.08em;">DEPLOYED</span>' if best else ""
        rows += f'<tr style="{rc}"><td style="padding:0.6rem 0.8rem;border-bottom:1px solid rgba(30,45,61,0.5);">{name}{tag}</td><td style="padding:0.6rem 0.8rem;border-bottom:1px solid rgba(30,45,61,0.5);">{acc:.4f}</td><td style="padding:0.6rem 0.8rem;border-bottom:1px solid rgba(30,45,61,0.5);">{f1w:.4f}</td><td style="padding:0.6rem 0.8rem;border-bottom:1px solid rgba(30,45,61,0.5);">{f1n:.4f}</td></tr>'

    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.76rem;">
        <thead><tr style="background:#111820;">
            <th style="text-align:left;padding:0.6rem 0.8rem;color:#4a5568;font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #1e2d3d;font-weight:400;">Model</th>
            <th style="text-align:left;padding:0.6rem 0.8rem;color:#4a5568;font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #1e2d3d;font-weight:400;">Accuracy</th>
            <th style="text-align:left;padding:0.6rem 0.8rem;color:#4a5568;font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #1e2d3d;font-weight:400;">F1 Weighted</th>
            <th style="text-align:left;padding:0.6rem 0.8rem;color:#4a5568;font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;border-bottom:1px solid #1e2d3d;font-weight:400;">F1 Neutral</th>
        </tr></thead>
        <tbody style="color:#cdd9e5;">{rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns(2)
    names = [m[0].replace(" (Tuned)","*") for m in models_data]

    with cl:
        st.markdown('<p style="font-size:0.6rem;color:#4a5568;letter-spacing:0.15em;text-transform:uppercase;border-bottom:1px solid #1e2d3d;padding-bottom:0.4rem;">Accuracy vs F1 Weighted</p>', unsafe_allow_html=True)
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(name="Accuracy",    x=names, y=[m[1] for m in models_data], marker_color="#00f5d4"))
        fig4.add_trace(go.Bar(name="F1 Weighted", x=names, y=[m[2] for m in models_data], marker_color="#1e2d3d"))
        fig4.update_layout(**PL, height=280, barmode="group", xaxis=dict(**GC, tickangle=-30, tickfont=dict(size=9)), yaxis=dict(**GC, range=[0.6,0.9]), legend=dict(font=dict(color="#4a5568",size=10)))
        st.plotly_chart(fig4, use_container_width=True)

    with cr:
        st.markdown('<p style="font-size:0.6rem;color:#4a5568;letter-spacing:0.15em;text-transform:uppercase;border-bottom:1px solid #1e2d3d;padding-bottom:0.4rem;">Neutral Class F1 (hardest class)</p>', unsafe_allow_html=True)
        fig5 = go.Figure(go.Bar(
            x=names, y=[m[3] for m in models_data],
            marker_color=["#00f5d4" if m[4] else "#1e2d3d" for m in models_data],
            text=[f"{m[3]:.3f}" for m in models_data], textposition="outside",
            textfont=dict(color="#cdd9e5",size=9,family="JetBrains Mono"),
        ))
        fig5.update_layout(**PL, height=280, xaxis=dict(**GC, tickangle=-30, tickfont=dict(size=9)), yaxis=dict(**GC, range=[0,0.5]))
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("""
    <div style="border:1px solid #1e2d3d;background:#0d1117;padding:1.2rem 1.4rem;border-left:3px solid #ffb703;margin-top:1rem;">
        <div style="font-size:0.6rem;color:#ffb703;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem;">⚑ Key ML Insight</div>
        <div style="font-size:0.76rem;line-height:1.8;color:#cdd9e5;">
            Linear models (SVM, Logistic Regression) consistently outperform tree-based models on sparse TF-IDF feature spaces.
            High-dimensional sparse vectors favour linear decision boundaries. The Neutral class remains the hardest to classify
            across all models — a known challenge in 3-class sentiment analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# ABOUT
# ═══════════════════════════════════════════════
elif page == "About":
    st.markdown('<h1 style="font-size:2.2rem;margin-bottom:0.2rem;">About This Project</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#4a5568;font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:2rem;">AI Customer Review Intelligence · Phase 2 Complete</p>', unsafe_allow_html=True)

    left, right = st.columns([3,2])
    with left:
        st.markdown('<p style="font-size:0.6rem;color:#4a5568;letter-spacing:0.15em;text-transform:uppercase;border-bottom:1px solid #1e2d3d;padding-bottom:0.4rem;">Project Summary</p>', unsafe_allow_html=True)
        st.markdown("""
        <div style="border:1px solid #1e2d3d;background:#0d1117;padding:1.2rem 1.4rem;border-left:3px solid #00f5d4;">
            <div style="font-size:0.76rem;line-height:1.9;color:#cdd9e5;">
                A complete end-to-end NLP ML system built from scratch. Processes Amazon customer reviews,
                extracts features via TF-IDF vectorization, benchmarks five classical ML models, and deploys
                the best-performing model (Tuned Linear SVM) as a real-time prediction system.<br><br>
                Demonstrates production-grade ML engineering: modular architecture, proper train/test separation,
                rigorous evaluation, artifact management, and deployment.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p style="font-size:0.6rem;color:#4a5568;letter-spacing:0.15em;text-transform:uppercase;border-bottom:1px solid #1e2d3d;padding-bottom:0.4rem;margin-top:1.5rem;">Project Architecture</p>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#111820;border:1px solid #1e2d3d;padding:1rem;font-size:0.72rem;line-height:2;color:#4a5568;">
            <span style="color:#00f5d4;">ai_review_intelligence/</span><br>
            &nbsp;&nbsp;├── <span style="color:#cdd9e5;">app.py</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#4a5568;"># Streamlit app</span><br>
            &nbsp;&nbsp;├── <span style="color:#cdd9e5;">src/</span><br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── <span style="color:#cdd9e5;">preprocessing.py</span> &nbsp;&nbsp;<span style="color:#4a5568;"># NLP pipeline</span><br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── <span style="color:#cdd9e5;">feature_engineering.py</span><br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── <span style="color:#cdd9e5;">train_ml.py</span><br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── <span style="color:#cdd9e5;">inference.py</span><br>
            &nbsp;&nbsp;├── <span style="color:#cdd9e5;">models/</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#4a5568;"># .pkl artifacts</span><br>
            &nbsp;&nbsp;├── <span style="color:#cdd9e5;">data/processed/</span> &nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#4a5568;"># TF-IDF + splits</span><br>
            &nbsp;&nbsp;├── <span style="color:#cdd9e5;">results/</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#4a5568;"># Metrics + figures</span><br>
            &nbsp;&nbsp;└── <span style="color:#cdd9e5;">notebooks/</span> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:#4a5568;"># EDA + experiments</span>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<p style="font-size:0.6rem;color:#4a5568;letter-spacing:0.15em;text-transform:uppercase;border-bottom:1px solid #1e2d3d;padding-bottom:0.4rem;">Dataset</p>', unsafe_allow_html=True)
        st.markdown("""
        <div style="border:1px solid #1e2d3d;background:#0d1117;padding:1rem;border-left:3px solid #4a5568;">
            <div style="font-size:0.74rem;font-weight:600;color:#e6edf3;margin-bottom:0.5rem;">Amazon Fine Food Reviews</div>
            <div style="font-size:0.7rem;color:#4a5568;line-height:1.9;">
                Source: Kaggle<br>Domain: Food product reviews<br>
                Classes: Positive · Neutral · Negative<br>Split: Stratified 80/20
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p style="font-size:0.6rem;color:#4a5568;letter-spacing:0.15em;text-transform:uppercase;border-bottom:1px solid #1e2d3d;padding-bottom:0.4rem;margin-top:1.5rem;">Evaluation Metrics</p>', unsafe_allow_html=True)
        for m in ["Accuracy","Precision (weighted)","Recall (weighted)","F1-score (weighted)","F1 per class","ROC-AUC","Confusion Matrix","Learning Curve","Error Analysis"]:
            st.markdown(f'<div style="font-size:0.72rem;color:#cdd9e5;padding:0.3rem 0;border-bottom:1px solid rgba(30,45,61,0.4);">▸ {m}</div>', unsafe_allow_html=True)

        st.markdown('<p style="font-size:0.6rem;color:#4a5568;letter-spacing:0.15em;text-transform:uppercase;border-bottom:1px solid #1e2d3d;padding-bottom:0.4rem;margin-top:1.5rem;">Known Limitations</p>', unsafe_allow_html=True)
        for text in ["Neutral class F1 ~0.36","TF-IDF loses word order","No deep learning (by design)","Classical NLP negation limits"]:
            st.markdown(f'<div style="font-size:0.72rem;color:#ffb703;padding:0.3rem 0;border-bottom:1px solid rgba(30,45,61,0.4);">⚠ {text}</div>', unsafe_allow_html=True)