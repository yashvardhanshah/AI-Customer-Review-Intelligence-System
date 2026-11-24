import streamlit as st
import joblib
import json
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(
    page_title="ReviewIQ · Understand how customers feel",
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
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] {font-family:'Inter',sans-serif !important; background-color:#080c10 !important; color:#cdd9e5 !important; font-size:16px;}
#MainMenu, footer { visibility: hidden; }
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="stSidebarHeader"] { display: none !important; }
button[data-testid="baseButton-headerNoPadding"] { display: none !important; }
.block-container { padding: 2rem 2.5rem !important; max-width: 1200px !important; }
section[data-testid="stSidebar"] { background-color:#0d1117 !important; border-right:1px solid #1e2d3d !important; }
section[data-testid="stSidebar"] * { font-family:'Inter',sans-serif !important; }
div[data-testid="stSelectbox"] > div > div { background-color:#111820 !important; border:1px solid #1e2d3d !important; color:#00f5d4 !important; }
.stTextArea textarea { background-color:#111820 !important; border:1px solid #1e2d3d !important; color:#e6edf3 !important; font-family:'Inter',sans-serif !important; font-size:1.05rem !important; line-height:1.6 !important; border-radius:10px !important; padding:1rem !important; }
.stTextArea textarea:focus { border-color:#00f5d4 !important; box-shadow:0 0 0 2px rgba(0,245,212,0.15) !important; }
.stButton > button { background:transparent !important; border:1px solid #1e2d3d !important; color:#cdd9e5 !important; font-size:.95rem !important; border-radius:10px !important; padding:.5rem 1rem !important; transition:all .15s; }
.stButton > button:hover { border-color:#00f5d4 !important; color:#00f5d4 !important; background:rgba(0,245,212,0.06) !important; }
.stButton > button[kind="primary"] { background:#00f5d4 !important; border:1px solid #00f5d4 !important; color:#06201c !important; font-weight:600 !important; }
.stButton > button[kind="primary"] p { color:#06201c !important; }
.stButton > button[kind="primary"]:hover { background:#33f7de !important; color:#06201c !important; box-shadow:0 0 18px rgba(0,245,212,0.3); }
h1,h2,h3 { font-family:'Syne',sans-serif !important; color:#e6edf3 !important; }
.title { font-family:'Syne',sans-serif; font-size:2.3rem; font-weight:800; color:#e6edf3; letter-spacing:-.02em; margin-bottom:.3rem; }
.sub { color:#8b98a9; font-size:1.05rem; margin-bottom:1.8rem; line-height:1.55; }
.lbl { color:#8b98a9; font-size:.85rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase; border-bottom:1px solid #1e2d3d; padding-bottom:.5rem; margin:1.6rem 0 1rem; }
.card { background:#0d1117; border:1px solid #1e2d3d; border-radius:12px; padding:1.3rem 1.5rem; margin-bottom:1rem; line-height:1.65; }
.card.acc { border-left:3px solid #00f5d4; }
.card h4 { font-family:'Syne',sans-serif; font-size:1.1rem; color:#e6edf3; margin:0 0 .5rem; }
.card p, .card li { color:#cdd9e5; font-size:1rem; margin:0 0 .4rem; }
.kpi { background:#0d1117; border:1px solid #1e2d3d; border-radius:12px; padding:1.2rem; text-align:center; }
.kpi b { display:block; font-family:'Syne',sans-serif; font-size:2.1rem; font-weight:800; line-height:1.1; }
.kpi span { color:#8b98a9; font-size:.9rem; }
.step { display:flex; gap:1rem; align-items:flex-start; margin-bottom:.8rem; }
.step i { font-style:normal; min-width:2rem; height:2rem; border-radius:50%; background:rgba(0,245,212,.12); color:#00f5d4; font-weight:700; display:flex; align-items:center; justify-content:center; }
.step div { color:#cdd9e5; line-height:1.55; } .step b { color:#e6edf3; }
.res { border:1px solid #1e2d3d; border-radius:14px; padding:1.8rem; background:#0d1117; animation:fade .35s ease; }
@keyframes fade { from {opacity:0; transform:translateY(8px);} to {opacity:1; transform:none;} }
.verdict { font-family:'Syne',sans-serif; font-size:2.6rem; font-weight:800; line-height:1.1; }
.chip { display:inline-block; padding:.3rem .8rem; border-radius:999px; font-size:.88rem; font-weight:600; }
.bar { height:10px; border-radius:999px; background:#1a2430; overflow:hidden; } .bar > div { height:100%; border-radius:999px; }
.row { display:flex; justify-content:space-between; margin:.9rem 0 .35rem; font-size:1rem; }
.why { background:#111820; border:1px solid #1e2d3d; border-radius:10px; padding:1rem 1.1rem; line-height:1.65; }
.note { color:#8b98a9; font-size:.85rem; margin-top:.9rem; }
.item { display:flex; gap:.8rem; align-items:center; padding:.65rem 0; border-bottom:1px solid #1a2430; font-size:.95rem; }
.item .t { flex:1; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; } .item .m { color:#8b98a9; font-size:.8rem; }
table.t { width:100%; border-collapse:collapse; font-size:.9rem; } table.t th { text-align:left; color:#8b98a9; font-weight:600; font-size:.8rem; padding:.6rem .7rem; border-bottom:1px solid #1e2d3d; } table.t td { padding:.6rem .7rem; border-bottom:1px solid #1a2430; }
@media (max-width:700px) { .block-container { padding:1rem !important; } .title { font-size:1.8rem; } .verdict { font-size:2rem; } }
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

# ── Results (read from results/metrics, so the app can never drift from the notebooks) ──
DEPLOYED = "Linear SVM (Tuned)"
BASELINE = "Always predict"
# Dataset funnel, copied from the notebook 01 / 02 outputs
DATASET = {"raw": 568_454, "dedup": 393_579, "sample": 157_432, "train": 125_945, "test": 31_487}
RESULT_COLS = {"Model", "Accuracy", "F1 (Macro)", "F1 (Weighted)", "F1 (Neutral)", "ROC-AUC (Macro)"}


def load_results():
    path = os.path.join(BASE, "results", "metrics", "model_comparison.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if not RESULT_COLS <= set(df.columns) or not (df["Model"] == DEPLOYED).any():
        return None
    return df


def load_test_support():
    path = os.path.join(BASE, "results", "metrics", "linear_svm_best_report.csv")
    if not os.path.exists(path):
        return None
    rep = pd.read_csv(path, index_col=0)
    return {c: int(rep.loc[c, "support"]) for c in ["Negative", "Neutral", "Positive"]}


@st.cache_data
def top_terms(n=12):
    names = vectorizer.get_feature_names_out()
    return {cls: [(str(names[j]), float(model.coef_[i][j])) for j in np.argsort(model.coef_[i])[::-1][:n]]
            for i, cls in enumerate(model.classes_)}


def explain(text, sentiment, k=6):
    """Terms pushing the SVM toward / away from the predicted class (coefficient x tf-idf value)."""
    vec = vectorizer.transform([preprocess_text(text)])
    names = vectorizer.get_feature_names_out()
    coefs = model.coef_[list(model.classes_).index(sentiment)]
    contrib = sorted(((str(names[j]), float(vec[0, j] * coefs[j])) for j in vec.nonzero()[1]),
                     key=lambda t: t[1], reverse=True)
    return [c for c in contrib if c[1] > 0][:k], [c for c in contrib[::-1] if c[1] < 0][:k]


results = load_results()
test_support = load_test_support()
n_features = len(vectorizer.vocabulary_)
if results is not None:
    dep = results[results["Model"] == DEPLOYED].iloc[0]
    is_base = results["Model"].str.startswith(BASELINE)
    base_acc = float(results.loc[is_base, "Accuracy"].iloc[0]) if is_base.any() else None
    n_models = results.loc[~is_base, "Model"].str.replace(" (Tuned)", "", regex=False).nunique()
    acc_txt = f"{dep['Accuracy'] * 100:.1f}%"
    svm_plain = results.loc[results["Model"] == "Linear SVM", "Accuracy"]
    tune_gain = (dep["Accuracy"] - float(svm_plain.iloc[0])) * 100 if len(svm_plain) else None
else:
    dep = base_acc = tune_gain = None
    n_models, acc_txt = 5, "n/a"

from html import escape

# ── Plain-language helpers (driven by the real model output) ──────────────────
STYLE = {"Positive": ("#00f5d4", "rgba(0,245,212,0.12)", "😊"), "Neutral": ("#ffb703", "rgba(255,183,3,0.12)", "😐"), "Negative": ("#ff4d6d", "rgba(255,77,109,0.12)", "😞")}
CONFIDENCE_SHARPNESS = 2.0   # display scaling of the model's scores: an estimate, not a calibrated probability
EXAMPLES = {"😊 Loved it": "Absolutely loved it! Fresh, delicious and it arrived fast. I'll definitely order again.",
            "👍 Pretty good": "Pretty good overall. Tasty and good value, though the packaging could be better.",
            "😐 It's okay": "It was okay. Nothing special, but not bad either for the price.",
            "👎 Not happy": "I am not satisfied. It arrived late and the quality was not what I expected.",
            "😞 Terrible": "I didn't like it at all. The taste was awful and it felt like a waste of money."}
PAGES = ["Home", "Analyze", "Dashboard", "Accuracy", "About"]


def analyze(text):
    label = predict(text)                                   # existing backend, unchanged
    scores = model.decision_function(vectorizer.transform([preprocess_text(text)]))[0] * CONFIDENCE_SHARPNESS
    p = np.exp(scores - scores.max()); p /= p.sum()
    pro, con = explain(text, label)
    return label, dict(zip(model.classes_, p)), pro, con


def humanize(term):
    return term.replace("_neg", " (negated)")


def level_of(conf):
    return "High" if conf >= 0.75 else "Medium" if conf >= 0.5 else "Low"


def explanation(label, conf, pro, con):
    if not pro and not con:
        return "We didn't recognise enough words in this text to judge it reliably, so treat this result as a rough guess."
    q = lambda items: ", ".join(f"“{humanize(t)}”" for t, _ in items[:4])
    msg = f"This reads as <b>{label.lower()}</b>" + (f", mainly because of {q(pro)}." if pro else ".")
    if con:
        msg += f" Words pulling the other way: {q(con)}."
    if label == "Neutral" or conf < 0.5:
        msg += " The wording is mixed, so this is a best guess rather than a clear-cut result."
    return msg


def bars_html(probs):
    return "".join(f'<div class="row"><span>{STYLE[c][2]} {c}</span><span>{probs[c]*100:.0f}%</span></div>'
                   f'<div class="bar"><div style="width:{probs[c]*100:.0f}%;background:{STYLE[c][0]}"></div></div>' for c in ["Positive", "Neutral", "Negative"])


def result_html(label, probs, pro, con):
    color, tint, icon = STYLE[label]; conf = probs[label]
    return f"""<div class="res" style="border-top:4px solid {color}">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:.8rem">
        <div class="verdict" style="color:{color}">{icon} {label}</div>
        <span class="chip" style="background:{tint};color:{color}">{level_of(conf)} confidence · {conf*100:.0f}%</span></div>
      <div style="margin-top:1.3rem;font-weight:600;color:#e6edf3">Sentiment breakdown</div>{bars_html(probs)}
      <div style="margin-top:1.5rem;margin-bottom:.5rem;font-weight:600;color:#e6edf3">Why we think so</div>
      <div class="why">{explanation(label, conf, pro, con)}</div>
      <div class="note">Percentages estimate how strongly the wording points to each sentiment. They are not guarantees.</div></div>"""


def kpi(col, value, label, color="#00f5d4"):
    col.markdown(f'<div class="kpi"><b style="color:{color}">{value}</b><span>{label}</span></div>', unsafe_allow_html=True)


def load_report():
    path = os.path.join(BASE, "results", "metrics", "linear_svm_best_report.csv")
    return pd.read_csv(path, index_col=0) if os.path.exists(path) else None


# ── State, plotting defaults, sidebar ─────────────────────────────────────────
ss = st.session_state
ss.setdefault("history", []); ss.setdefault("nav", "Home"); ss.setdefault("review_text", "")
PL = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter", color="#8b98a9", size=12), margin=dict(l=10, r=10, t=20, b=10))
GC = dict(gridcolor="#1e2d3d", zeroline=False)


def goto(page):
    ss.nav = page


def set_text(text):
    ss.review_text = text


with st.sidebar:
    st.markdown('<div style="padding:1.2rem 0 1rem;border-bottom:1px solid #1e2d3d;margin-bottom:1rem;">'
                '<span style="font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;color:#00f5d4;">Review</span>'
                '<span style="font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;color:#8b98a9;">IQ</span>'
                '<div style="font-size:.85rem;color:#8b98a9;margin-top:.2rem;">Understand how customers feel</div></div>', unsafe_allow_html=True)
    page = st.selectbox("PAGE", PAGES, key="nav", label_visibility="collapsed")
    if results is not None:
        st.markdown(f'<div style="margin-top:1rem;padding:.9rem;background:#111820;border:1px solid rgba(0,245,212,0.15);border-radius:10px;font-size:.9rem;color:#8b98a9;line-height:1.5;">Usually right about <b style="color:#00f5d4">{acc_txt}</b> of the time on reviews it had never seen.</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="margin-top:1rem;font-size:.85rem;color:#8b98a9;">Analyzed this session<br><span style="font-family:Syne,sans-serif;font-size:1.8rem;font-weight:800;color:#00f5d4;">{len(ss.history)}</span></div>', unsafe_allow_html=True)

# ═════════════════════════ HOME ═════════════════════════
if page == "Home":
    st.markdown('<div class="title">Understand how your customers feel</div><div class="sub">Paste a review or message. ReviewIQ tells you whether it is positive, negative or neutral, and explains why, in seconds.</div>', unsafe_allow_html=True)
    st.button("Analyze a review →", type="primary", on_click=goto, args=("Analyze",), key="cta_home")
    st.markdown("<br>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, acc_txt, "Overall accuracy"); kpi(k2, f"{DATASET['sample'] // 1000}K+", "Real reviews it learned from", "#ffb703")
    kpi(k3, "3", "Sentiments it detects", "#e6edf3"); kpi(k4, f"{n_features // 1000}k", "Words and phrases it knows", "#e6edf3")
    left, right = st.columns([3, 2])
    with left:
        st.markdown('<div class="lbl">What is ReviewIQ?</div>', unsafe_allow_html=True)
        st.markdown('<div class="card acc"><p>ReviewIQ reads customer feedback and tells you the mood behind it: <b style="color:#00f5d4">Positive</b>, <b style="color:#ffb703">Neutral</b> or <b style="color:#ff4d6d">Negative</b>. It learned from real product reviews, so it understands everyday wording, including phrases like “not good”.</p></div>', unsafe_allow_html=True)
        st.markdown('<div class="lbl">How it works</div>', unsafe_allow_html=True)
        for n, b, t in [("1", "You paste some text", "A review, a comment, a survey answer, anything written."),
                        ("2", "We read it carefully", "We tidy the text and keep track of words like “not” and “never”."),
                        ("3", "We compare it with what we learned", "It has seen thousands of reviews and knows how happy and unhappy people write."),
                        ("4", "You get the answer and the reasons", "The sentiment, how sure we are, and the words that decided it.")]:
            st.markdown(f'<div class="step"><i>{n}</i><div><b>{b}</b><br>{t}</div></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="lbl">Good to know</div>', unsafe_allow_html=True)
        st.markdown('<div class="card"><ul style="padding-left:1.1rem;margin:0"><li>Clear praise or complaints are judged most reliably.</li><li>Neutral or mixed reviews are the hardest, even for people.</li><li>Sarcasm can fool it.</li><li>Results are estimates, not guarantees.</li></ul></div>', unsafe_allow_html=True)
        st.markdown('<div class="card"><h4>Your privacy</h4><p>The app does not save your text. Your session summary disappears when you close the tab.</p></div>', unsafe_allow_html=True)

# ═════════════════════════ ANALYZE ═════════════════════════
elif page == "Analyze":
    st.markdown('<div class="title">Analyze a review</div><div class="sub">Paste any review, comment or message, then click Analyze.</div>', unsafe_allow_html=True)
    main, side = st.columns([3, 2])
    with main:
        st.markdown('<div class="lbl">Try an example</div>', unsafe_allow_html=True)
        for col, (name, ex) in zip(st.columns(len(EXAMPLES)), EXAMPLES.items()):
            col.button(name, key=f"ex_{name}", on_click=set_text, args=(ex,), use_container_width=True)
        st.markdown('<div class="lbl">Your text</div>', unsafe_allow_html=True)
        st.text_area("Your text", key="review_text", height=160, label_visibility="collapsed", placeholder="e.g. The packaging was nice, but the product itself was disappointing…")
        if st.button("Analyze sentiment", type="primary"):
            text = ss.review_text.strip()
            if not text:
                st.warning("Please enter some text first.")
            else:
                with st.spinner("Reading your text…"):
                    label, probs, pro, con = analyze(text)
                ss.history.insert(0, {"text": text, "label": label, "conf": probs[label], "time": datetime.now().strftime("%H:%M")})
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(result_html(label, probs, pro, con), unsafe_allow_html=True)
    with side:
        st.markdown('<div class="lbl">This session</div>', unsafe_allow_html=True)
        total = len(ss.history)
        for c in ["Positive", "Neutral", "Negative"]:
            n = sum(h["label"] == c for h in ss.history); pct = round(n / total * 100) if total else 0
            st.markdown(f'<div class="row" style="margin-top:.4rem"><span style="color:{STYLE[c][0]}">{c}</span><span style="color:#8b98a9">{n} · {pct}%</span></div><div class="bar"><div style="width:{pct}%;background:{STYLE[c][0]}"></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="lbl">Recent</div>', unsafe_allow_html=True)
        if not ss.history:
            st.markdown('<div style="color:#8b98a9">Nothing analyzed yet.</div>', unsafe_allow_html=True)
        for h in ss.history[:8]:
            st.markdown(f'<div class="item"><span style="color:{STYLE[h["label"]][0]}">●</span><span class="t">{escape(h["text"][:60])}</span><span class="m">{h["time"]}</span></div>', unsafe_allow_html=True)

# ═════════════════════════ DASHBOARD ═════════════════════════
elif page == "Dashboard":
    st.markdown('<div class="title">Dashboard</div><div class="sub">A summary of everything you have analyzed in this session.</div>', unsafe_allow_html=True)
    hist = ss.history
    if not hist:
        st.markdown('<div class="card" style="text-align:center;padding:2.5rem"><h4>Nothing to show yet</h4><p>Analyze a review and your results will appear here.</p></div>', unsafe_allow_html=True)
        st.button("Analyze a review →", type="primary", on_click=goto, args=("Analyze",), key="cta_dash")
    else:
        counts = {c: sum(h["label"] == c for h in hist) for c in ["Positive", "Neutral", "Negative"]}
        avg = sum(h["conf"] for h in hist) / len(hist)
        for col, (v, k, c) in zip(st.columns(5), [(len(hist), "Analyzed", "#e6edf3")] + [(counts[k], k, STYLE[k][0]) for k in counts] + [(f"{avg*100:.0f}%", "Average confidence", "#8b98a9")]):
            kpi(col, v, k, c)
        left, right = st.columns(2)
        with left:
            st.markdown('<div class="lbl">Sentiment mix</div>', unsafe_allow_html=True)
            shown = [(k, v) for k, v in counts.items() if v]
            fig = go.Figure(go.Pie(labels=[k for k, _ in shown], values=[v for _, v in shown], hole=.65, sort=False, textinfo="percent",
                                   marker=dict(colors=[STYLE[k][0] for k, _ in shown], line=dict(color="#080c10", width=3))))
            fig.update_layout(**PL, height=280, legend=dict(font=dict(color="#cdd9e5")))
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown('<div class="lbl">How sure were the results?</div>', unsafe_allow_html=True)
            for lv in ["High", "Medium", "Low"]:
                n = sum(level_of(h["conf"]) == lv for h in hist)
                st.markdown(f'<div class="row"><span>{lv} confidence</span><span style="color:#8b98a9">{n}</span></div><div class="bar"><div style="width:{n/len(hist)*100:.0f}%;background:#00f5d4"></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="lbl">Timeline</div>', unsafe_allow_html=True)
        order = list(reversed(hist)); levels = {"Positive": 1, "Neutral": 0, "Negative": -1}
        fig = go.Figure()
        for c in levels:
            idx = [i + 1 for i, h in enumerate(order) if h["label"] == c]
            fig.add_trace(go.Scatter(x=idx, y=[levels[c]] * len(idx), mode="markers", name=c, marker=dict(color=STYLE[c][0], size=12)))
        fig.update_layout(**PL, height=200, xaxis=dict(**GC, dtick=1, title="Analysis number"), yaxis=dict(**GC, tickvals=[-1, 0, 1], ticktext=["Negative", "Neutral", "Positive"]), legend=dict(font=dict(color="#cdd9e5")))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="lbl">Recent analyses</div>', unsafe_allow_html=True)
        st.markdown("".join(f'<div class="item"><span class="chip" style="background:{STYLE[h["label"]][1]};color:{STYLE[h["label"]][0]}">{h["label"]}</span><span class="t">{escape(h["text"][:100])}</span><span class="m">{h["conf"]*100:.0f}% · {h["time"]}</span></div>' for h in hist[:10]), unsafe_allow_html=True)

# ═════════════════════════ ACCURACY ═════════════════════════
elif page == "Accuracy":
    st.markdown(f'<div class="title">How accurate is ReviewIQ?</div><div class="sub">Tested on {DATASET["test"]:,} real reviews it had never seen before.</div>', unsafe_allow_html=True)
    report = load_report()
    if results is None or report is None:
        st.info("Accuracy results are not available yet.")
        st.stop()
    st.markdown(f'<div class="card acc"><h4><span style="color:#00f5d4;font-size:1.6rem">{acc_txt}</span> of reviews were labelled correctly</h4><p>Most reviews are positive ({base_acc*100:.0f}% of this test set), so a tool that always said “positive” would already be right {base_acc*100:.0f}% of the time. That is why we also check how well each sentiment is spotted:</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="lbl">How well it spots each sentiment</div>', unsafe_allow_html=True)
    notes = {"Positive": "Very reliable.", "Negative": "Reliable.", "Neutral": "The hardest. Mixed reviews are often confused with positive or negative ones."}
    for col, c in zip(st.columns(3), ["Positive", "Neutral", "Negative"]):
        rec, prec = float(report.loc[c, "recall"]), float(report.loc[c, "precision"])
        col.markdown(f'<div class="card"><h4 style="color:{STYLE[c][0]}">{STYLE[c][2]} {c}</h4>'
                     f'<div class="row"><span>Spotted</span><span>{rec*100:.0f}%</span></div><div class="bar"><div style="width:{rec*100:.0f}%;background:{STYLE[c][0]}"></div></div>'
                     f'<div class="row"><span>Right when it says so</span><span>{prec*100:.0f}%</span></div><div class="bar"><div style="width:{prec*100:.0f}%;background:{STYLE[c][0]}"></div></div>'
                     f'<p style="margin-top:.9rem;color:#8b98a9;font-size:.92rem">{notes[c]}</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="lbl">Words that most often signal each sentiment</div>', unsafe_allow_html=True)
    terms = top_terms(10)
    for col, c in zip(st.columns(3), ["Positive", "Neutral", "Negative"]):
        col.markdown(f'<div class="card"><h4 style="color:{STYLE[c][0]}">{c}</h4>' + "".join(f'<div class="item"><span class="t">{humanize(t)}</span></div>' for t, _ in terms.get(c, [])) + '</div>', unsafe_allow_html=True)
    st.markdown('<div class="note">Want the technical details? See the “For the technically curious” section on the About page.</div>', unsafe_allow_html=True)

# ═════════════════════════ ABOUT ═════════════════════════
else:
    st.markdown('<div class="title">About ReviewIQ</div><div class="sub">What it does, where it works best, and where it struggles.</div>', unsafe_allow_html=True)
    left, right = st.columns([3, 2])
    with left:
        st.markdown('<div class="card acc"><h4>What it does</h4><p>ReviewIQ reads a piece of customer feedback and tells you whether the mood is positive, neutral or negative, how sure it is, and which words decided it.</p></div>', unsafe_allow_html=True)
        g, h = st.columns(2)
        g.markdown('<div class="card"><h4 style="color:#00f5d4">Works well on</h4><ul style="padding-left:1.1rem;margin:0"><li>Clear praise or complaints</li><li>Product and service reviews</li><li>Phrases like “not good”</li></ul></div>', unsafe_allow_html=True)
        h.markdown('<div class="card"><h4 style="color:#ffb703">Finds hard</h4><ul style="padding-left:1.1rem;margin:0"><li>Neutral or mixed opinions</li><li>Sarcasm and double negatives</li><li>Words it has never seen</li></ul></div>', unsafe_allow_html=True)
    with right:
        st.markdown(f'<div class="card"><h4>About the data</h4><p>It learned from {DATASET["sample"]:,} real Amazon food product reviews, so it is best on reviews of that kind. Other topics may be less accurate.</p></div>', unsafe_allow_html=True)
    with st.expander("For the technically curious"):
        if results is None:
            st.write("Benchmark results are not available yet.")
        else:
            st.write(f"The text is cleaned, contractions are expanded and negation is marked (NLTK), then converted to TF-IDF features ({n_features:,} words and word pairs). A linear SVM scores each sentiment. Data: {DATASET['raw']:,} reviews, {DATASET['dedup']:,} after removing duplicates, a stratified 40% sample of {DATASET['sample']:,} split into {DATASET['train']:,} for training and {DATASET['test']:,} for testing. The percentages shown on the Analyze page are estimates derived from the SVM's scores.")
            cols = [c for c in ["Model", "Accuracy", "F1 (Macro)", "F1 (Weighted)", "F1 (Neutral)", "ROC-AUC (Macro)", "CV F1 (Weighted)"] if c in results.columns]
            body = "".join(f'<tr style="{"color:#00f5d4;font-weight:600" if r["Model"] == DEPLOYED else ""}"><td>{r["Model"]}{" (in use)" if r["Model"] == DEPLOYED else ""}</td>' + "".join(f"<td>{'–' if pd.isna(r[c]) else format(r[c], '.4f')}</td>" for c in cols[1:]) + "</tr>" for _, r in results.iterrows())
            st.markdown(f'<div style="overflow-x:auto"><table class="t"><tr>{"".join(f"<th>{c}</th>" for c in cols)}</tr>{body}</table></div>', unsafe_allow_html=True)
            st.caption("Built with Python, scikit-learn, NLTK, Streamlit and Plotly. Random Forest and XGBoost were not cross-validation-tuned.")