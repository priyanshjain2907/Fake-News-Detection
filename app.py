import streamlit as st
import pickle
import re

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# =============================
# LOAD MODEL
# =============================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# =============================
# TEXT CLEANING (IMPORTANT)
# =============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# =============================
# UI STYLING
# =============================
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #00ffcc;
    }
    .subtitle {
        text-align: center;
        color: #aaaaaa;
        font-size: 16px;
    }
    .box {
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# =============================
# HEADER
# =============================
st.markdown('<div class="title">📰 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by ML (TF-IDF + Classification)</div>', unsafe_allow_html=True)

st.write("")

# =============================
# INPUT
# =============================
news_input = st.text_area("✍️ Enter News Content:", height=200)



# =============================
# ANALYZE BUTTON
# =============================
if st.button("🔍 Analyze News"):

    if news_input.strip() == "":
        st.warning("⚠️ Please enter some text.")

    elif len(news_input.split()) < 5:
        st.warning("⚠️ Enter more meaningful content.")

    else:
        # 🔥 CLEAN TEXT BEFORE PREDICTION
        cleaned_text = clean_text(news_input)

        vec = vectorizer.transform([cleaned_text])
        prediction = model.predict(vec)[0]
        prob = model.predict_proba(vec).max()

        st.write("")

        # =============================
        # RESULT LOGIC
        # =============================
        if prob < 0.52:
            st.markdown(
                f'<div class="box" style="background-color:#ffa726;">⚠️ UNCERTAIN<br>Confidence: {prob:.2f}</div>',
                unsafe_allow_html=True
            )

        elif prediction == "FAKE":
            st.markdown(
                f'<div class="box" style="background-color:#ff4b4b;">❌ FAKE NEWS<br>Confidence: {prob:.2f}</div>',
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                f'<div class="box" style="background-color:#00c853;">✅ REAL NEWS<br>Confidence: {prob:.2f}</div>',
                unsafe_allow_html=True
            )

        # =============================
        # CONFIDENCE BAR
        # =============================
        st.write("")
        st.write("### Confidence Level")
        st.progress(int(prob * 100))

# =============================
# FOOTER
# =============================
st.write("---")
st.caption("Built using Machine Learning")
st.caption("Note: Model detects patterns, not factual truth")