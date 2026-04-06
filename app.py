import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# ------------------ UI SETTINGS ------------------
st.set_page_config(page_title="Police AI Assistant", layout="centered")

st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
    color: white;
}
h1, h2, h3 {
    color: #facc15;
}
textarea {
    background-color: #1e293b !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
X = pickle.load(open("matrix.pkl", "rb"))
df = pickle.load(open("data.pkl", "rb"))

# ------------------ TITLE ------------------
st.title("🚓 Police AI Assistant (BNS)")
st.write("Analyze crime descriptions and generate FIR instantly")

# ------------------ INPUT ------------------
user_input = st.text_area("📝 Describe the incident:", height=150)

# ------------------ FUNCTIONS ------------------

def explain_simple(text):
    return "This law deals with: " + text[:150] + "..."

def generate_fir(user_input, section, crime):
    date = datetime.now().strftime("%d-%m-%Y %H:%M")

    return f"""
FIRST INFORMATION REPORT (FIR)

Date & Time: {date}

Incident Description:
{user_input}

Applicable Law:
Section {section} - {crime}

Details:
The above-mentioned act is identified as a punishable offense under Bharatiya Nyaya Sanhita (BNS).

Action:
This complaint is recorded for further legal investigation.

Filed by:
Police Assistant System
"""

# ------------------ MAIN LOGIC ------------------

if st.button("🔍 Analyze Crime"):
    if user_input:

        user_vec = vectorizer.transform([user_input])
        similarity = cosine_similarity(user_vec, X)

        top_indices = similarity[0].argsort()[-3:][::-1]

        st.success("Top Matches Found ✅")

        for idx, i in enumerate(top_indices):
            result = df.iloc[i]

            st.write("---")
            st.write(f"## 🔹 Match {idx+1}")

            st.write("### 📜 Section")
            st.write(result["Section"])

            st.write("### ⚖️ Crime")
            st.write(result["Section_name"])

            st.write("### 📖 Description")
            st.write(result["Description"])

            # Simple Explanation
            st.write("### 🧠 Simple Explanation")
            st.write(explain_simple(result["Description"]))

            # FIR Generator per result
            st.write("### 📝 FIR Draft")
            fir_text = generate_fir(user_input, result["Section"], result["Section_name"])
            st.code(fir_text)

    else:
        st.warning("⚠️ Please enter incident description")