
import streamlit as st
import joblib
import os

# -----------------------------------------
# 1. Page configuration
# -----------------------------------------

st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="📧",
    layout="centered"
)

# -----------------------------------------
# 2. Load trained model
# -----------------------------------------

MODEL_PATH = os.path.join("model", "spam_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Run train_model.py first.")
    st.stop()

model = load_model()

# -----------------------------------------
# 3. Application heading
# -----------------------------------------

st.title("📧 Spam Message Detector")

st.write(
    "Enter a message below to check whether "
    "the machine learning model classifies it as spam."
)

st.divider()

# -----------------------------------------
# 4. Message input
# -----------------------------------------

message = st.text_area(
    "Enter your message:",
    placeholder="Example: Congratulations! You have won a prize...",
    height=150
)

# -----------------------------------------
# 5. Prediction button
# -----------------------------------------

if st.button("Analyze Message", type="primary"):

    if not message.strip():
        st.warning("Please enter a message first.")

    else:
        prediction = int(model.predict([message])[0])
        probabilities = model.predict_proba([message])[0]

        spam_probability = float(probabilities[1]) * 100

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("🚨 The message is classified as SPAM.")
        else:
            st.success("✅ The message is classified as NOT SPAM.")

        st.metric(
            "Model-estimated spam probability",
            f"{spam_probability:.2f}%"
        )

        st.caption(
            "This percentage is the model's estimated probability, "
            "not a guarantee that the message is spam."
        )

st.divider()

# -----------------------------------------
# 6. Project information
# -----------------------------------------

with st.expander("About this project"):

    st.write(
        """
        **Project:** Spam Message Detection Using Machine Learning

        **Algorithm:** Multinomial Naive Bayes

        **Feature Extraction:** TF-IDF Vectorization

        **Libraries:** Python, Pandas, Scikit-learn, Streamlit

        The model was trained using labelled messages and predicts
        whether new text resembles the spam or legitimate messages
        in its training dataset.
        """
    )

st.caption("Machine Learning Internship Project")
