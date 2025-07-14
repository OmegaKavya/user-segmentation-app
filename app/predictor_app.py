import streamlit as st
import numpy as np
import joblib
import pandas as pd
import json
from PIL import Image

# Load your logo
logo = Image.open("app/omegakavya.jpeg")
st.sidebar.image(logo, use_container_width=True)

# Load models and encoders
kmeans = joblib.load("models/kmeans_model.joblib")
label_encoders = joblib.load("models/label_encoders.joblib")
scaler = joblib.load("models/scaler.joblib")
mlb = joblib.load("models/mlb.joblib")
feature_order = joblib.load("models/feature_order.joblib")

# Segment labels and descriptions
segment_names = {
    0: "Digital Natives",
    1: "Casual Browsers",
    2: "Power Users",
    3: "Premium Engagers"
}

segment_profiles = {
    "Digital Natives": {
        "Who": "Tech-savvy users aged 18–24 with high online activity.",
        "Behavior": "Constant engagement, early adopters, multi-device users.",
        "Suggestions": [
            "Use gamified or interactive campaigns.",
            "Optimize for mobile-first UX.",
            "Leverage social proof and influencer marketing."
        ]
    },
    "Casual Browsers": {
        "Who": "Mid-income users aged 25–34 who occasionally browse.",
        "Behavior": "Moderate online activity with interests in fitness, reading, and tech.",
        "Suggestions": [
            "Simplify UI and enhance CTAs.",
            "Send email nudges to re-engage.",
            "Use minimal visual campaigns."
        ]
    },
    "Power Users": {
        "Who": "Highly active users aged 30–45, often professionals.",
        "Behavior": "High conversion, explore deeply before purchasing.",
        "Suggestions": [
            "Provide detailed comparisons and reviews.",
            "Use remarketing and tailored offers.",
            "Highlight premium features."
        ]
    },
    "Premium Engagers": {
        "Who": "Affluent users 35+ with consistent high CTR and loyalty.",
        "Behavior": "High-value conversions, repeat customers.",
        "Suggestions": [
            "Offer loyalty programs and early access.",
            "Focus on high-quality visuals.",
            "Use targeted, data-driven remarketing."
        ]
    }
}

# Page config
st.set_page_config(page_title="Segment Predictor", layout="centered")
st.title("🧠 Predict User Segment")
st.markdown("Use this tool to classify users into meaningful segments based on their demographics, behavior, and interests.")

# Input form
with st.form("segment_form"):
    st.subheader("📋 Enter User Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.selectbox("📅 Age", label_encoders['Age'].classes_)
        gender = st.selectbox("🧍 Gender", label_encoders['Gender'].classes_)
        location = st.selectbox("📍 Location", label_encoders['Location'].classes_)
        language = st.selectbox("🗣️ Language", label_encoders['Language'].classes_)
        education = st.selectbox("🎓 Education Level", label_encoders['Education Level'].classes_)

    with col2:
        device = st.selectbox("💻 Device Usage", label_encoders['Device Usage'].classes_)
        income = st.selectbox("💵 Income Level", label_encoders['Income Level'].classes_)
        interests = st.multiselect("🎯 Top Interests", mlb.classes_)
        weekday_time = st.number_input("⏱️ Time Spent Online (hrs/weekday)", min_value=0.0, format="%.2f")
        weekend_time = st.number_input("🕒 Time Spent Online (hrs/weekend)", min_value=0.0, format="%.2f")
        ctr = st.slider("📈 Click-Through Rate (CTR)", 0.0, 1.0, step=0.01, format="%.2f")
        conversion = st.slider("✅ Conversion Rate", 0.0, 1.0, step=0.01, format="%.2f")

    submitted = st.form_submit_button("🔍 Predict Segment")

# Prediction logic
if submitted:
    user_data = {
        "Age": label_encoders["Age"].transform([age])[0],
        "Gender": label_encoders["Gender"].transform([gender])[0],
        "Location": label_encoders["Location"].transform([location])[0],
        "Language": label_encoders["Language"].transform([language])[0],
        "Education Level": label_encoders["Education Level"].transform([education])[0],
        "Device Usage": label_encoders["Device Usage"].transform([device])[0],
        "Income Level": label_encoders["Income Level"].transform([income])[0],
        "Time Spent Online (hrs/weekday)": weekday_time,
        "Time Spent Online (hrs/weekend)": weekend_time,
        "Click-Through Rates (CTR)": ctr,
        "Conversion Rates": conversion
    }

    # Interests
    interest_vec = mlb.transform([interests if interests else []])[0]
    interest_data = dict(zip(mlb.classes_, interest_vec))

    full_feature_dict = {**user_data, **interest_data}
    aligned_features = [full_feature_dict.get(f, 0) for f in feature_order]
    features_scaled = scaler.transform([aligned_features])

    segment_id = kmeans.predict(features_scaled)[0]
    predicted_segment = segment_names.get(segment_id, "Unknown Segment")

    # Output
    st.success(f"🎯 **Predicted Segment:** {predicted_segment}")

    with st.expander("📚 Segment Details", expanded=True):
        profile = segment_profiles.get(predicted_segment)
        if profile:
            st.markdown(f"**👥 Who They Are:** {profile['Who']}")
            st.markdown(f"**🧠 Behavioral Traits:** {profile['Behavior']}")
            st.markdown("**🛠️ Strategic Suggestions:**")
            for tip in profile["Suggestions"]:
                st.markdown(f"- {tip}")

    # Debug
    with st.expander("🔧 Debug Logs (optional)", expanded=False):
        st.write("📋 Encoded Inputs:", user_data)
        st.write("🎯 Interests Vector:", interest_data)
        st.write("📊 Final Feature Vector (ordered):", aligned_features)

    # Export
    export_data = {
        "Inputs": {
            "Age": age, "Gender": gender, "Location": location, "Language": language,
            "Education Level": education, "Device Usage": device, "Income Level": income,
            "Weekday Time": weekday_time, "Weekend Time": weekend_time,
            "CTR": ctr, "Conversion Rate": conversion,
            "Interests": interests
        },
        "Predicted Segment": predicted_segment
    }
    json_str = json.dumps(export_data, indent=2)
    st.download_button("📤 Export Result as JSON", data=json_str, file_name="segment_result.json", mime="application/json")

# Optional Explorer
with st.expander("📊 Explore All Segment Profiles", expanded=False):
    for seg, profile in segment_profiles.items():
        st.markdown(f"### 🔸 {seg}")
        st.markdown(f"**👥 Who They Are:** {profile['Who']}")
        st.markdown(f"**🧠 Behavioral Traits:** {profile['Behavior']}")
        st.markdown("**🛠️ Strategic Suggestions:**")
        for tip in profile["Suggestions"]:
            st.markdown(f"- {tip}")
        st.markdown("---")

# Footer
st.markdown("---")
st.markdown("© 2025 User Segmentation App · Built with ❤️ using Streamlit")