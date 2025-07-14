import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

logo = Image.open("omegakavya.jpeg")
st.sidebar.image(logo, use_column_width=True)
st.sidebar.title("📊 Segmentation Dashboard")
st.sidebar.markdown("Explore data insights for clustered user segments.")
st.markdown("<style>section[data-testid='stSidebar'] { overflow-y: auto; }</style>", unsafe_allow_html=True)

# ---------- Page Config ----------
st.set_page_config(page_title="🎯 User Segmentation Dashboard", layout="wide")

# ---------- Load Data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("data/user_profiles_with_segments.csv")
    df.columns = df.columns.str.strip()  # Clean column names
    return df

df = load_data()

# ---------- Sidebar Filters ----------
st.sidebar.header("🔍 Filter Options")
segments = df['Segment_Name'].unique()
selected_segments = st.sidebar.multiselect("Select Segments", segments, default=list(segments))

filtered_df = df[df['Segment_Name'].isin(selected_segments)]

# ---------- Dashboard Title ----------
st.title("🎯 User Segmentation Analysis Dashboard")

# ---------- KPI Metrics ----------
st.markdown("### 📌 Overview Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Users", f"{len(filtered_df)}")
col2.metric("Avg CTR", f"{filtered_df['Click-Through Rates (CTR)'].mean():.2%}")
col3.metric("Avg Conversion", f"{filtered_df['Conversion Rates'].mean():.2%}")

# ---------- Segment Distribution ----------
st.markdown("### 📊 Segment Distribution")
segment_counts = filtered_df['Segment_Name'].value_counts()
st.bar_chart(segment_counts)

# ---------- Engagement Patterns ----------
st.markdown("### 🕒 Engagement Patterns")
engagement = filtered_df.groupby('Segment_Name')[
    ['Time Spent Online (hrs/weekday)', 'Time Spent Online (hrs/weekend)']
].mean()
st.line_chart(engagement)

# ---------- CTR & Conversion ----------
st.markdown("### 📈 CTR & Conversion Rates")
conversion = filtered_df.groupby('Segment_Name')[
    ['Click-Through Rates (CTR)', 'Conversion Rates']
].mean()
st.area_chart(conversion)

# ---------- Income Distribution ----------
st.markdown("### 💰 Income Distribution by Segment")
income = pd.crosstab(filtered_df['Segment_Name'], filtered_df['Income Level'])
st.bar_chart(income)

# ---------- Heatmap Comparison (Optional) ----------
st.markdown("### 🔥 Segment Metric Heatmap")
heatmap_data = filtered_df.groupby("Segment_Name")[
    ['Click-Through Rates (CTR)', 'Conversion Rates', 'Time Spent Online (hrs/weekday)', 'Time Spent Online (hrs/weekend)']
].mean()

fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
st.pyplot(fig)

# ---------- Segment Profiles ----------
st.markdown("### 📌 Segment Insights & Strategic Recommendations")

segment_profiles = {
    "Digital Natives": {
        "Demographics": "35–44, Female",
        "Income": "100k+",
        "Online Weekday": "2.8h", "Online Weekend": "4.6h",
        "CTR": "12.6%", "CR": "5.0%",
        "Device": "Desktop Only",
        "Interests": "Travel (122), Gardening (103), Digital Marketing (45)",
        "Size": 207,
        "Strategy": [
            "📱 Prioritize mobile-first experience and social-led promotions",
            "📸 Use travel/gardening content hooks in social ads",
            "🎯 Retarget via Instagram & lifestyle platforms"
        ]
    },
    "Casual Browsers": {
        "Demographics": "25–34, Female",
        "Income": "40k–60k",
        "Online Weekday": "2.8h", "Online Weekend": "4.5h",
        "CTR": "12.2%", "CR": "4.9%",
        "Device": "Desktop Only",
        "Interests": "Fitness (96), Reading (95), Digital Marketing (94)",
        "Size": 484,
        "Strategy": [
            "🖼️ Simplify site UI with clear CTAs",
            "📧 Send regular personalized email nudges",
            "🎨 Visually-driven campaigns with light interactions"
        ]
    },
    "Power Users": {
        "Demographics": "25–34, Male",
        "Income": "0–20k",
        "Online Weekday": "2.8h", "Online Weekend": "4.7h",
        "CTR": "12.9%", "CR": "4.8%",
        "Device": "Mobile Only",
        "Interests": "Finance (155), Cooking (32), Wellness (27)",
        "Size": 155,
        "Strategy": [
            "📊 Use dashboards, push insights, and real-time nudges",
            "📈 Upsell premium finance tools or investment content",
            "🧪 A/B test deep-link features for power workflows"
        ]
    },
    "Premium Engagers": {
        "Demographics": "25–34, Female",
        "Income": "20k–40k",
        "Online Weekday": "2.7h", "Online Weekend": "4.7h",
        "CTR": "13.1%", "CR": "5.3%",
        "Device": "Desktop Only",
        "Interests": "Pet Care (154), Data Science (30), Digital Marketing (25)",
        "Size": 154,
        "Strategy": [
            "🎁 Offer exclusives: early access, beta invites, curated newsletters",
            "📣 Focus on value-driven campaigns with loyalty perks",
            "🧠 Use data-driven storytelling in email & blog formats"
        ]
    }
}

# Render Segment Cards
for seg in selected_segments:
    if seg in segment_profiles:
        prof = segment_profiles[seg]
        with st.expander(f"📂 {seg} — {prof['Size']} users", expanded=True):
            st.markdown(f"**👥 Demographics:** {prof['Demographics']}")
            st.markdown(f"**💵 Income Level:** {prof['Income']}")
            st.markdown(f"**⏱️ Online Time:** {prof['Online Weekday']} weekdays, {prof['Online Weekend']} weekends")
            st.markdown(f"**📊 CTR / Conversion:** {prof['CTR']} / {prof['CR']}")
            st.markdown(f"**💻 Preferred Device:** {prof['Device']}")
            st.markdown(f"**🎯 Top Interests:** {prof['Interests']}")
            st.markdown("**💡 Strategic Actions:**")
            for tip in prof['Strategy']:
                st.markdown(f"- {tip}")

# ---------- Export Button ----------
st.markdown("### 📥 Export Data")
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name='filtered_user_segments.csv',
    mime='text/csv'
)