import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

# ---------- Sidebar Logo & Title ----------
logo = Image.open("app/omegakavya.jpeg")
st.sidebar.image(logo, use_container_width=True)
st.sidebar.title("\U0001F4CA Segmentation Dashboard")
st.sidebar.markdown("Explore data insights for clustered user segments.")
st.markdown("<style>section[data-testid='stSidebar'] { overflow-y: auto; }</style>", unsafe_allow_html=True)

# ---------- Page Config ----------
st.set_page_config(page_title="\U0001F3AF User Segmentation Dashboard", layout="wide")

# ---------- Load Data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("data/user_profiles_with_segments.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# ---------- Sidebar Filters ----------
st.sidebar.header("\U0001F50D Filter Options")
segments = df['Segment_Name'].unique()
selected_segments = st.sidebar.multiselect("Select Segments", segments, default=list(segments))
filtered_df = df[df['Segment_Name'].isin(selected_segments)]

# ---------- Dashboard Title ----------
st.title("\U0001F3AF User Segmentation Analysis Dashboard")
# ---------- Introductory Description ----------
st.markdown("""
<div style='font-size:16px; line-height:1.6;'>
Welcome to the <b>User Segmentation Dashboard</b> — an interactive platform to explore behavioral and demographic patterns of users across distinct segments.<br><br>
Use the filters in the sidebar to dive deeper into CTR trends, income groups, engagement patterns, and actionable strategies tailored to each cluster.
</div>
""", unsafe_allow_html=True)

# ---------- KPI Metrics ----------
st.markdown("### \U0001F4CC Overview Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Users", f"{len(filtered_df)}")
col2.metric("Avg CTR", f"{filtered_df['Click-Through Rates (CTR)'].mean():.2%}")
col3.metric("Avg Conversion", f"{filtered_df['Conversion Rates'].mean():.2%}")

# ---------- Segment Distribution ----------
st.markdown("### \U0001F4CA Segment Distribution")
segment_counts = filtered_df['Segment_Name'].value_counts()
seg_df = pd.DataFrame({
    "Segment": segment_counts.index,
    "User Count": segment_counts.values
})
fig1 = px.bar(
    seg_df,
    x="Segment",
    y="User Count",
    color="Segment",
    title="Segment Distribution",
    hover_data=["Segment", "User Count"],
    color_discrete_sequence=px.colors.qualitative.Set2
)
st.plotly_chart(fig1, use_container_width=True)

# ---------- Engagement Patterns ----------
st.markdown("### \U0001F552 Engagement Patterns")
engagement = filtered_df.groupby('Segment_Name')[
    ['Time Spent Online (hrs/weekday)', 'Time Spent Online (hrs/weekend)']
].mean().reset_index()
fig2 = px.line(
    engagement.melt(id_vars="Segment_Name"),
    x="Segment_Name",
    y="value",
    color="variable",
    markers=True,
    labels={"value": "Avg Hours", "Segment_Name": "Segment", "variable": "Day Type"},
    title="Engagement Patterns",
    hover_data=["Segment_Name", "variable", "value"]
)
st.plotly_chart(fig2, use_container_width=True)

# ---------- CTR & Conversion ----------
st.markdown("### \U0001F4C8 CTR & Conversion Rates")
conversion = filtered_df.groupby('Segment_Name')[
    ['Click-Through Rates (CTR)', 'Conversion Rates']
].mean().reset_index()
fig3 = px.area(
    conversion.melt(id_vars="Segment_Name"),
    x="Segment_Name",
    y="value",
    color="variable",
    markers=True,
    labels={"value": "Rate", "Segment_Name": "Segment", "variable": "Metric"},
    title="CTR & Conversion Rates",
    hover_data=["Segment_Name", "variable", "value"]
)
st.plotly_chart(fig3, use_container_width=True)

# ---------- Income Distribution ----------
st.markdown("### \U0001F4B0 Income Distribution by Segment")
income = pd.crosstab(filtered_df['Segment_Name'], filtered_df['Income Level'])
income = income.reset_index().melt(id_vars="Segment_Name", var_name="Income Level", value_name="User Count")
fig4 = px.bar(
    income,
    x="Segment_Name",
    y="User Count",
    color="Income Level",
    barmode="group",
    title="Income Distribution",
    labels={"Segment_Name": "Segment"},
    hover_data=["Segment_Name", "Income Level", "User Count"]
)
st.plotly_chart(fig4, use_container_width=True)

# ---------- Heatmap Comparison (Optional) ----------
st.markdown("### \U0001F525 Segment Metric Heatmap")
heatmap_data = filtered_df.groupby("Segment_Name")[
    ['Click-Through Rates (CTR)', 'Conversion Rates', 'Time Spent Online (hrs/weekday)', 'Time Spent Online (hrs/weekend)']
].mean()
fig5 = px.imshow(
    heatmap_data,
    labels=dict(x="Metrics", y="Segment", color="Value"),
    color_continuous_scale="YlGnBu",
    title="Segment Metric Heatmap",
    text_auto=".2f"
)
st.plotly_chart(fig5, use_container_width=True)

# ---------- Segment Profiles ----------
st.markdown("### \U0001F4CC Segment Insights & Strategic Recommendations")

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
        with st.expander(f"\U0001F4C2 {seg} — {prof['Size']} users", expanded=True):
            st.markdown(f"**\U0001F465 Demographics:** {prof['Demographics']}")
            st.markdown(f"**\U0001F4B5 Income Level:** {prof['Income']}")
            st.markdown(f"**⏱️ Online Time:** {prof['Online Weekday']} weekdays, {prof['Online Weekend']} weekends")
            st.markdown(f"**\U0001F4CA CTR / Conversion:** {prof['CTR']} / {prof['CR']}")
            st.markdown(f"**\U0001F4BB Preferred Device:** {prof['Device']}")
            st.markdown(f"**\U0001F3AF Top Interests:** {prof['Interests']}")
            st.markdown("**\U0001F4A1 Strategic Actions:**")
            for tip in prof['Strategy']:
                st.markdown(f"- {tip}")

# ---------- Export Button ----------
st.markdown("### \U0001F4E5 Export Data")
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name='filtered_user_segments.csv',
    mime='text/csv'
)
# ---------- Simulated Footer ----------
footer = """
<style>
    .footer {
        position: relative;
        bottom: 0;
        width: 100%;
        text-align: center;
        color: #888;
        padding: 10px;
        font-size: 1.0em;
    }
</style>
<div class="footer">
    Made by <a href="https://github.com/OmegaKavya" target="_blank" style="text-decoration: none; color: #888;">OmegaKavya</a>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)