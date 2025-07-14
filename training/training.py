import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# --- 1. Data Loading and Initial Exploration ---
df = pd.read_csv("../data/user_profiles_with_segments.csv")

# --- 2. Data Preprocessing and Feature Engineering ---
df = df.drop('User ID', axis=1)

# Encode categorical features
categorical_cols = ['Age', 'Gender', 'Location', 'Language', 'Education Level', 'Device Usage', 'Income Level']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Process 'Top Interests'
df['Top Interests'] = df['Top Interests'].fillna('')
interests_list = df['Top Interests'].str.split(', ')

mlb = MultiLabelBinarizer()
interests_encoded = mlb.fit_transform(interests_list)
interests_df = pd.DataFrame(interests_encoded, columns=mlb.classes_)

df_processed = pd.concat([df.drop('Top Interests', axis=1), interests_df], axis=1)

# --- 3. Feature Scaling ---
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_processed)

# --- 4. Determine Optimal Clusters ---
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# --- 5. K-Means Clustering ---
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans.fit(df_scaled)
df['Segment'] = kmeans.labels_

# --- 6. Enhanced Segment Analysis ---

# Create meaningful segment names
segment_names = {
    0: "Digital Natives",
    1: "Casual Browsers",
    2: "Power Users",
    3: "Premium Engagers"
}

df['Segment_Name'] = df['Segment'].map(segment_names)

print("🎯 USER SEGMENTATION ANALYSIS")
print("=" * 50)

# Segment distribution
segment_counts = df['Segment_Name'].value_counts()
print(f"\n📊 SEGMENT DISTRIBUTION:")
for segment, count in segment_counts.items():
    percentage = (count / len(df)) * 100
    print(f"   {segment}: {count} users ({percentage:.1f}%)")

# Create comprehensive segment profiles
print(f"\n🔍 DETAILED SEGMENT PROFILES:")
print("=" * 50)

# Decode categorical columns for analysis
df_decoded = df.copy()
for col in categorical_cols:
    df_decoded[col] = label_encoders[col].inverse_transform(df_decoded[col])

# Analyze each segment
for segment_id, segment_name in segment_names.items():
    segment_data = df_decoded[df_decoded['Segment'] == segment_id]

    print(f"\n🏷️  {segment_name.upper()} (Segment {segment_id})")
    print(f"    Size: {len(segment_data)} users")
    print(f"    Key Characteristics:")

    # Demographics
    most_common_age = segment_data['Age'].mode()[0]
    most_common_gender = segment_data['Gender'].mode()[0]
    most_common_income = segment_data['Income Level'].mode()[0]

    print(f"    • Primary Demographics: {most_common_age}, {most_common_gender}")
    print(f"    • Income Level: {most_common_income}")

    # Engagement metrics
    avg_weekday_time = segment_data['Time Spent Online (hrs/weekday)'].mean()
    avg_weekend_time = segment_data['Time Spent Online (hrs/weekend)'].mean()
    avg_ctr = segment_data['Click-Through Rates (CTR)'].mean()
    avg_conversion = segment_data['Conversion Rates'].mean()

    print(f"    • Online Time: {avg_weekday_time:.1f}h weekdays, {avg_weekend_time:.1f}h weekends")
    print(f"    • Click Rate: {avg_ctr:.1%} | Conversion Rate: {avg_conversion:.1%}")

    # Device preference
    most_common_device = segment_data['Device Usage'].mode()[0]
    print(f"    • Preferred Device: {most_common_device}")

    # Top interests - use the original interests_df with segment mapping
    segment_interests_data = interests_df[df['Segment'] == segment_id]
    if len(segment_interests_data) > 0:
        interest_sums = segment_interests_data.sum().sort_values(ascending=False)
        top_interests = interest_sums.head(3)
        if len(top_interests) > 0 and top_interests.iloc[0] > 0:
            interests_str = ", ".join([f"{interest} ({count})" for interest, count in top_interests.items() if count > 0])
            print(f"    • Top Interests: {interests_str}")
        else:
            print(f"    • Top Interests: No specific interests identified")

# --- 7. Visualizations ---
print(f"\n📈 GENERATING VISUALIZATIONS...")

# Create a comprehensive dashboard
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('User Segmentation Dashboard', fontsize=16, fontweight='bold')

# 1. Segment Distribution
segment_counts.plot(kind='pie', ax=axes[0,0], autopct='%1.1f%%', startangle=90)
axes[0,0].set_title('User Distribution by Segment')
axes[0,0].set_ylabel('')

# 2. Engagement Comparison
engagement_data = df_decoded.groupby('Segment_Name')[['Time Spent Online (hrs/weekday)',
                                                      'Time Spent Online (hrs/weekend)']].mean()
engagement_data.plot(kind='bar', ax=axes[0,1])
axes[0,1].set_title('Average Online Time by Segment')
axes[0,1].set_ylabel('Hours')
axes[0,1].tick_params(axis='x', rotation=45)

# 3. Conversion Performance
performance_data = df_decoded.groupby('Segment_Name')[['Click-Through Rates (CTR)',
                                                       'Conversion Rates']].mean()
performance_data.plot(kind='bar', ax=axes[1,0])
axes[1,0].set_title('Performance Metrics by Segment')
axes[1,0].set_ylabel('Rate')
axes[1,0].tick_params(axis='x', rotation=45)

# 4. Income Distribution
income_crosstab = pd.crosstab(df_decoded['Segment_Name'], df_decoded['Income Level'])
income_crosstab.plot(kind='bar', stacked=True, ax=axes[1,1])
axes[1,1].set_title('Income Distribution by Segment')
axes[1,1].set_ylabel('Number of Users')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# --- 8. Marketing Recommendations ---
print(f"\n💡 MARKETING RECOMMENDATIONS:")
print("=" * 50)

recommendations = {
    "Digital Natives": "Focus on mobile-first campaigns with social media integration",
    "Casual Browsers": "Use simple, clear messaging with strong visual appeal",
    "Power Users": "Leverage data-driven content and advanced features",
    "Premium Engagers": "Highlight premium features and exclusive content"
}

for segment, recommendation in recommendations.items():
    segment_size = segment_counts[segment]
    print(f"\n🎯 {segment} ({segment_size} users):")
    print(f"    Strategy: {recommendation}")

# --- 9. Save Models and Data ---
feature_order = list(df_processed.columns)
joblib.dump(kmeans, 'kmeans_model.joblib')
joblib.dump(label_encoders, 'label_encoders.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(mlb, 'mlb.joblib')
joblib.dump(feature_order, 'feature_order.joblib')

# Save with segment names
for col in categorical_cols:
    df[col] = label_encoders[col].inverse_transform(df[col])
df.to_csv('user_profiles_with_segments.csv', index=False)

print(f"\n✅ Analysis complete! Model and data saved successfully.")
print(f"   📁 Files saved: kmeans_model.joblib, user_profiles_with_segments.csv")