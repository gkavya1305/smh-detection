#Influence Of Social Media On Mental Health

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
# Load dataset
df = pd.read_csv("synthetic_social_media_addiction_dataset.csv")
# Features and label
X = df[['Daily_Hours', 'App_Opens', 'Night_Usage', 'Notifications_Checked',
        'Engagement_Score', 'Self_Reported_Addiction']]
y = df['Label']
# Split and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)
# Streamlit App
st.title("📱 Social Media Addiction Detection")
st.sidebar.header("📊 Input User Activity")
user_input = {
    "Daily_Hours": st.sidebar.slider("Daily Usage Hours", 0.0, 12.0, 3.0),
    "App_Opens": st.sidebar.slider("App Opens per Day", 0, 200, 50),
    "Night_Usage": st.sidebar.slider("Nighttime Usage (hours)", 0.0, 6.0, 1.0),
    "Notifications_Checked": st.sidebar.slider("Notifications Checked", 0, 500, 150),
    "Engagement_Score": st.sidebar.slider("Engagement Score", 0, 500, 200),
    "Self_Reported_Addiction": st.sidebar.slider("Self-reported Level (1-5)", 1, 5, 3)
}
input_df = pd.DataFrame([user_input])
# Prediction
risk_score = model.predict_proba(input_df)[0][1]
label = "Addicted" if risk_score > 0.5 else "Not Addicted"
# Output
st.subheader("🧠 Prediction")
st.write(f"*Addiction Risk Score:* {risk_score:.2f}")
st.write(f"*Classification:* {label}")
# Recommendations
st.subheader("✅ Recommendations")
if risk_score > 0.5:
    st.markdown("""
    - ⏱ Set daily time limits for social media
    - 🚫 Use app blockers or focus mode
    - 🌙 Schedule social media-free nights
    - 🧘 Engage in offline activities
    """)
else:
    st.markdown("Keep up the healthy habits! 👍")
# Visualizations
st.subheader("📈 Dashboard")
col1, col2 = st.columns(2)
with col1:
    st.markdown("*Daily Usage Distribution*")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['Daily_Hours'], kde=True, bins=20, ax=ax1)
    st.pyplot(fig1)
with col2:
    st.markdown("*Engagement by Addiction*")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='Label', y='Engagement_Score', data=df, ax=ax2)
    ax2.set_xticklabels(["Not Addicted", "Addicted"])
    st.pyplot(fig2)
# Heatmap for night usage frequency
st.markdown("*Night Usage Heatmap*")
fig3, ax3 = plt.subplots()
sns.histplot(df['Night_Usage'], bins=15, color="purple", kde=True, ax=ax3)
st.pyplot(fig3)
