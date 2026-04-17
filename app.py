import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Cloud Cost Intelligence", layout="wide")

# ------------------ HEADER ------------------
st.title("Cloud Cost Intelligence Dashboard")
st.caption("eCommerce Cloud Cost Monitoring & Optimization System")

# ------------------ LOAD DATA ------------------
df = pd.read_csv("outputs/reports/ecommerce_cloud_cost_dataset.csv")

# ------------------ FEATURE ENGINEERING ------------------
df['compute_cost'] = df['compute_hours'] * 0.05
df['storage_cost'] = df['storage_usage_gb'] * 0.01
df['transfer_cost'] = df['data_transfer_gb'] * 0.02
df['api_cost'] = df['api_requests'] * 0.0001

df['total_cost'] = (
    df['compute_cost'] +
    df['storage_cost'] +
    df['transfer_cost'] +
    df['api_cost']
)

df['cloud_service'] = df['service_type'].map({
    'Compute': 'EC2 / Virtual Machines',
    'Storage': 'S3 / Blob Storage',
    'Database': 'RDS / NoSQL Databases',
    'Machine Learning': 'SageMaker / AI Services'
})

# ------------------ SIDEBAR FILTER ------------------
st.sidebar.header("Filter Data")

selected_dept = st.sidebar.multiselect(
    "Department",
    df['department'].unique(),
    default=df['department'].unique()
)

df = df[df['department'].isin(selected_dept)]

# ------------------ WASTE DETECTION ------------------
low_cpu = df["cpu_utilization"].quantile(0.25)
high_cost = df["total_cost"].quantile(0.75)
low_user = df["active_users"].quantile(0.25)
high_storage = df["storage_usage_gb"].quantile(0.75)

df["is_waste"] = (
    ((df["cpu_utilization"] < low_cpu) & (df["compute_hours"] > 5)) |
    ((df["total_cost"] > high_cost) & (df["active_users"] < low_user)) |
    ((df["storage_usage_gb"] > high_storage) & (df["active_users"] < low_user))
)

df["waste_cost"] = df["total_cost"] * df["is_waste"] * 0.5

# ------------------ ANOMALY DETECTION ------------------
features = df[['compute_cost','storage_cost','transfer_cost','api_cost','total_cost']]

model = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = model.fit_predict(features)

df['anomaly_label'] = df['anomaly'].map({1: 'Normal', -1: 'Anomaly'})

# ------------------ RECOMMENDATION ------------------
def generate_recommendation(row):
    if row['anomaly'] == -1:
        return "Investigate abnormal cost spike"
    elif row['is_waste']:
        return "Optimize or remove underutilized resources"
    elif row['compute_cost'] > row['total_cost'] * 0.5:
        return "Use reserved instances or autoscaling"
    elif row['storage_cost'] > row['total_cost'] * 0.4:
        return "Optimize storage usage"
    elif row['api_cost'] > row['total_cost'] * 0.3:
        return "Reduce unnecessary API calls"
    else:
        return "Monitor usage"

df['recommendation'] = df.apply(generate_recommendation, axis=1)

# ------------------ KPI CARDS ------------------
st.markdown("### Key Metrics")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Cost", f"${df['total_cost'].sum():,.0f}")
c2.metric("Waste Cost", f"${df['waste_cost'].sum():,.0f}")
c3.metric("Anomalies", int((df['anomaly'] == -1).sum()))
c4.metric("Waste Resources", int(df['is_waste'].sum()))

st.markdown("---")

# ------------------ CHARTS (SIDE BY SIDE) ------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Cost by Department")
    st.bar_chart(df.groupby('department')['total_cost'].sum())

with col2:
    st.subheader("Cost Breakdown")
    st.bar_chart(df[['compute_cost','storage_cost','transfer_cost','api_cost']].sum())

st.markdown("---")

# ------------------ ANOMALY CHART ------------------
st.subheader("Anomaly Detection")

fig, ax = plt.subplots()

normal = df[df['anomaly'] == 1]
anomaly = df[df['anomaly'] == -1]

ax.scatter(normal.index, normal['total_cost'], label='Normal')
ax.scatter(anomaly.index, anomaly['total_cost'], label='Anomaly')

ax.legend()
st.pyplot(fig)

st.markdown("---")

# ------------------ TOP WASTE ------------------
st.subheader("Top Waste Resources")

waste_df = df[df['is_waste'] == True]

st.dataframe(
    waste_df.sort_values(by="waste_cost", ascending=False)
    [['department','cloud_service','total_cost','waste_cost']]
    .head(10)
)

st.markdown("---")

# ------------------ RECOMMENDATIONS ------------------
st.subheader("Recommendations")

st.dataframe(
    df[['department','cloud_service','total_cost','anomaly_label','recommendation']]
)