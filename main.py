
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

BASE_DIR = Path(__file__).resolve().parent
FEATURES = ["Age", "Annual_Income_k", "Spending_Score"]
DATASETS = [
    BASE_DIR / "Mall_Customers.csv",
    BASE_DIR / "Second_Customers.csv"
]

st.set_page_config(
    page_title="Smart Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Smart Customer Segmentation")
st.caption("AI-powered customer analysis using K-Means Machine Learning")

@st.cache_data
def load_and_process():
    frames = []

    for file in DATASETS:
        if not file.exists():
            raise FileNotFoundError(f"Missing dataset: {file.name}")

        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()

        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise ValueError(
                f"{file.name} is missing required columns: {missing}"
            )

        df["Dataset_Source"] = file.name
        frames.append(df)

    data = pd.concat(frames, ignore_index=True, sort=False)

    for col in FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=FEATURES)
    data = data.drop_duplicates(subset=FEATURES).reset_index(drop=True)

    if len(data) < 3:
        raise ValueError("At least 3 valid customer records are required.")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data[FEATURES])

    results = []
    for k in range(2, min(10, len(data) - 1) + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(scaled)

        results.append({
            "K": k,
            "WCSS": model.inertia_,
            "Silhouette Score": silhouette_score(scaled, labels)
        })

    evaluation = pd.DataFrame(results)
    best_row = evaluation.loc[evaluation["Silhouette Score"].idxmax()]
    best_k = int(best_row["K"])

    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    data["Cluster"] = model.fit_predict(scaled)

    # Give every cluster its own unique readable label
    cluster_means = data.groupby("Cluster")[FEATURES].mean()
    income_mid = data["Annual_Income_k"].median()
    spending_mid = data["Spending_Score"].median()

    names = {}
    for cluster, row in cluster_means.iterrows():
        income = "High Income" if row["Annual_Income_k"] >= income_mid else "Low Income"
        spending = "High Spending" if row["Spending_Score"] >= spending_mid else "Low Spending"
        names[cluster] = f"{income} - {spending} (Group {cluster + 1})"

    data["Segment"] = data["Cluster"].map(names)

    profiles = data.groupby(
        ["Cluster", "Segment"], as_index=False
    ).agg(
        Customer_Count=("Age", "size"),
        Average_Age=("Age", "mean"),
        Average_Annual_Income=("Annual_Income_k", "mean"),
        Average_Spending_Score=("Spending_Score", "mean")
    )

    data.to_csv(BASE_DIR / "Combined_Customers.csv", index=False)
    data.to_csv(BASE_DIR / "segmented_customers.csv", index=False)
    profiles.to_csv(BASE_DIR / "customer_cluster_profiles.csv", index=False)
    evaluation.to_csv(BASE_DIR / "cluster_evaluation_metrics.csv", index=False)

    return data, profiles, evaluation, best_k, float(best_row["Silhouette Score"]), float(best_row["WCSS"])


try:
    data, profiles, evaluation, best_k, best_score, best_wcss = load_and_process()

    st.success("Customer segmentation completed successfully!")

    # Key metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", len(data))
    c2.metric("Number of Clusters", best_k)
    c3.metric("Silhouette Score", f"{best_score:.4f}")
    c4.metric("WCSS", f"{best_wcss:.2f}")

    st.divider()

    st.subheader("📊 Customer Segment Summary")
    st.dataframe(profiles.round(2), use_container_width=True, hide_index=True)

    left, right = st.columns(2)

    with left:
        st.subheader("Customers in Each Cluster")
        st.bar_chart(profiles.set_index("Segment")["Customer_Count"])

    with right:
        st.subheader("Model Evaluation")
        st.dataframe(evaluation.round(4), use_container_width=True, hide_index=True)

    st.subheader("📉 Elbow Method (WCSS)")
    st.line_chart(evaluation.set_index("K")[["WCSS"]])

    st.subheader("🎯 Silhouette Score by K")
    st.line_chart(evaluation.set_index("K")[["Silhouette Score"]])

    st.subheader("Customer Income vs Spending")
    chart_data = data[["Annual_Income_k", "Spending_Score", "Segment"]].copy()
    st.scatter_chart(
        chart_data,
        x="Annual_Income_k",
        y="Spending_Score",
        color="Segment"
    )

    st.subheader("🔎 Explore Customer Records")
    selected_segment = st.selectbox(
        "Filter by customer segment",
        ["All Segments"] + sorted(data["Segment"].unique().tolist())
    )

    if selected_segment == "All Segments":
        shown = data
    else:
        shown = data[data["Segment"] == selected_segment]

    st.dataframe(shown, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download Segmented Customer Data",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="segmented_customers.csv",
        mime="text/csv"
    )

    st.info(
        "The best K is selected from the tested values using the highest "
        "Silhouette Score. Scores closer to 1 generally indicate better "
        "separation; your current score can still be improved."
    )

except Exception as error:
    st.error(str(error))
    st.info(
        "Check that both CSV files are in the same folder as main.py "
        "and contain Age, Annual_Income_k, and Spending_Score."
    )
