# streamlit_clustering.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import io

sns.set_style("whitegrid")

# ------------------------------
# Helper functions
# ------------------------------
@st.cache_data
def load_sample_data():
    # if you want a quick sample similar to Mall dataset:
    np.random.seed(42)
    n = 200
    ages = np.random.randint(18, 70, n)
    income = np.random.randint(15, 150, n)  # k$
    score = np.random.randint(1, 100, n)
    df = pd.DataFrame({
        "Age": ages,
        "Annual Income (k$)": income,
        "Spending Score (1-100)": score
    })
    return df

def clean_numeric_dataframe(df):
    """Keep only numeric columns (int, float). Convert if possible."""
    # Try to coerce convertible columns to numeric
    for col in df.columns:
        # skip if already numeric
        if df[col].dtype == object:
            # attempt to remove commas and coerce
            try:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='ignore')
            except Exception:
                pass
    numeric_df = df.select_dtypes(include=['int64', 'float64']).copy()
    return numeric_df

@st.cache_data
def compute_wcss(X_scaled, max_k=10):
    wcss = []
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
    return wcss

def run_kmeans(X_scaled, k):
    km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels

def describe_clusters(df_numeric, labels):
    dfc = df_numeric.copy()
    dfc['Cluster'] = labels
    summary = dfc.groupby('Cluster').mean().round(2)
    counts = dfc['Cluster'].value_counts().sort_index()
    summary['Count'] = counts
    return summary

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Customer Segmentation — KMeans Clustering Dashboard")
st.markdown("""
Upload your customer CSV (recommended columns: Age, Annual Income (k$), Spending Score (1-100)).
Or use the sample dataset to experiment.
""")

# Upload or sample
uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
use_sample = st.checkbox("Use sample dataset instead", value=False)

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file, encoding='utf-8')
    except Exception:
        # try latin-1 if utf-8 fails
        df_raw = pd.read_csv(uploaded_file, encoding='latin-1')
    st.success("File loaded successfully!")
elif use_sample:
    df_raw = load_sample_data()
    st.info("Using generated sample dataset.")
else:
    st.info("Upload a CSV or check 'Use sample dataset instead' to try demo data.")
    st.stop()

st.subheader("Raw Data (first 10 rows)")
st.dataframe(df_raw.head(10))

# Clean and keep numeric features
df_numeric = clean_numeric_dataframe(df_raw)
if df_numeric.shape[1] == 0:
    st.error("No numeric columns found. Please upload a CSV with numeric features (Age, Income, Spending Score, etc.).")
    st.stop()

st.subheader("Numeric features detected")
st.write(df_numeric.columns.tolist())

# Allow user to pick features to use
with st.expander("Choose features to include for clustering", expanded=True):
    selected_features = st.multiselect("Select numeric columns", df_numeric.columns.tolist(), default=df_numeric.columns.tolist())
    if len(selected_features) == 0:
        st.error("Select at least one numeric column.")
        st.stop()

X = df_numeric[selected_features].copy()

# Option to scale
scale_option = st.checkbox("Scale features (StandardScaler)", value=True)
if scale_option:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
else:
    X_scaled = X.values

# Elbow method
st.subheader("Elbow Method (WCSS)")
max_k = st.slider("Max clusters to test for Elbow", 5, 15, 8)
wcss = compute_wcss(X_scaled, max_k=max_k)
fig1, ax1 = plt.subplots(figsize=(7,4))
ax1.plot(range(1, max_k+1), wcss, marker='o')
ax1.set_xlabel("Number of clusters (k)")
ax1.set_ylabel("WCSS (Inertia)")
ax1.set_title("Elbow Method")
st.pyplot(fig1)

# Optionally show silhouette for range
st.subheader("Silhouette score (optional quick check)")
sil_scores = {}
for k in range(2, min(11, max_k+1)):
    km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    sil = silhouette_score(X_scaled, km_tmp.labels_)
    sil_scores[k] = sil
st.write(pd.Series(sil_scores).sort_values(ascending=False).round(3))

# Choose k (either slider or select from elbow)
k_choice = st.slider("Choose number of clusters (k)", 2, min(15, max_k), 5)

# Run KMeans
if st.button("Run K-Means"):
    km_model, labels = run_kmeans(X_scaled, k_choice)
    df_result = df_raw.copy()
    df_result['Cluster'] = labels

    st.success(f"K-Means finished with k={k_choice}.")
    st.subheader("Cluster counts")
    st.write(df_result['Cluster'].value_counts().sort_index())

    # Visualization: pairplot-like scatter matrix for first two selected features
    st.subheader("Cluster visualization")
    # if at least 2 features selected, show 2D scatter; else show histogram
    if len(selected_features) >= 2:
        x_feat = selected_features[0]
        y_feat = selected_features[1]
        fig2, ax2 = plt.subplots(figsize=(8,6))
        palette = sns.color_palette("tab10", k_choice)
        sns.scatterplot(
            x=df_result[x_feat], y=df_result[y_feat],
            hue=df_result['Cluster'], palette=palette, ax=ax2, s=60, edgecolor='k'
        )
        ax2.set_xlabel(x_feat)
        ax2.set_ylabel(y_feat)
        ax2.set_title(f"Clusters (k={k_choice}) on {x_feat} vs {y_feat}")
        ax2.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig2)
    else:
        st.write("Only one feature selected — showing histogram by cluster.")
        fig3, ax3 = plt.subplots()
        for c in sorted(df_result['Cluster'].unique()):
            sns.kdeplot(df_result.loc[df_result['Cluster']==c, selected_features[0]], label=f"Cluster {c}", ax=ax3)
        ax3.set_title("Feature distribution per cluster")
        st.pyplot(fig3)

    # Show cluster summary (numeric only)
    st.subheader("Cluster summary (numeric features mean and counts)")
    summary = describe_clusters(df_numeric[selected_features], labels)
    st.dataframe(summary)

    # Provide textual summary for each cluster
    st.subheader("Cluster characteristics (short text summary)")
    for cl in summary.index:
        desc = summary.loc[cl].to_dict()
        count = int(desc.pop("Count", 0)) if "Count" in desc else 0
        text = f"Cluster {cl} (n={count}): "
        parts = []
        for kf, kv in desc.items():
            parts.append(f"{kf}={kv}")
        text += ", ".join(parts)
        st.write(text)

    # Download results
    csv = df_result.to_csv(index=False).encode('utf-8')
    st.download_button("Download clustered CSV", data=csv, file_name="clustered_customers.csv", mime="text/csv")

    # Optionally show inertia and centroids
    st.write("Model inertia (WCSS):", float(km_model.inertia_))
    if hasattr(km_model, 'cluster_centers_'):
        centroids = km_model.cluster_centers_
        st.write("Centroids (in scaled space):")
        st.write(pd.DataFrame(centroids, columns=selected_features))

