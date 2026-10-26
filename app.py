import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# RetainIQ — Customer Retention Intelligence Platform
# Customer Engagement & Product Utilization Analytics
# ============================================================

st.set_page_config(
    page_title="RetainIQ | Customer Retention Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
      .stApp {
        background: linear-gradient(135deg, #f7f9ff 0%, #f4f6ff 48%, #f8f7ff 100%);
      }
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101a3a 0%, #18295b 100%);
      }
      [data-testid="stSidebar"] * { color: #f4f7ff !important; }
      .brand {
        font-size: 2rem; font-weight: 850; letter-spacing: -1px;
        background: linear-gradient(90deg, #76a9ff, #d2a6ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      }
      .subtitle { color: #cbd5ff; font-size: .88rem; margin-bottom: 1.2rem; }
      .hero {
        padding: 1.4rem 1.6rem; border-radius: 20px; color: white;
        background: linear-gradient(115deg, #1d4ed8 0%, #5b4ce2 55%, #8b5cf6 100%);
        box-shadow: 0 12px 30px rgba(60, 80, 180, .16);
        margin-bottom: 1.2rem;
      }
      .hero h1 { color: white; margin: 0 0 .35rem 0; }
      .hero p { color: #eef2ff; margin: 0; font-size: 1.02rem; }
      .small-note { color: #667085; font-size: .86rem; }
      div[data-testid="stMetric"] {
        background: rgba(255,255,255,.9); border: 1px solid #e5e9f5;
        padding: 14px 16px; border-radius: 15px;
        box-shadow: 0 4px 14px rgba(35,50,100,.04);
      }
      .section-head { font-size: 1.25rem; font-weight: 750; color: #172554; margin-top: .3rem; }
      .callout {
        padding: 12px 15px; border-radius: 12px; background: #eef2ff;
        border-left: 4px solid #6366f1; color: #25315f; margin: .5rem 0 1rem 0;
      }
      footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Dataset loading ----------
DEFAULT_CANDIDATES = [
    Path("European_Bank 2.csv"),
    Path("European_Bank 3.csv"),
    Path("European_Bank.csv"),
    Path("data/European_Bank 2.csv"),
    Path("data/European_Bank 3.csv"),
    Path("data/European_Bank.csv"),
]

def load_dataset():
    uploaded = st.sidebar.file_uploader(
        "Upload another customer CSV (optional)", type=["csv"],
        help="If uploaded, this file is used instead of the default repository dataset."
    )
    if uploaded is not None:
        try:
            return pd.read_csv(uploaded), f"Uploaded file: {uploaded.name}"
        except Exception as exc:
            st.error(f"Could not read the uploaded CSV: {exc}")
            st.stop()

    for candidate in DEFAULT_CANDIDATES:
        if candidate.exists():
            try:
                return pd.read_csv(candidate), f"Default dataset: {candidate.as_posix()}"
            except Exception as exc:
                st.error(f"Found {candidate}, but could not read it: {exc}")
                st.stop()

    st.sidebar.warning("No bundled CSV found. Upload a CSV above to use the dashboard.")
    st.info(
        "To make the dashboard load automatically for evaluators, add your dataset CSV "
        "to the same GitHub repository as `app.py`. Supported default filenames include "
        "`European_Bank 2.csv`, `European_Bank 3.csv`, and `European_Bank.csv`."
    )
    st.stop()

def normalize_columns(data):
    data = data.copy()
    data.columns = [str(c).strip() for c in data.columns]
    # Match common column spelling variations without requiring exact capitalization.
    lookup = {c.lower().replace(" ", "").replace("_", ""): c for c in data.columns}
    aliases = {
        "customerid": "CustomerId", "surname": "Surname",
        "creditscore": "CreditScore", "geography": "Geography",
        "gender": "Gender", "age": "Age", "tenure": "Tenure",
        "balance": "Balance", "numofproducts": "NumOfProducts",
        "hascrcard": "HasCrCard", "isactivemember": "IsActiveMember",
        "estimatedsalary": "EstimatedSalary", "exited": "Exited",
        "year": "Year",
    }
    rename = {}
    for normalized, standard in aliases.items():
        if normalized in lookup:
            rename[lookup[normalized]] = standard
    data = data.rename(columns=rename)
    return data

def as_numeric(series):
    return pd.to_numeric(series, errors="coerce")

df, dataset_source = load_dataset()
df = normalize_columns(df)

required_cols = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"
]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    st.error(
        "The CSV is missing required columns: " + ", ".join(missing_cols) +
        ". Please upload the project dataset with the expected customer fields."
    )
    st.stop()

numeric_cols = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"
]
for col in numeric_cols:
    df[col] = as_numeric(df[col])

# Remove rows where the target is unavailable; preserve all other rows and report missingness.
rows_before = len(df)
df = df.dropna(subset=["Exited"]).copy()
df["Exited"] = df["Exited"].astype(int)
df["Exited"] = df["Exited"].clip(0, 1)
for col in ["HasCrCard", "IsActiveMember"]:
    df[col] = df[col].fillna(0).clip(0, 1).astype(int)
df["NumOfProducts"] = df["NumOfProducts"].fillna(0).clip(lower=0)
df["Balance"] = df["Balance"].fillna(0).clip(lower=0)
df["EstimatedSalary"] = df["EstimatedSalary"].fillna(0).clip(lower=0)
df["Age"] = df["Age"].fillna(df["Age"].median())
df["CreditScore"] = df["CreditScore"].fillna(df["CreditScore"].median())
df["Tenure"] = df["Tenure"].fillna(df["Tenure"].median())
df["Geography"] = df["Geography"].fillna("Unknown").astype(str)
df["Gender"] = df["Gender"].fillna("Unknown").astype(str)
if "CustomerId" not in df.columns:
    df["CustomerId"] = [f"Customer-{i+1}" for i in range(len(df))]
if "Surname" not in df.columns:
    df["Surname"] = "Not provided"

# Derived labels and metrics
df["ChurnStatus"] = np.where(df["Exited"].eq(1), "Churned", "Retained")
df["EngagementStatus"] = np.where(df["IsActiveMember"].eq(1), "Active", "Inactive")
df["HasCreditCardLabel"] = np.where(df["HasCrCard"].eq(1), "Credit card holder", "No credit card")
df["BalanceBand"] = np.where(df["Balance"] >= 100000, "High balance (≥100k)", "Balance below 100k")
df["ProductBand"] = df["NumOfProducts"].apply(
    lambda x: f"{int(x)} product(s)" if float(x).is_integer() else f"{x:g} products"
)

# Transparent exploratory Relationship Strength Index: 0–100, not a trained model.
product_component = (df["NumOfProducts"].clip(0, 4) / 4) * 30
activity_component = df["IsActiveMember"] * 30
card_component = df["HasCrCard"] * 20
balance_component = (df["Balance"].gt(0).astype(int)) * 20
df["RelationshipStrengthIndex"] = (
    activity_component + product_component + card_component + balance_component
).round(1)
df["RelationshipStrengthBand"] = pd.cut(
    df["RelationshipStrengthIndex"],
    bins=[-0.1, 39.9, 69.9, 100],
    labels=["Needs attention", "Developing", "Strong"],
).astype(str)

# High-value risk flag: editable thresholds on the Customer Risk page.
# Default labels are re-evaluated using selected thresholds on that page.

# ---------- Sidebar navigation ----------
st.sidebar.markdown('<div class="brand">RetainIQ</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div class="subtitle">Customer Retention Intelligence Platform</div>',
    unsafe_allow_html=True,
)
st.sidebar.caption(dataset_source)
pages = [
    "Executive Overview",
    "Engagement Intelligence",
    "Product Utilization",
    "High-Value Customer Detector",
    "Retention Strength Assessment",
    "Segmentation Studio",
    "Data Explorer & Methodology",
]
page = st.sidebar.radio("NAVIGATION", pages, index=0)
st.sidebar.markdown("---")
st.sidebar.markdown("**Project focus**")
st.sidebar.caption("Customer engagement • Product usage • Retention strategy")
st.sidebar.markdown(
    '<div class="small-note">Understand customer behavior. Predict churn risk. Build stronger relationships.</div>',
    unsafe_allow_html=True,
)

# ---------- Common functions ----------
def pct(numerator, denominator):
    return float(numerator) / float(denominator) * 100 if denominator else 0.0

def churn_rate(data):
    return pct(data["Exited"].sum(), len(data))

def retention_rate(data):
    return 100 - churn_rate(data) if len(data) else 0.0

def fmt_pct(value):
    return f"{value:.2f}%"

def grouped_churn(data, group_col, label_col=None):
    out = data.groupby(group_col, dropna=False).agg(
        Customers=("Exited", "size"),
        Churned=("Exited", "sum"),
        Retention=("Exited", lambda x: (1 - x.mean()) * 100),
    ).reset_index()
    out["Churn rate"] = out["Churned"] / out["Customers"] * 100
    out["Retention rate"] = out["Retention"]
    if label_col and group_col != label_col:
        out = out.rename(columns={group_col: label_col})
    return out

def chart_layout(fig, height=360):
    fig.update_layout(
        height=height, margin=dict(l=15, r=15, t=55, b=15),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#25315f"),
        title_font=dict(size=17, color="#172554"),
        legend_title_text="",
    )
    return fig

def section_title(title, description=None):
    st.markdown(f'<div class="section-head">{title}</div>', unsafe_allow_html=True)
    if description:
        st.caption(description)

def kpi_row(data):
    total = len(data)
    churned = int(data["Exited"].sum())
    retained = total - churned
    active_rate = pct(data["IsActiveMember"].sum(), total)
    avg_products = data["NumOfProducts"].mean() if total else 0
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Customers", f"{total:,}")
    c2.metric("Churn rate", fmt_pct(churn_rate(data)))
    c3.metric("Retention rate", fmt_pct(retention_rate(data)))
    c4.metric("Active members", fmt_pct(active_rate))
    c5.metric("Avg. products", f"{avg_products:.2f}")

def high_balance_disengagement(data, threshold):
    high_balance = data[data["Balance"] >= threshold]
    if len(high_balance) == 0:
        return 0.0, 0, 0
    inactive_high = high_balance[high_balance["IsActiveMember"] == 0]
    return pct(len(inactive_high), len(high_balance)), len(inactive_high), len(high_balance)

def download_csv(data, filename="retainiQ_filtered_customers.csv"):
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download current data as CSV", data=csv, file_name=filename,
        mime="text/csv", use_container_width=False
    )

# ---------- Page header ----------
st.markdown(
    """
    <div class="hero">
      <h1>RetainIQ</h1>
      <p>Customer Engagement &amp; Product Utilization Analytics for Retention Strategy</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    f"Data source: {dataset_source} • Rows available for analysis: {len(df):,} "
    f"• Rows removed due to missing churn target: {rows_before - len(df):,}"
)

# ---------- Page 1: Executive Overview ----------
if page == "Executive Overview":
    st.subheader("Executive Overview")
    st.write("A high-level view of retention, engagement, and product relationships in the customer dataset.")
    kpi_row(df)

    left, right = st.columns(2)
    with left:
        section_title("Retention outcome", "Retained customers compared with churned customers.")
        outcome = df["ChurnStatus"].value_counts().rename_axis("Status").reset_index(name="Customers")
        fig = px.pie(outcome, names="Status", values="Customers", hole=0.58,
                     color="Status", color_discrete_map={"Retained": "#4f46e5", "Churned": "#f97316"})
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(chart_layout(fig, 350), use_container_width=True)
    with right:
        section_title("Churn by engagement status", "Compare observed churn rates for active and inactive members.")
        eng = grouped_churn(df, "EngagementStatus")
        fig = px.bar(eng, x="EngagementStatus", y="Churn rate", text=eng["Churn rate"].map(lambda x: f"{x:.1f}%"),
                     color="EngagementStatus", color_discrete_map={"Active": "#4f46e5", "Inactive": "#f97316"})
        fig.update_layout(yaxis_title="Churn rate (%)", xaxis_title="")
        st.plotly_chart(chart_layout(fig, 350), use_container_width=True)

    left, right = st.columns(2)
    with left:
        section_title("Product count and retention")
        product = grouped_churn(df, "NumOfProducts").sort_values("NumOfProducts")
        product["NumOfProducts"] = product["NumOfProducts"].astype(str)
        fig = px.bar(product, x="NumOfProducts", y="Retention rate", text=product["Retention rate"].map(lambda x: f"{x:.1f}%"))
        fig.update_traces(marker_color="#6366f1", textposition="outside")
        fig.update_layout(xaxis_title="Number of products", yaxis_title="Retention rate (%)")
        st.plotly_chart(chart_layout(fig, 340), use_container_width=True)
    with right:
        section_title("Customer activity and balance")
        fig = px.box(df, x="EngagementStatus", y="Balance", color="EngagementStatus",
                     color_discrete_map={"Active": "#4f46e5", "Inactive": "#f97316"},
                     points=False)
        fig.update_layout(xaxis_title="", yaxis_title="Account balance")
        st.plotly_chart(chart_layout(fig, 340), use_container_width=True)

    st.markdown('<div class="callout"><b>Interpretation note:</b> These charts describe associations in the available dataset. They do not prove that a behavior causes churn.</div>', unsafe_allow_html=True)

# ---------- Page 2: Engagement Intelligence ----------
elif page == "Engagement Intelligence":
    st.subheader("Engagement Intelligence")
    st.write("Measure the relationship between customer activity and retention.")
    active = df[df["IsActiveMember"] == 1]
    inactive = df[df["IsActiveMember"] == 0]
    active_churn = churn_rate(active)
    inactive_churn = churn_rate(inactive)
    active_retention = retention_rate(active)
    inactive_retention = retention_rate(inactive)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active customer churn", fmt_pct(active_churn))
    c2.metric("Inactive customer churn", fmt_pct(inactive_churn))
    c3.metric("Active customer retention", fmt_pct(active_retention))
    c4.metric("Inactive customer retention", fmt_pct(inactive_retention))
    st.caption("Engagement Retention Ratio is defined here as active-member retention rate divided by inactive-member retention rate. It is a ratio, not a percentage.")
    ratio = active_retention / inactive_retention if inactive_retention else np.nan
    st.metric("Engagement Retention Ratio", f"{ratio:.3f}×" if pd.notna(ratio) else "N/A")

    eng_table = grouped_churn(df, "EngagementStatus")
    eng_table = eng_table[["EngagementStatus", "Customers", "Churned", "Churn rate", "Retention rate"]]
    st.dataframe(eng_table.style.format({"Churn rate": "{:.2f}%", "Retention rate": "{:.2f}%"}), use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        section_title("Engagement by geography")
        geo = df.groupby("Geography").agg(
            Customers=("Exited", "size"),
            ChurnRate=("Exited", lambda x: x.mean() * 100),
            ActiveShare=("IsActiveMember", lambda x: x.mean() * 100),
        ).reset_index()
        fig = px.bar(geo, x="Geography", y="ChurnRate", color="Geography",
                     text=geo["ChurnRate"].map(lambda x: f"{x:.1f}%"))
        fig.update_layout(xaxis_title="", yaxis_title="Churn rate (%)")
        st.plotly_chart(chart_layout(fig), use_container_width=True)
    with right:
        section_title("Age group and churn")
        age_bins = [0, 25, 35, 45, 55, 65, 120]
        age_labels = ["≤25", "26–35", "36–45", "46–55", "56–65", "66+"]
        age_df = df.copy()
        age_df["Age group"] = pd.cut(age_df["Age"], bins=age_bins, labels=age_labels, include_lowest=True)
        age = age_df.groupby("Age group", observed=False).agg(
            Customers=("Exited", "size"), ChurnRate=("Exited", lambda x: x.mean() * 100)
        ).reset_index()
        fig = px.bar(age, x="Age group", y="ChurnRate", text=age["ChurnRate"].map(lambda x: f"{x:.1f}%"))
        fig.update_traces(marker_color="#8b5cf6")
        fig.update_layout(xaxis_title="Age group", yaxis_title="Churn rate (%)")
        st.plotly_chart(chart_layout(fig), use_container_width=True)

    st.markdown("**Suggested retention actions**")
    st.write("- Identify inactive customers for a targeted check-in or re-engagement campaign.")
    st.write("- Compare campaign outcomes across customer groups before scaling an intervention.")
    st.write("- Monitor churn and engagement trends over time if dated customer activity data becomes available.")

# ---------- Page 3: Product Utilization ----------
elif page == "Product Utilization":
    st.subheader("Product Utilization")
    st.write("Analyze product depth and how product count relates to retention.")
    product_summary = df.groupby("NumOfProducts").agg(
        Customers=("Exited", "size"),
        Churned=("Exited", "sum"),
        ChurnRate=("Exited", lambda x: x.mean() * 100),
        RetentionRate=("Exited", lambda x: (1 - x.mean()) * 100),
        AvgBalance=("Balance", "mean"),
        ActiveShare=("IsActiveMember", lambda x: x.mean() * 100),
    ).reset_index().sort_values("NumOfProducts")
    product_summary["Product Depth Index"] = product_summary["NumOfProducts"] / max(df["NumOfProducts"].max(), 1) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Average products per customer", f"{df['NumOfProducts'].mean():.2f}")
    c2.metric("Customers with 1 product", f"{pct((df['NumOfProducts'] == 1).sum(), len(df)):.1f}%")
    c3.metric("Customers with 2+ products", f"{pct((df['NumOfProducts'] >= 2).sum(), len(df)):.1f}%")

    left, right = st.columns(2)
    with left:
        section_title("Customer count by product count")
        fig = px.bar(product_summary, x="NumOfProducts", y="Customers", text="Customers")
        fig.update_traces(marker_color="#4f46e5")
        fig.update_layout(xaxis_title="Number of products", yaxis_title="Customers")
        st.plotly_chart(chart_layout(fig), use_container_width=True)
    with right:
        section_title("Retention rate by product count")
        fig = px.bar(product_summary, x="NumOfProducts", y="RetentionRate",
                     text=product_summary["RetentionRate"].map(lambda x: f"{x:.1f}%"))
        fig.update_traces(marker_color="#8b5cf6", textposition="outside")
        fig.update_layout(xaxis_title="Number of products", yaxis_title="Retention rate (%)")
        st.plotly_chart(chart_layout(fig), use_container_width=True)

    section_title("Product utilization summary")
    st.dataframe(
        product_summary.rename(columns={
            "NumOfProducts": "Products per customer", "ChurnRate": "Churn rate (%)",
            "RetentionRate": "Retention rate (%)", "AvgBalance": "Average balance",
            "ActiveShare": "Active members (%)"
        }).style.format({
            "Churn rate (%)": "{:.2f}", "Retention rate (%)": "{:.2f}",
            "Average balance": "{:,.2f}", "Active members (%)": "{:.2f}",
            "Product Depth Index": "{:.1f}"
        }),
        use_container_width=True, hide_index=True
    )
    st.caption("Product Depth Index is defined for this dashboard as product count scaled to the maximum product count observed in the dataset (0–100). It is descriptive, not a validated industry benchmark.")
    st.markdown("**Recommendations**")
    st.write("- Review customers with low product depth and low engagement for relevant, needs-based offers.")
    st.write("- Avoid assuming that more products automatically cause retention; compare customer groups and monitor outcomes.")
    st.write("- Investigate very small product-count groups separately because their churn rates may be unstable.")

# ---------- Page 4: High-Value Customer Detector ----------
elif page == "High-Value Customer Detector":
    st.subheader("High-Value Customer Detector")
    st.write("Find potentially valuable customers who may need proactive engagement.")
    c1, c2, c3 = st.columns(3)
    default_balance = float(df["Balance"].quantile(0.75)) if len(df) else 100000.0
    default_salary = float(df["EstimatedSalary"].quantile(0.75)) if len(df) else 80000.0
    balance_threshold = c1.number_input("High-balance threshold", min_value=0.0, value=round(default_balance, 2), step=5000.0)
    salary_threshold = c2.number_input("High-salary threshold", min_value=0.0, value=round(default_salary, 2), step=5000.0)
    max_products = int(max(df["NumOfProducts"].max(), 1))

max_product_filter = c3.slider(
    "Maximum products for low product-depth flag",
    min_value=0,
    max_value=max_products,
    value=min(2, max_products)
)


    active_filter = st.selectbox("Engagement filter", ["All", "Inactive only", "Active only"])
    churn_filter = st.selectbox("Churn status filter", ["All", "Retained only", "Churned only"])
    geo_options = ["All"] + sorted(df["Geography"].dropna().astype(str).unique().tolist())
    selected_geo = st.selectbox("Geography filter", geo_options)

    risk = df.copy()
    risk["HighBalance"] = risk["Balance"] >= balance_threshold
    risk["HighSalary"] = risk["EstimatedSalary"] >= salary_threshold
    risk["LowProductDepth"] = risk["NumOfProducts"] <= max_product_filter
    risk["Disengaged"] = risk["IsActiveMember"] == 0
    risk["PotentiallyHighValueAtRisk"] = (
        (risk["HighBalance"] | risk["HighSalary"]) & (risk["Disengaged"] | risk["LowProductDepth"])
    )
    if active_filter == "Inactive only":
        risk = risk[risk["IsActiveMember"] == 0]
    elif active_filter == "Active only":
        risk = risk[risk["IsActiveMember"] == 1]
    if churn_filter == "Retained only":
        risk = risk[risk["Exited"] == 0]
    elif churn_filter == "Churned only":
        risk = risk[risk["Exited"] == 1]
    if selected_geo != "All":
        risk = risk[risk["Geography"].astype(str) == selected_geo]

    candidates = risk[risk["PotentiallyHighValueAtRisk"]].copy()
    high_disengagement_rate, inactive_high_count, high_balance_count = high_balance_disengagement(df, balance_threshold)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Potentially at-risk customers", f"{len(candidates):,}")
    c2.metric("High-balance disengagement rate", fmt_pct(high_disengagement_rate))
    c3.metric("Inactive high-balance customers", f"{inactive_high_count:,}")
    c4.metric("High-balance customers", f"{high_balance_count:,}")

    st.caption("High-Balance Disengagement Rate = inactive customers whose balance meets the selected threshold ÷ all customers whose balance meets that threshold. High-value at-risk flag is a transparent rule, not a predictive model.")
    display_cols = [
        "CustomerId", "Surname", "Geography", "Age", "Gender", "Balance",
        "EstimatedSalary", "NumOfProducts", "IsActiveMember", "HasCrCard",
        "Exited", "RelationshipStrengthIndex"
    ]
    display_cols = [c for c in display_cols if c in candidates.columns]
    if len(candidates):
        st.dataframe(candidates[display_cols].sort_values(
            ["Balance", "EstimatedSalary"], ascending=False
        ), use_container_width=True, hide_index=True)
        download_csv(candidates[display_cols], "retainiQ_high_value_at_risk_customers.csv")
    else:
        st.info("No customers match the current thresholds and filters. Try lowering the thresholds or changing filters.")

    st.markdown("**Recommended follow-up**")
    st.write("- Validate the customer list with business rules before contacting customers.")
    st.write("- Use personalized, consent-respecting outreach; do not infer sensitive personal circumstances.")
    st.write("- Track outcomes to evaluate whether the intervention improves retention.")

# ---------- Page 5: Retention Strength Assessment ----------
elif page == "Retention Strength Assessment":
    st.subheader("Retention Strength Assessment")
    st.write("An interpretable, rule-based relationship strength score using activity and product indicators.")
    st.markdown(
        '<div class="callout"><b>Important:</b> This score is an exploratory heuristic for analysis. '
        'It is not a trained machine-learning model and has not been validated to predict future churn.</div>',
        unsafe_allow_html=True
    )
    st.markdown("**Scoring formula (0–100)**")
    st.write("- Active member: 30 points")
    st.write("- Product count: up to 30 points, scaled as (product count ÷ 4) × 30")
    st.write("- Has credit card: 20 points")
    st.write("- Positive account balance: 20 points")
    st.write("The score is capped naturally by the component ranges; product count contribution is capped at four products.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Average relationship score", f"{df['RelationshipStrengthIndex'].mean():.1f}/100")
    c2.metric("Strong relationship band", f"{pct((df['RelationshipStrengthBand'] == 'Strong').sum(), len(df)):.1f}%")
    c3.metric("Needs-attention band", f"{pct((df['RelationshipStrengthBand'] == 'Needs attention').sum(), len(df)):.1f}%")

    band = df.groupby("RelationshipStrengthBand", dropna=False).agg(
        Customers=("Exited", "size"),
        ChurnRate=("Exited", lambda x: x.mean() * 100),
        AverageScore=("RelationshipStrengthIndex", "mean"),
    ).reset_index()
    left, right = st.columns(2)
    with left:
        section_title("Customers by relationship band")
        fig = px.bar(band, x="RelationshipStrengthBand", y="Customers", color="RelationshipStrengthBand",
                     color_discrete_map={"Needs attention": "#f97316", "Developing": "#6366f1", "Strong": "#10b981"})
        fig.update_layout(xaxis_title="", yaxis_title="Customers")
        st.plotly_chart(chart_layout(fig), use_container_width=True)
    with right:
        section_title("Observed churn rate by relationship band")
        fig = px.bar(band, x="RelationshipStrengthBand", y="ChurnRate",
                     text=band["ChurnRate"].map(lambda x: f"{x:.1f}%"),
                     color="RelationshipStrengthBand",
                     color_discrete_map={"Needs attention": "#f97316", "Developing": "#6366f1", "Strong": "#10b981"})
        fig.update_layout(xaxis_title="", yaxis_title="Observed churn rate (%)")
        st.plotly_chart(chart_layout(fig), use_container_width=True)

    st.dataframe(band.rename(columns={
        "RelationshipStrengthBand": "Relationship band",
        "ChurnRate": "Observed churn rate (%)",
        "AverageScore": "Average score"
    }).style.format({"Observed churn rate (%)": "{:.2f}", "Average score": "{:.1f}"}),
        use_container_width=True, hide_index=True)

    st.markdown("**Credit card stickiness score**")
    st.caption("Defined as the retention rate of credit-card holders minus the retention rate of customers without a credit card, in percentage points.")
    card = df.groupby("HasCreditCardLabel").agg(
        Customers=("Exited", "size"),
        RetentionRate=("Exited", lambda x: (1 - x.mean()) * 100),
        ChurnRate=("Exited", lambda x: x.mean() * 100),
    ).reset_index()
    holder = card.loc[card["HasCreditCardLabel"] == "Credit card holder", "RetentionRate"]
    nonholder = card.loc[card["HasCreditCardLabel"] == "No credit card", "RetentionRate"]
    stickiness = float(holder.iloc[0] - nonholder.iloc[0]) if len(holder) and len(nonholder) else np.nan
    m1, m2, m3 = st.columns(3)
    m1.metric("Card-holder retention", fmt_pct(float(holder.iloc[0])) if len(holder) else "N/A")
    m2.metric("Non-holder retention", fmt_pct(float(nonholder.iloc[0])) if len(nonholder) else "N/A")
    m3.metric("Credit Card Stickiness Score", f"{stickiness:+.2f} pp" if pd.notna(stickiness) else "N/A")
    st.dataframe(card.style.format({"RetentionRate": "{:.2f}%", "ChurnRate": "{:.2f}%"}), use_container_width=True, hide_index=True)

    st.markdown("**Customer-level score explorer**")
    score_min, score_max = st.slider("Relationship score range", 0, 100, (0, 100))
    score_view = df[df["RelationshipStrengthIndex"].between(score_min, score_max)]
    columns = ["CustomerId", "Geography", "Age", "Balance", "NumOfProducts",
               "IsActiveMember", "HasCrCard", "RelationshipStrengthIndex",
               "RelationshipStrengthBand", "ChurnStatus"]
    st.dataframe(score_view[[c for c in columns if c in score_view.columns]].head(1000),
                 use_container_width=True, hide_index=True)
    download_csv(score_view[[c for c in columns if c in score_view.columns]],
                 "retainiQ_relationship_strength_assessment.csv")

# ---------- Page 6: Segmentation Studio ----------
elif page == "Segmentation Studio":
    st.subheader("Segmentation Studio")
    st.write("Filter customer segments to explore patterns and export the matching records.")
    with st.expander("Segment filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        geos = ["All"] + sorted(df["Geography"].dropna().astype(str).unique().tolist())
        geo = c1.selectbox("Geography", geos, key="seg_geo")
        genders = ["All"] + sorted(df["Gender"].dropna().astype(str).unique().tolist())
        gender = c2.selectbox("Gender", genders, key="seg_gender")
        engagement = c3.selectbox("Engagement", ["All", "Active", "Inactive"], key="seg_engagement")

        c4, c5, c6 = st.columns(3)
        age_min = int(max(0, np.floor(df["Age"].min())))
        age_max = int(min(120, np.ceil(df["Age"].max())))
        age_range = c4.slider("Age range", age_min, max(age_min + 1, age_max),
                              (age_min, max(age_min + 1, age_max)))
        prod_min = int(max(0, np.floor(df["NumOfProducts"].min())))
        prod_max = int(max(prod_min + 1, np.ceil(df["NumOfProducts"].max())))
        product_range = c5.slider("Product count", prod_min, prod_max, (prod_min, prod_max))
        balance_min = float(df["Balance"].min())
        balance_max = float(df["Balance"].max())
        balance_range = c6.slider("Balance range", min_value=balance_min, max_value=max(balance_min + 1, balance_max),
                                  value=(balance_min, max(balance_min + 1, balance_max)))

        c7, c8 = st.columns(2)
        status = c7.selectbox("Outcome", ["All", "Retained", "Churned"], key="seg_outcome")
        card = c8.selectbox("Credit card", ["All", "Has card", "No card"], key="seg_card")

    seg = df.copy()
    if geo != "All":
        seg = seg[seg["Geography"].astype(str) == geo]
    if gender != "All":
        seg = seg[seg["Gender"].astype(str) == gender]
    if engagement == "Active":
        seg = seg[seg["IsActiveMember"] == 1]
    elif engagement == "Inactive":
        seg = seg[seg["IsActiveMember"] == 0]
    seg = seg[seg["Age"].between(age_range[0], age_range[1])]
    seg = seg[seg["NumOfProducts"].between(product_range[0], product_range[1])]
    seg = seg[seg["Balance"].between(balance_range[0], balance_range[1])]
    if status == "Retained":
        seg = seg[seg["Exited"] == 0]
    elif status == "Churned":
        seg = seg[seg["Exited"] == 1]
    if card == "Has card":
        seg = seg[seg["HasCrCard"] == 1]
    elif card == "No card":
        seg = seg[seg["HasCrCard"] == 0]

    kpi_row(seg)
    section_title("Segment churn by geography")
    if len(seg):
        geo_seg = seg.groupby("Geography").agg(
            Customers=("Exited", "size"), ChurnRate=("Exited", lambda x: x.mean() * 100)
        ).reset_index()
        fig = px.bar(geo_seg, x="Geography", y="ChurnRate", color="Geography",
                     text=geo_seg["ChurnRate"].map(lambda x: f"{x:.1f}%"))
        fig.update_layout(xaxis_title="", yaxis_title="Churn rate (%)")
        st.plotly_chart(chart_layout(fig), use_container_width=True)
        st.dataframe(seg, use_container_width=True, hide_index=True)
        download_csv(seg)
    else:
        st.info("No customers match the selected filters. Broaden the filters to see results.")

# ---------- Page 7: Data Explorer & Methodology ----------
elif page == "Data Explorer & Methodology":
    st.subheader("Data Explorer & Methodology")
    st.write("Inspect dataset quality, understand KPI definitions, and review the analysis limitations.")
    total_cells = int(df.shape[0] * df.shape[1])
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing cells after preparation", f"{missing_cells:,}")
    c4.metric("Duplicate rows", f"{duplicate_rows:,}")

    section_title("Data preview")
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)
    download_csv(df, "retainiQ_prepared_customer_dataset.csv")

    section_title("Data quality summary")
    quality = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(df[c].dtype) for c in df.columns],
        "Missing values": [int(df[c].isna().sum()) for c in df.columns],
        "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    st.dataframe(quality, use_container_width=True, hide_index=True)

    section_title("KPI definitions")
    definitions = pd.DataFrame([
        ["Engagement Retention Ratio", "Active-member retention rate ÷ inactive-member retention rate. Values above 1 mean the active group has a higher observed retention rate."],
        ["Product Depth Index", "Product count scaled against the maximum product count in this dataset to a 0–100 scale. Descriptive and dataset-relative."],
        ["High-Balance Disengagement Rate", "Among customers whose balance meets the selected threshold, the percentage who are inactive."],
        ["Credit Card Stickiness Score", "Retention rate for card holders minus retention rate for non-holders, in percentage points."],
        ["Relationship Strength Index", "Exploratory 0–100 heuristic: active membership (30), product count up to 30, credit card (20), positive balance (20). Not validated."],
        ["Churn rate", "Number of records with Exited = 1 divided by all analyzed customer records."],
        ["Retention rate", "100% minus churn rate."],
    ], columns=["KPI", "Definition"])
    st.dataframe(definitions, use_container_width=True, hide_index=True)

    section_title("Methodology")
    st.markdown("""
    1. **Load and validate:** read the bundled CSV or optional uploaded CSV; check required columns.
    2. **Prepare:** standardize common column names, convert numeric fields, and exclude records without a churn outcome.
    3. **Explore:** compare churn/retention by engagement, geography, age group, product count, and credit-card status.
    4. **Identify potential risk:** apply transparent balance, salary, engagement, and product-depth thresholds.
    5. **Score relationship strength:** calculate a rule-based exploratory score with its formula documented.
    6. **Recommend action:** use the observed patterns to prioritize experiments, then measure actual campaign results.
    """)
    section_title("Limitations and responsible use")
    st.markdown("""
    - The dashboard reports associations, not proof of causation.
    - The rule-based risk flag and Relationship Strength Index are not validated future-churn predictions.
    - Thresholds are adjustable and should be selected with business stakeholders.
    - The dataset may not contain dates, contact history, product categories, campaign exposure, or customer consent data.
    - Do not use the dashboard alone for consequential decisions about individual customers.
    - Validate recommendations with domain experts and track outcomes before operational use.
    """)

# ---------- Footer ----------
st.markdown("---")
st.caption(
    "RetainIQ • Customer Retention Intelligence Platform • Built for the Customer Engagement & "
    "Product Utilization Analytics for Retention Strategy project."
)
