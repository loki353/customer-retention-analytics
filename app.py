import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------
# BANKPULSE INTELLIGENCE
# Customer Engagement & Product Utilization Analytics
# --------------------------------------------------

st.set_page_config(
    page_title="RetainIQ | Customer Intelligence",
    page_icon="📌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- DESIGN ----------------------

st.markdown("""
<style>
   @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F7F8FC;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #171B45 0%, #302A68 100%);
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

.hero {
    background: linear-gradient(115deg, #3155D9 0%, #7651D5 100%);
    padding: 32px 36px;
    border-radius: 22px;
    color: white;
    margin-bottom: 26px;
    box-shadow: 0 12px 32px rgba(75, 80, 190, 0.16);
}

.hero h1 {
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 10px;
    color: white;
}

.hero p {
    font-size: 14px;
    color: #F0F1FF;
    line-height: 1.8;
}

.eyebrow {
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 11px;
    font-weight: 700;
    color: #D9D5FF;
}

.metric-card {
    background: #FFFFFF;
    padding: 22px;
    border: 1px solid #EAECF5;
    border-radius: 18px;
    min-height: 130px;
    box-shadow: 0 5px 20px rgba(31, 41, 85, 0.045);
}

.metric-label {
    color: #7A8199;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 12px;
}

.metric-value {
    color: #252C58;
    font-size: 29px;
    font-weight: 800;
    margin-bottom: 7px;
}

.metric-note {
    color: #858BA2;
    font-size: 11px;
}

.section-title {
    color: #252C58;
    font-size: 21px;
    font-weight: 800;
    margin-top: 16px;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #81879C;
    font-size: 13px;
    margin-bottom: 18px;
}

.insight {
    background: #FFFFFF;
    border-left: 4px solid #6554D9;
    border-radius: 12px;
    padding: 18px 20px;
    margin: 10px 0 18px 0;
    color: #414761;
    font-size: 13px;
    line-height: 1.8;
    box-shadow: 0 4px 16px rgba(31, 41, 85, 0.04);
}

.warning {
    background: #FFF8E8;
    border-left: 4px solid #E8A23A;
    border-radius: 12px;
    padding: 16px;
    color: #684B16;
    font-size: 13px;
    line-height: 1.8;
}

div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #EAECF5;
    border-radius: 14px;
    padding: 16px;
}

.stTabs [data-baseweb="tab"] {
    background: #FFFFFF;
    border-radius: 10px;
    padding: 10px 16px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- DATA LOADING ----------------

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def prepare_data(data):
    data = data.copy()
    data.columns = data.columns.str.strip()

    required = [
        "Exited", "IsActiveMember", "NumOfProducts",
        "Geography", "Age", "Balance"
    ]

    missing = [col for col in required if col not in data.columns]
    if missing:
        st.error(
            "The uploaded file is missing required columns: "
            + ", ".join(missing)
        )
        st.stop()

    for col in [
        "Exited", "IsActiveMember", "NumOfProducts",
        "Age", "Balance", "CreditScore",
        "Tenure", "EstimatedSalary", "HasCrCard"
    ]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data["Customer Status"] = np.where(
        data["Exited"] == 1, "Churned", "Retained"
    )

    data["Engagement Status"] = np.where(
        data["IsActiveMember"] == 1, "Active", "Inactive"
    )

    data["Product Group"] = (
        data["NumOfProducts"].fillna(-1).astype(int).astype(str)
        .replace({"-1": "Unknown"})
    )

    data["Age Group"] = pd.cut(
        data["Age"],
        bins=[0, 25, 35, 45, 55, 120],
        labels=["18–25", "26–35", "36–45", "46–55", "56+"],
        include_lowest=True
    )

    return data

def churn_rate(frame):
    if len(frame) == 0:
        return 0.0
    return frame["Exited"].eq(1).mean() * 100

def money(value):
    return f"€{value:,.0f}"

# -------------------- SIDEBAR ---------------------

st.sidebar.markdown("""
<div style="padding: 18px 0 24px 0;">
    <div style="font-size:12px; letter-spacing:2px; color:#77D7CE;">
        BANKING ANALYTICS
    </div>
    <div style="font-size:25px; font-weight:800; color:white;">
        BankPulse
    </div>
    <div style="font-size:12px; color:#C5D2E5;">
        Intelligence Platform
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Workspace")

page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Overview",
        "Engagement Intelligence",
        "Product Utilization",
        "Customer Risk",
        "Segmentation Studio",
        "Data Explorer & Methodology"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Data source")

uploaded_file = st.sidebar.file_uploader(
    "Upload your bank customer CSV",
    type=["csv"]
)

st.sidebar.caption(
    "Upload the European_Bank 2.csv dataset "
    "to explore your own customer records."
)

# -------------------- LOAD DATA -------------------

if uploaded_file is not None:
    raw_df = load_data(uploaded_file)
else:
    raw_df = None

if raw_df is None:
    st.markdown("""
    <div class="hero">
        <div class="eyebrow">Customer Intelligence • Retention Analytics</div>
        <h1>BankPulse Intelligence</h1>
        <p>
            Understand customer engagement, product adoption and churn.
            Turn banking data into clear, actionable retention strategies.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "To begin, upload your dataset using the sidebar on the left. "
        "Use your European_Bank 2.csv file."
    )

    st.markdown("### What this platform helps you understand")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### 01 · Retention")
        st.write(
            "Measure customer churn, retention performance, "
            "and differences between customer groups."
        )

    with c2:
        st.markdown("#### 02 · Engagement")
        st.write(
            "Compare active and inactive customers to understand "
            "how engagement relates to retention."
        )

    with c3:
        st.markdown("#### 03 · Product adoption")
        st.write(
            "Explore product holding patterns and identify "
            "customer groups that may need attention."
        )

    st.markdown("---")
    st.caption(
        "BankPulse Intelligence | Customer Engagement & Product "
        "Utilization Analytics for Retention Strategy"
    )
    st.stop()

df = prepare_data(raw_df)

# -------------------- GLOBAL FILTERS --------------

st.sidebar.markdown("### Global filters")

geographies = sorted(df["Geography"].dropna().unique().tolist())
selected_geographies = st.sidebar.multiselect(
    "Geography",
    geographies,
    default=geographies
)

engagement_options = ["Active", "Inactive"]
selected_engagement = st.sidebar.multiselect(
    "Engagement status",
    engagement_options,
    default=engagement_options
)

product_options = sorted(df["Product Group"].unique().tolist())
selected_products = st.sidebar.multiselect(
    "Number of products",
    product_options,
    default=product_options
)

filtered_df = df[
    df["Geography"].isin(selected_geographies)
    & df["Engagement Status"].isin(selected_engagement)
    & df["Product Group"].isin(selected_products)
].copy()

st.sidebar.markdown("---")
st.sidebar.caption(f"Records in current view: {len(filtered_df):,}")

# -------------------- SHARED HELPERS --------------

def page_header(kicker, title, description):
    st.markdown(f"""
    <div class="hero">
        <div class="eyebrow">{kicker}</div>
        <h1>{title}</h1>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)

def metric_card(label, value, note):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)

def section(title, subtitle=""):
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True
    )
    if subtitle:
        st.markdown(
            f'<div class="section-subtitle">{subtitle}</div>',
            unsafe_allow_html=True
        )

def chart_layout(fig, height=370):
    fig.update_layout(
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter", color="#334155", size=12),
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E9EEF5")
    return fig

# -------------------- EXECUTIVE OVERVIEW ----------

if page == "Executive Overview":

    page_header(
        "Executive command center",
        "Customer Retention Overview",
        "A portfolio-level view of customer retention, churn exposure, "
        "and the engagement patterns that can inform retention planning."
    )

    total = len(filtered_df)
    exited = int(filtered_df["Exited"].eq(1).sum())
    retained = int(filtered_df["Exited"].eq(0).sum())
    rate = churn_rate(filtered_df)
    retention = 100 - rate
    active_rate = (
        filtered_df["IsActiveMember"].eq(1).mean() * 100
        if total else 0
    )

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        metric_card("CUSTOMER RECORDS", f"{total:,}", "Current filtered portfolio")
    with k2:
        metric_card("RETENTION RATE", f"{retention:.2f}%", f"{retained:,} retained records")
    with k3:
        metric_card("CHURN RATE", f"{rate:.2f}%", f"{exited:,} exited records")
    with k4:
        metric_card("ACTIVE MEMBERS", f"{active_rate:.1f}%", "Share classified as active")

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1, 1])

    with left:
        section(
            "Portfolio outcome",
            "How the currently selected customer records are distributed."
        )

        outcome = pd.DataFrame({
            "Customer outcome": ["Retained", "Churned"],
            "Customers": [retained, exited]
        })

        fig = px.pie(
            outcome,
            names="Customer outcome",
            values="Customers",
            hole=0.68,
            color="Customer outcome",
            color_discrete_map={
                "Retained": "#1B8A8F",
                "Churned": "#E16B6B"
            }
        )

        fig.update_traces(
            textposition="outside",
            textinfo="label+percent"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(chart_layout(fig, 360), use_container_width=True)

    with right:
        section(
            "Churn by engagement",
            "Compare historical churn rates among active and inactive members."
        )

        engagement = (
            filtered_df.groupby("Engagement Status", observed=False)
            .agg(
                Customers=("Exited", "size"),
                Churn_Rate=("Exited", "mean")
            )
            .reset_index()
        )
        engagement["Churn Rate (%)"] = engagement["Churn_Rate"] * 100

        fig = px.bar(
            engagement,
            x="Engagement Status",
            y="Churn Rate (%)",
            color="Engagement Status",
            text=engagement["Churn Rate (%)"].map(lambda x: f"{x:.1f}%"),
            color_discrete_map={
                "Active": "#1B8A8F",
                "Inactive": "#E6A23C"
            },
            hover_data={"Customers": True, "Churn_Rate": False}
        )
        fig.update_traces(textposition="outside")
        fig.update_yaxes(title="Customers who exited (%)")
        st.plotly_chart(chart_layout(fig, 360), use_container_width=True)

    st.markdown("""
    <div class="insight">
        <strong>How to interpret this page</strong><br>
        Start with the overall churn rate, then compare it across engagement
        groups. If inactive customers have a higher observed churn rate,
        inactivity may be a useful signal for prioritizing retention outreach.
        This pattern is an association in the dataset, not proof that inactivity
        directly causes churn.
    </div>
    """, unsafe_allow_html=True)

    section(
        "Portfolio by geography",
        "Compare customer volume and churn rate across the selected geographies."
    )

    geo = (
        filtered_df.groupby("Geography", dropna=False)
        .agg(
            Customers=("Exited", "size"),
            Churn_Rate=("Exited", "mean")
        )
        .reset_index()
    )
    geo["Churn Rate (%)"] = geo["Churn_Rate"] * 100

    fig = px.bar(
        geo,
        x="Geography",
        y="Churn Rate (%)",
        text=geo["Churn Rate (%)"].map(lambda x: f"{x:.1f}%"),
        hover_data=["Customers"],
        color="Churn Rate (%)",
        color_continuous_scale=["#BDE8E4", "#197D88", "#173B68"]
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(chart_layout(fig, 360), use_container_width=True)

# -------------------- ENGAGEMENT ------------------

elif page == "Engagement Intelligence":

    page_header(
        "Behavioral analytics",
        "Engagement Intelligence",
        "Understand how active membership relates to retention and "
        "identify engagement groups that may deserve additional attention."
    )

    engagement = (
        filtered_df.groupby("Engagement Status")
        .agg(
            Customers=("Exited", "size"),
            Churn_Rate=("Exited", "mean"),
            Average_Balance=("Balance", "mean"),
            Average_Age=("Age", "mean")
        )
        .reset_index()
    )

    engagement["Churn Rate (%)"] = engagement["Churn_Rate"] * 100

    a, b = st.columns(2)

    with a:
        section("Customer distribution", "Number of customers by engagement.")
        fig = px.bar(
            engagement,
            x="Engagement Status",
            y="Customers",
            color="Engagement Status",
            text="Customers",
            color_discrete_map={
                "Active": "#1B8A8F",
                "Inactive": "#E6A23C"
            }
        )
        st.plotly_chart(chart_layout(fig), use_container_width=True)

    with b:
        section("Observed churn rate", "Percentage of each engagement group that exited.")
        fig = px.bar(
            engagement,
            x="Engagement Status",
            y="Churn Rate (%)",
            color="Engagement Status",
            text=engagement["Churn Rate (%)"].map(lambda x: f"{x:.1f}%"),
            color_discrete_map={
                "Active": "#1B8A8F",
                "Inactive": "#E6A23C"
            }
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(chart_layout(fig), use_container_width=True)

    section("Engagement group metrics", "Supporting figures for the current filtered records.")
    display = engagement[
        ["Engagement Status", "Customers", "Churn Rate (%)",
         "Average_Balance", "Average_Age"]
    ].rename(columns={
        "Average_Balance": "Average balance",
        "Average_Age": "Average age"
    })

    st.dataframe(display.round(2), use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="insight">
        <strong>Recommended business interpretation</strong><br>
        Compare the size of each group as well as its churn rate. A small group
        with a high churn rate and a large group with a moderate churn rate may
        require different interventions. Consider testing relevant engagement
        campaigns and measuring subsequent outcomes before claiming impact.
    </div>
    """, unsafe_allow_html=True)

# -------------------- PRODUCT UTILIZATION ---------

elif page == "Product Utilization":

    page_header(
        "Relationship depth",
        "Product Utilization Analytics",
        "Explore the relationship between the number of products held and "
        "customer retention. Product count is a proxy for relationship depth, "
        "not a complete measure of product usage."
    )

    products = (
        filtered_df.groupby("Product Group")
        .agg(
            Customers=("Exited", "size"),
            Churn_Rate=("Exited", "mean")
        )
        .reset_index()
    )

    products["Churn Rate (%)"] = products["Churn_Rate"] * 100

    left, right = st.columns(2)

    with left:
        section("Product holding distribution", "Number of customer records in each product-count group.")
        fig = px.bar(
            products,
            x="Product Group",
            y="Customers",
            text="Customers",
            color="Customers",
            color_continuous_scale=["#CDE9F5", "#2D7EAE", "#173B68"]
        )
        fig.update_xaxes(title="Number of products")
        st.plotly_chart(chart_layout(fig), use_container_width=True)

    with right:
        section("Churn by product count", "Observed exit rate within each product-count group.")
        fig = px.bar(
            products,
            x="Product Group",
            y="Churn Rate (%)",
            text=products["Churn Rate (%)"].map(lambda x: f"{x:.1f}%"),
            color="Churn Rate (%)",
            color_continuous_scale=["#BDE8E4", "#E6A23C", "#D96B6B"]
        )
        fig.update_traces(textposition="outside")
        fig.update_xaxes(title="Number of products")
        st.plotly_chart(chart_layout(fig), use_container_width=True)

    section("Product group summary")
    st.dataframe(
        products[["Product Group", "Customers", "Churn Rate (%)"]].round(2),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("""
    <div class="insight">
        <strong>Business question</strong><br>
        Do customers with different numbers of products show different churn
        patterns? Use these results to investigate possible opportunities for
        suitable product bundles. Do not assume that adding products will
        automatically reduce churn; suitability, customer needs and service
        quality also matter.
    </div>
    """, unsafe_allow_html=True)

# -------------------- CUSTOMER RISK ---------------

elif page == "Customer Risk":

    page_header(
        "Retention attention",
        "Customer Risk Explorer",
        """
Create a transparent list of customer records that may merit review.
        This is an exploratory prioritization tool, not a validated prediction
        of who will leave.
"""
    )

    risk_df = filtered_df.copy()

    risk_df["Attention Reason"] = np.select(
        [
            (risk_df["IsActiveMember"] == 0)
            & (risk_df["Balance"] > risk_df["Balance"].median()),

            (risk_df["IsActiveMember"] == 0),

            (risk_df["NumOfProducts"] == 1)
        ],
        [
            "Inactive with above-median balance",
            "Inactive membership",
            "Single-product relationship"
        ],
        default="General review"
    )

    risk_df["Attention Level"] = np.select(
        [
            (risk_df["IsActiveMember"] == 0)
            & (risk_df["Balance"] > risk_df["Balance"].median()),

            (risk_df["IsActiveMember"] == 0)
            | (risk_df["NumOfProducts"] == 1)
        ],
        [
            "Priority review",
            "Review"
        ],
        default="Routine"
    )

    r1, r2, r3 = st.columns(3)

    with r1:
        metric_card(
            "PRIORITY REVIEW",
            f"{(risk_df['Attention Level'] == 'Priority review').sum():,}",
            "Inactive and above-median balance"
        )

    with r2:
        metric_card(
            "REVIEW",
            f"{(risk_df['Attention Level'] == 'Review').sum():,}",
            "Inactive or single-product group"
        )

    with r3:
        metric_card(
            "ROUTINE",
            f"{(risk_df['Attention Level'] == 'Routine').sum():,}",
            "No listed attention indicator"
        )

    st.markdown("""
    <div class="warning">
        <strong>Important limitation</strong><br>
        These categories are transparent business rules created for exploration.
        They are not machine-learning predictions, calibrated probabilities,
        or proof that a customer will churn. Validate any operational use with
        appropriate business review and future outcome data.
    </div>
    """, unsafe_allow_html=True)

    section(
        "Customer review table",
        "Filter the records below and download the current result for further analysis."
    )

    level = st.selectbox(
        "Attention category",
        ["All", "Priority review", "Review", "Routine"]
    )

    shown = risk_df.copy()
    if level != "All":
        shown = shown[shown["Attention Level"] == level]

    possible_columns = [
        "CustomerId", "Surname", "Geography", "Age", "Balance",
        "NumOfProducts", "IsActiveMember", "Exited",
        "Attention Level", "Attention Reason"
    ]
    visible_columns = [c for c in possible_columns if c in shown.columns]

    st.dataframe(
        shown[visible_columns],
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download customer review CSV",
        data=shown[visible_columns].to_csv(index=False).encode("utf-8"),
        file_name="bankpulse_customer_review.csv",
        mime="text/csv"
    )

# -------------------- SEGMENTATION ----------------

elif page == "Segmentation Studio":

    page_header(
        "Interactive exploration",
        "Customer Segmentation Studio",
        "Build customer segments using available customer attributes, "
        "compare churn outcomes, and export the records for further analysis."
    )

    segment_options = [
        c for c in [
            "Geography", "Gender", "Age Group", "Product Group",
            "Engagement Status", "HasCrCard"
        ] if c in filtered_df.columns
    ]

    dimension = st.selectbox(
        "Choose a segmentation dimension",
        segment_options
    )

    segment = (
        filtered_df.groupby(dimension, dropna=False)
        .agg(
            Customers=("Exited", "size"),
            Churn_Rate=("Exited", "mean"),
            Average_Balance=("Balance", "mean")
        )
        .reset_index()
    )

    segment["Churn Rate (%)"] = segment["Churn_Rate"] * 100

    left, right = st.columns(2)

    with left:
        section("Segment size", "Compare the number of records in each segment.")
        fig = px.bar(
            segment,
            x=dimension,
            y="Customers",
            text="Customers",
            color="Customers",
            color_continuous_scale=["#CDE9F5", "#2D7EAE", "#173B68"]
        )
        st.plotly_chart(chart_layout(fig), use_container_width=True)

    with right:
        section("Segment churn", "Compare the observed churn percentage.")
        fig = px.bar(
            segment,
            x=dimension,
            y="Churn Rate (%)",
            text=segment["Churn Rate (%)"].map(lambda x: f"{x:.1f}%"),
            color="Churn Rate (%)",
            color_continuous_scale=["#BDE8E4", "#E6A23C", "#D96B6B"]
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(chart_layout(fig), use_container_width=True)

    st.dataframe(
        segment[[dimension, "Customers", "Churn Rate (%)", "Average_Balance"]]
        .rename(columns={"Average_Balance": "Average balance"})
        .round(2),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Export current filtered customers",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="bankpulse_filtered_customers.csv",
        mime="text/csv"
    )

# -------------------- DATA EXPLORER ---------------

elif page == "Data Explorer & Methodology":

    page_header(
        "Data transparency",
        "Data Explorer & Methodology",
        "Review the dataset structure, check data quality, inspect customer "
        "records, and understand how this platform defines its metrics."
    )

    rows, columns = df.shape
    missing_total = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        metric_card("ROWS", f"{rows:,}", "Customer records")
    with d2:
        metric_card("COLUMNS", f"{columns}", "Available fields")
    with d3:
        metric_card("MISSING CELLS", f"{missing_total:,}", "Across all columns")
    with d4:
        metric_card("DUPLICATE ROWS", f"{duplicate_rows:,}", "Exact duplicate records")

    section("Dataset preview", "First records from the uploaded dataset.")
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)

    section("Column data types")
    dtypes = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(dtype) for dtype in df.dtypes],
        "Missing values": [int(df[col].isna().sum()) for col in df.columns]
    })
    st.dataframe(dtypes, use_container_width=True, hide_index=True)

    section("Metric definitions")

    definitions = pd.DataFrame([
        {
            "Metric": "Churn rate",
            "Definition": "Records with Exited = 1 divided by all eligible records."
        },
        {
            "Metric": "Retention rate",
            "Definition": "Records with Exited = 0 divided by all eligible records."
        },
        {
            "Metric": "Active membership",
            "Definition": "Records with IsActiveMember = 1."
        },
        {
            "Metric": "Product count",
            "Definition": "Number of products recorded for a customer."
        },
        {
            "Metric": "Attention category",
            "Definition": "Exploratory rule-based category, not a validated churn prediction."
        }
    ])

    st.dataframe(definitions, use_container_width=True, hide_index=True)

    st.download_button(
        "Download filtered dataset",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="bankpulse_filtered_dataset.csv",
        mime="text/csv"
    )

# -------------------- FOOTER ----------------------

st.markdown("---")
st.caption(
    "BANKPULSE INTELLIGENCE  |  Customer Engagement & Product Utilization "
    "Analytics for Retention Strategy  |  Exploratory analytics, not "
    "a validated churn prediction system."
)
