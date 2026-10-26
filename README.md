# RetainIQ — Customer Retention Intelligence Platform

Streamlit dashboard for **Customer Engagement & Product Utilization Analytics for Retention Strategy**.

## Included dashboard modules
1. Executive Overview
2. Engagement Intelligence
3. Product Utilization
4. High-Value Customer Detector
5. Retention Strength Assessment
6. Segmentation Studio
7. Data Explorer & Methodology

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Default dataset
Place your project CSV in the same folder as `app.py`, using one of these filenames:
- `European_Bank 2.csv`
- `European_Bank 3.csv`
- `European_Bank.csv`

The app also supports uploading a CSV through the sidebar. The CSV should include:
`CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Exited`

`CustomerId`, `Surname`, and `Year` are optional.

## Deploy on Streamlit Community Cloud
1. Push `app.py`, `requirements.txt`, `README.md`, and your permitted-to-share CSV to GitHub.
2. In Streamlit Community Cloud, create or update the app from the repository.
3. Set the main file path to `app.py`.
4. Ensure the CSV filename in GitHub matches one of the supported names above.
5. Open the deployed app and test all navigation pages, filters, and downloads.

## KPI definitions
- **Engagement Retention Ratio:** active-member retention rate divided by inactive-member retention rate.
- **Product Depth Index:** product count scaled to the maximum product count observed in this dataset (0–100).
- **High-Balance Disengagement Rate:** inactive customers above the chosen balance threshold divided by all customers above that threshold.
- **Credit Card Stickiness Score:** card-holder retention rate minus non-holder retention rate, in percentage points.
- **Relationship Strength Index:** exploratory heuristic based on activity, product count, card ownership, and positive balance.

## Important limitations
The risk flag and Relationship Strength Index are transparent exploratory rules, not validated predictive models. Observed associations do not prove causation. Check dataset licensing and privacy before placing the CSV in a public repository.
