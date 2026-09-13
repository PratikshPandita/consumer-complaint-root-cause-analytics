"""
03_eda_and_trends.py
---------------------------------------------------------------------
Local pandas equivalent of scripts/01_bigquery_queries.sql -- runs the
same trend / root-cause / SLA logic against whatever CSV is in
data/sample_complaints.csv (synthetic by default, real once you swap
in scripts/05_bigquery_connector.py's output). Produces PNG charts
in outputs/.
---------------------------------------------------------------------
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = "/home/claude/complaints_project/data/sample_complaints.csv"
OUT_DIR = "/home/claude/complaints_project/outputs"

df = pd.read_csv(DATA_PATH, parse_dates=["date_received"])
df["month"] = df["date_received"].dt.to_period("M").dt.to_timestamp()

print(f"Loaded {len(df):,} complaints spanning {df['date_received'].min().date()} "
      f"to {df['date_received'].max().date()}")

# ---------------------------------------------------------------------
# 1. Volume trend by month and product
# ---------------------------------------------------------------------
monthly_product = (
    df.groupby(["month", "product"]).size().reset_index(name="count")
)
top_products = df["product"].value_counts().nlargest(5).index
pivot = monthly_product[monthly_product["product"].isin(top_products)].pivot(
    index="month", columns="product", values="count"
).fillna(0)

fig, ax = plt.subplots(figsize=(10, 5))
pivot.plot(ax=ax)
ax.set_title("Complaint Volume Trend by Product (Monthly)")
ax.set_ylabel("Complaints")
ax.set_xlabel("Month")
ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_volume_trend_by_product.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------
# 2. Root-cause breakdown: top issues per product
# ---------------------------------------------------------------------
top_issues = (
    df.groupby(["product", "issue"]).size().reset_index(name="count")
    .sort_values(["product", "count"], ascending=[True, False])
    .groupby("product").head(3)
)
print("\n=== Top root-cause issues per product ===")
print(top_issues.to_string(index=False))
top_issues.to_csv(f"{OUT_DIR}/top_issues_per_product.csv", index=False)

# ---------------------------------------------------------------------
# 3. Geographic hot spots
# ---------------------------------------------------------------------
state_top_issue = (
    df.groupby(["state", "issue"]).size().reset_index(name="count")
    .sort_values(["state", "count"], ascending=[True, False])
    .groupby("state").head(1)
    .sort_values("count", ascending=False)
)
print("\n=== Leading root-cause issue by state (top 10 states by volume) ===")
print(state_top_issue.head(10).to_string(index=False))

# ---------------------------------------------------------------------
# 4. Regulatory SLA reporting: timely response + dispute rate by company
# ---------------------------------------------------------------------
company_sla = df.groupby("company").agg(
    total_complaints=("complaint_id", "count"),
    pct_timely=("timely_response", lambda s: round((s == "Yes").mean() * 100, 1)),
    pct_disputed=("consumer_disputed", lambda s: round((s == "Yes").mean() * 100, 1)),
).reset_index().sort_values("pct_timely")
print("\n=== Regulatory SLA snapshot by company ===")
print(company_sla.to_string(index=False))
company_sla.to_csv(f"{OUT_DIR}/company_sla_snapshot.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(company_sla["company"], company_sla["pct_timely"], color="#2E86AB")
ax.set_xlabel("% Timely Response")
ax.set_title("Regulatory SLA: Timely Response Rate by Company")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_sla_timely_response_by_company.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------
# 5. Company response effectiveness
# ---------------------------------------------------------------------
response_mix = df["company_response_to_consumer"].value_counts(normalize=True).mul(100).round(1)
print("\n=== Company response mix (%) ===")
print(response_mix.to_string())

fig, ax = plt.subplots(figsize=(7, 5))
response_mix.plot(kind="bar", ax=ax, color="#A23B72")
ax.set_ylabel("% of Complaints")
ax.set_title("Company Response Type Distribution")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_response_type_distribution.png", dpi=120)
plt.close()

print(f"\nCharts written to {OUT_DIR}/")
