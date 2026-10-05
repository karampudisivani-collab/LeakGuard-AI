
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest

st.set_page_config(
    page_title="LeakGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Styling ----------
st.markdown("""
<style>
    .main {background: #0b0f14;}
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    .hero {
        padding: 1.4rem 1.6rem;
        border: 1px solid #2a3340;
        border-radius: 18px;
        background: linear-gradient(135deg,#111823,#0c1118);
        margin-bottom: 1rem;
    }
    .hero h1 {margin:0; font-size: 2.35rem;}
    .hero p {color:#9aa7b5; margin:.45rem 0 0;}
    .risk-card {
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid #2a3340;
        background:#111720;
        margin-bottom:.7rem;
    }
    .risk-high {border-left: 5px solid #ff9f43;}
    .risk-critical {border-left: 5px solid #ff4d4d;}
    .small {color:#9aa7b5;font-size:.88rem;}
    .metric-label {color:#9aa7b5;font-size:.9rem;}
    .metric-value {font-size:1.75rem;font-weight:700;}
    .pill {
        display:inline-block;padding:.22rem .55rem;border-radius:99px;
        font-size:.78rem;font-weight:700;background:#26303c;
    }
</style>
""", unsafe_allow_html=True)

def money(x):
    return f"₹{float(x):,.0f}"

@st.cache_data
def load_demo():
    tx = pd.read_csv("sample_transactions.csv")
    inv = pd.read_csv("sample_inventory.csv")
    return tx, inv

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🛡️ LeakGuard AI")
    st.caption("Business leakage intelligence")
    st.divider()
    st.markdown("### Detection controls")
    price_threshold = st.slider("Price anomaly threshold (%)", 10, 60, 20)
    qty_threshold = st.slider("Quantity anomaly threshold (%)", 25, 150, 50)
    ml_weight = st.slider("ML anomaly weight", 5, 30, 15)
    st.divider()
    st.markdown("**Prototype note**")
    st.caption("Risk signals support human review. They are not proof of fraud.")

# ---------- Upload ----------
st.markdown("""
<div class="hero">
<h1>🛡️ LeakGuard AI</h1>
<p>Detect • Explain • Prioritize potential business leakage</p>
</div>
""", unsafe_allow_html=True)

u1, u2 = st.columns(2)
with u1:
    uploaded = st.file_uploader("📤 Upload transaction CSV", type=["csv"])
with u2:
    inv_uploaded = st.file_uploader("📦 Upload inventory CSV", type=["csv"])

# Start completely empty. Demo data is loaded only when the user explicitly clicks
# the Demo Mode button.
demo_inv = None
use_demo = st.sidebar.button("🎬 Load Demo Data", use_container_width=True)
st.sidebar.caption("The app starts empty. Use Demo Mode only when you want to test the sample business data.")

if uploaded:
    tx = pd.read_csv(uploaded)
elif use_demo:
    tx, demo_inv = load_demo()
else:
    st.info("👋 Welcome to LeakGuard AI. Upload transaction data to begin, or choose **🎬 Load Demo Data** in the sidebar to explore the prototype.")
    st.stop()

if inv_uploaded:
    inv = pd.read_csv(inv_uploaded)
elif use_demo and demo_inv is not None:
    inv = demo_inv
else:
    inv = pd.DataFrame()

required = {"transaction_id","vendor_id","vendor_name","product_id","quantity","unit_price","total_amount"}
missing = required - set(tx.columns)
if missing:
    st.error("Missing transaction columns: " + ", ".join(sorted(missing)))
    st.stop()

# ---------- Detection engine ----------
tx = tx.copy()
tx["date"] = pd.to_datetime(tx.get("date"), errors="coerce")

stats = tx.groupby("product_id").agg(
    avg_price=("unit_price","mean"),
    median_price=("unit_price","median"),
    avg_qty=("quantity","mean")
).reset_index()

tx = tx.merge(stats, on="product_id", how="left")
tx["price_variance_pct"] = ((tx["unit_price"]-tx["median_price"]) / tx["median_price"] * 100).replace([np.inf,-np.inf],0).fillna(0)
tx["qty_variance_pct"] = ((tx["quantity"]-tx["avg_qty"]) / tx["avg_qty"] * 100).replace([np.inf,-np.inf],0).fillna(0)

# Duplicate patterns
dup_cols = ["vendor_id","product_id","quantity","unit_price","total_amount"]
tx["duplicate_flag"] = tx.duplicated(subset=dup_cols, keep=False)

# ML
features = tx[["quantity","unit_price","total_amount","price_variance_pct","qty_variance_pct"]].replace([np.inf,-np.inf],0).fillna(0)
if len(tx) >= 30:
    model = IsolationForest(contamination=0.08, random_state=42)
    tx["ml_flag"] = model.fit_predict(features) == -1
else:
    tx["ml_flag"] = False

# More demo-friendly, evidence-based score
def score_row(r):
    score = 0
    reasons = []
    exposure = 0.0

    if r["duplicate_flag"]:
        score += 40
        reasons.append("Possible duplicate transaction")
        exposure += float(r["total_amount"])

    if r["price_variance_pct"] > price_threshold:
        score += 35
        reasons.append(f"Price is {r['price_variance_pct']:.0f}% above product baseline")
        exposure += max(0, float(r["unit_price"]-r["median_price"]) * float(r["quantity"]))

    if r["qty_variance_pct"] > qty_threshold:
        score += 15
        reasons.append(f"Quantity is {r['qty_variance_pct']:.0f}% above normal")
        exposure += float(r["total_amount"]) * 0.10

    if r["ml_flag"]:
        score += ml_weight
        reasons.append("ML model identified an unusual transaction pattern")

    # Severity boost for combinations
    if r["duplicate_flag"] and r["price_variance_pct"] > price_threshold:
        score += 12
        reasons.append("Duplicate + abnormal pricing combination increases review priority")

    if r["price_variance_pct"] > 40:
        score += 8
        reasons.append("Severe price deviation")

    return min(int(score),100), reasons, max(exposure, float(r["total_amount"])*0.05)

results = tx.apply(score_row, axis=1, result_type="expand")
tx["risk_score"] = results[0]
tx["reasons"] = results[1]
tx["potential_exposure"] = results[2]
tx["risk_level"] = pd.cut(
    tx["risk_score"], [-1,30,60,80,101],
    labels=["Low","Medium","High","Critical"]
)

# ---------- Inventory ----------
inv_issues = pd.DataFrame()
if not inv.empty:
    required_inv = {"product_id","opening_stock","purchased_quantity","received_quantity","sold_quantity","closing_stock"}
    if required_inv.issubset(inv.columns):
        inv = inv.copy()
        wastage = inv["wastage"] if "wastage" in inv.columns else 0
        inv["receipt_gap"] = inv["purchased_quantity"] - inv["received_quantity"]
        inv["expected_closing"] = inv["opening_stock"] + inv["received_quantity"] - inv["sold_quantity"] - wastage
        inv["stock_gap"] = inv["expected_closing"] - inv["closing_stock"]
        inv_issues = inv[(inv["receipt_gap"] > 0) | (inv["stock_gap"] > 0)].copy()

# Estimate inventory exposure using product median price
price_map = tx.groupby("product_id")["median_price"].mean()
if not inv_issues.empty:
    inv_issues["estimated_leakage"] = inv_issues["receipt_gap"].clip(lower=0) * inv_issues["product_id"].map(price_map).fillna(0)
else:
    inv_issues["estimated_leakage"] = pd.Series(dtype=float)

high_cases = tx[tx["risk_score"] >= 61]
medium_cases = tx[(tx["risk_score"] >= 31) & (tx["risk_score"] < 61)]

transaction_exposure = float(high_cases["potential_exposure"].sum())
inventory_exposure = float(inv_issues["estimated_leakage"].sum()) if not inv_issues.empty else 0
potential_leakage = transaction_exposure + inventory_exposure

# ---------- KPI row ----------
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("💰 Potential Leakage", money(potential_leakage))
k2.metric("📊 Transactions", f"{len(tx):,}")
k3.metric("🔴 High/Critical", f"{len(high_cases):,}")
k4.metric("🟡 Medium", f"{len(medium_cases):,}")
k5.metric("📦 Inventory Issues", f"{len(inv_issues):,}")

st.divider()

# ---------- Charts ----------
left, right = st.columns(2)

with left:
    st.subheader("Leakage risk distribution")
    counts = tx["risk_level"].value_counts().reindex(["Low","Medium","High","Critical"]).fillna(0).reset_index()
    counts.columns = ["Risk Level","Cases"]
    fig = px.bar(counts, x="Risk Level", y="Cases", text="Cases")
    fig.update_layout(height=300, margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Potential exposure by risk level")
    exposure = tx.groupby("risk_level", observed=False)["potential_exposure"].sum().reindex(["Low","Medium","High","Critical"]).fillna(0).reset_index()
    exposure.columns = ["Risk Level","Exposure"]
    fig2 = px.bar(exposure, x="Risk Level", y="Exposure", text_auto=".2s")
    fig2.update_layout(height=300, margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ---------- Tabs ----------
tabs = st.tabs(["🔴 Risk Cases","🏢 Vendor Intelligence","📦 Inventory","🤖 AI Investigation","📈 Insights"])

# Risk cases
with tabs[0]:
    st.subheader("Prioritized leakage cases")
    st.caption("Cases are ranked using transaction rules + ML anomaly signals.")

    high = tx[tx["risk_score"] >= 31].sort_values(["risk_score","potential_exposure"], ascending=False)

    if high.empty:
        st.success("No review-worthy cases detected with the current thresholds.")
    else:
        view = high[[
            "transaction_id","vendor_name","product_id","quantity",
            "unit_price","total_amount","risk_score","risk_level","potential_exposure"
        ]].head(100).copy()
        view["unit_price"] = view["unit_price"].round(2)
        view["total_amount"] = view["total_amount"].round(2)
        view["potential_exposure"] = view["potential_exposure"].round(2)
        st.dataframe(view, use_container_width=True, hide_index=True)

        chosen = st.selectbox("🔎 Investigate a case", high["transaction_id"].tolist())
        r = tx[tx["transaction_id"] == chosen].iloc[0]

        cls = "risk-critical" if r["risk_score"] >= 81 else "risk-high"
        reasons_html = "".join([f"<li>{x}</li>" for x in r["reasons"]])
        st.markdown(f"""
        <div class="risk-card {cls}">
        <h3>{'🔴' if r['risk_score']>=81 else '🟠'} {r['transaction_id']} — {r['risk_level']} risk ({int(r['risk_score'])}/100)</h3>
        <p><b>Vendor:</b> {r['vendor_name']} &nbsp; | &nbsp; <b>Product:</b> {r['product_id']}</p>
        <p><b>Transaction value:</b> {money(r['total_amount'])} &nbsp; | &nbsp; <b>Potential exposure:</b> {money(r['potential_exposure'])}</p>
        <p><b>Evidence:</b></p><ul>{reasons_html if reasons_html else '<li>ML anomaly signal only</li>'}</ul>
        </div>
        """, unsafe_allow_html=True)

        st.warning("Recommended action: review the purchase order, invoice, goods-received note, approval trail and inventory records. This is a review signal, not proof of fraud.")

# Vendors
with tabs[1]:
    st.subheader("Vendor Intelligence")
    vr = tx.groupby(["vendor_id","vendor_name"]).agg(
        transactions=("transaction_id","count"),
        flagged_cases=("risk_score",lambda s:int((s>=31).sum())),
        high_critical=("risk_score",lambda s:int((s>=61).sum())),
        avg_price_variance=("price_variance_pct","mean"),
        transaction_value=("total_amount","sum"),
        potential_exposure=("potential_exposure","sum")
    ).reset_index()

    vr["vendor_risk"] = np.clip(
        vr["high_critical"]*20 +
        vr["flagged_cases"]*5 +
        np.maximum(vr["avg_price_variance"],0)*1.0, 0, 100
    ).round().astype(int)

    vr = vr.sort_values(["vendor_risk","potential_exposure"], ascending=False)
    st.dataframe(vr, use_container_width=True, hide_index=True)

    figv = px.bar(vr.head(8), x="vendor_name", y="potential_exposure",
                  title="Top vendors by potential exposure")
    figv.update_layout(height=330, xaxis_title="", yaxis_title="₹")
    st.plotly_chart(figv, use_container_width=True)

# Inventory
with tabs[2]:
    st.subheader("Purchase → Receipt → Stock Reconciliation")
    if inv.empty or not required_inv.issubset(inv.columns):
        st.info("Upload an inventory CSV containing opening, purchased, received, sold and closing stock.")
    else:
        cols = ["product_id"] + (["product_name"] if "product_name" in inv.columns else []) + [
            "purchased_quantity","received_quantity","receipt_gap",
            "sold_quantity","closing_stock","stock_gap"
        ]
        st.dataframe(inv[cols], use_container_width=True, hide_index=True)

        if not inv_issues.empty:
            st.error(f"⚠️ {len(inv_issues)} inventory discrepancies require review.")
            st.metric("Estimated inventory exposure", money(inv_issues["estimated_leakage"].sum()))
            st.dataframe(
                inv_issues[["product_id"] + (["product_name"] if "product_name" in inv_issues.columns else []) +
                           ["receipt_gap","stock_gap","estimated_leakage"]],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("No inventory discrepancies detected.")

# AI assistant
with tabs[3]:
    st.subheader("🤖 AI Investigation Assistant")
    st.caption("For the MVP, this assistant generates an evidence-based investigation summary from detected signals.")

    question = st.text_input(
        "Ask LeakGuard",
        placeholder="Why was the highest-risk transaction flagged?"
    )

    if question:
        r = tx.sort_values(["risk_score","potential_exposure"], ascending=False).iloc[0]
        reasons = r["reasons"]
        st.markdown(f"### Investigation: `{r['transaction_id']}`")
        st.write(
            f"**Risk score:** {int(r['risk_score'])}/100  \n"
            f"**Risk level:** {r['risk_level']}  \n"
            f"**Transaction value:** {money(r['total_amount'])}  \n"
            f"**Potential exposure:** {money(r['potential_exposure'])}"
        )
        st.markdown("**Why it was flagged:**")
        for reason in reasons:
            st.write("• " + reason)
        st.markdown("**Recommended human review:**")
        st.write("1. Verify purchase order and approved supplier quote.")
        st.write("2. Match invoice against the goods-received note.")
        st.write("3. Check whether the transaction was already paid.")
        st.write("4. Compare the vendor's recent prices and discrepancy history.")
        st.caption("LeakGuard AI identifies potential leakage; an authorized human should make the final decision.")

# Insights
with tabs[4]:
    st.subheader("Management Insights")

    col1,col2,col3 = st.columns(3)
    with col1:
        top_product = tx.groupby("product_id")["potential_exposure"].sum().sort_values(ascending=False).head(1)
        st.metric("Highest-exposure product", top_product.index[0] if len(top_product) else "—")
    with col2:
        top_vendor = vr.iloc[0]["vendor_name"] if len(vr) else "—"
        st.metric("Highest-risk vendor", top_vendor)
    with col3:
        avg_price_gap = max(0, float(tx["price_variance_pct"].mean()))
        st.metric("Avg. price variance", f"{avg_price_gap:.1f}%")

    st.markdown("### What management should review first")
    priorities = []
    if len(high_cases): priorities.append(f"🔴 Review **{len(high_cases)} high/critical transactions**.")
    if len(inv_issues): priorities.append(f"📦 Reconcile **{len(inv_issues)} inventory discrepancies**.")
    if len(vr) and vr.iloc[0]["vendor_risk"] >= 60: priorities.append(f"🏢 Investigate **{vr.iloc[0]['vendor_name']}** due to elevated vendor risk.")
    if not priorities: priorities.append("🟢 No major review priorities under current thresholds.")
    for p in priorities:
        st.write(p)

# ---------- Report ----------
st.divider()
report = tx[tx["risk_score"]>=31][[
    "transaction_id","vendor_name","product_id","total_amount","risk_score","risk_level","potential_exposure"
]].sort_values("risk_score",ascending=False)

csv_report = report.to_csv(index=False).encode("utf-8")
st.download_button(
    "📄 Download Leakage Review Report (CSV)",
    data=csv_report,
    file_name="leakguard_review_report.csv",
    mime="text/csv"
)

st.caption("LeakGuard AI • College MVP • Synthetic demonstration data • Risk indicators require human verification.")
