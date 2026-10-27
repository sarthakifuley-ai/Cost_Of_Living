import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

st.set_page_config(page_title="Cost of Living Explorer", page_icon="🌍", layout="wide")

# ---- palette ----------------------------------------------------------------
TXT, MUTED = "#e8ecf8", "#8a94b8"
GRID = "rgba(255,255,255,0.07)"
CYAN, VIOLET, PINK, AMBER, GREEN = "#22d3ee", "#a78bfa", "#f472b6", "#fbbf24", "#34d399"
FONT = "Inter, Segoe UI, Helvetica, Arial, sans-serif"
COMPARE_COLORS = [CYAN, PINK, AMBER, GREEN]

METRICS = {
    "Cost of living": "Cost_of_Living_Index",
    "Rent": "Rent_Index",
    "Groceries": "Groceries_Index",
    "Restaurants": "Restaurant_Price_Index",
    "Purchasing power": "Local_Purchasing_Power_Index",
}
RADAR_LABELS, RADAR_COLS = list(METRICS), list(METRICS.values())
FEATURES = ["Rent_Index", "Groceries_Index", "Restaurant_Price_Index"]
CSV_PATH = "cost_of_living.csv"
NUMBEO_URL = "https://www.numbeo.com/cost-of-living/rankings_by_country.jsp?title=2026"


# ---- data & model -----------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    # fallback: scrape Numbeo exactly like the notebook does
    df = pd.read_html(NUMBEO_URL)[1].drop(columns=["Rank"])
    df.columns = ["Country", "Cost_of_Living_Index", "Rent_Index",
                  "Cost_of_Living_Plus_Rent_Index", "Groceries_Index",
                  "Restaurant_Price_Index", "Local_Purchasing_Power_Index"]
    df.to_csv(CSV_PATH, index=False)
    return df


@st.cache_resource
def train_best_model(df: pd.DataFrame):
    """Same as Part H of the notebook: pick the best regressor by R²."""
    X, y = df[FEATURES], df["Cost_of_Living_Index"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr, X_te = scaler.fit_transform(X_tr), scaler.transform(X_te)
    models = {
        "Multiple Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42),
    }
    scores = {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        scores[name] = r2_score(y_te, m.predict(X_te))
    best = max(scores, key=scores.get)
    return models[best], scaler, best


df = load_data()
cost = df["Cost_of_Living_Index"]
model, scaler, best_name = train_best_model(df)
q1, q2 = cost.quantile([1 / 3, 2 / 3])
pct = df[RADAR_COLS].rank(pct=True) * 100  # percentile so all 5 axes are comparable


def style(fig, h=520):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family=FONT, color=TXT), height=h, margin=dict(l=10, r=30, t=60, b=20))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


# ---- charts -----------------------------------------------------------------
def rankings_fig(metric_label, n, order):
    col = METRICS[metric_label]
    d = (df.nlargest(n, col) if order == "Highest" else df.nsmallest(n, col)).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d[col], y=d["Country"], orientation="h", text=d[col].round(1),
        textposition="outside", cliponaxis=False,
        marker=dict(color=d[col], colorscale="Plasma", line_width=0),
        hovertemplate="<b>%{y}</b><br>" + metric_label + ": %{x:.1f}<extra></extra>"))
    fig.update_layout(title=f"<b>{order} {n}</b> · {metric_label} index", bargap=0.25)
    return style(fig, max(380, 24 * n + 130))


def compare_fig(countries):
    fig = go.Figure()
    for c, colr in zip(countries, COMPARE_COLORS):
        i = df.index[df["Country"] == c][0]
        raw, p = df.loc[i, RADAR_COLS].tolist(), pct.loc[i].tolist()
        fig.add_trace(go.Scatterpolar(
            r=p + p[:1], theta=RADAR_LABELS + RADAR_LABELS[:1], name=c, fill="toself", opacity=0.6,
            line=dict(color=colr, width=2), customdata=raw + raw[:1],
            hovertemplate="<b>" + c + "</b><br>%{theta}: %{customdata:.1f}<br>Percentile: %{r:.0f}<extra></extra>"))
    fig.update_layout(
        title="<b>Country profile</b> · percentile rank (100 = higher than every other country)",
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(range=[0, 100], gridcolor=GRID, linecolor=GRID, tickfont=dict(size=10)),
                   angularaxis=dict(gridcolor=GRID, linecolor=GRID)),
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"))
    return style(fig, 560)


def gauge_fig(p):
    top = max(140, p * 1.1)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p, number=dict(font=dict(size=54, color=TXT), valueformat=".1f"),
        title=dict(text="Predicted cost-of-living index", font=dict(size=16)),
        gauge=dict(axis=dict(range=[0, top], tickcolor=MUTED), bar=dict(color=PINK, thickness=0.28),
                   bgcolor="rgba(0,0,0,0)", borderwidth=0,
                   steps=[dict(range=[0, q1], color="rgba(34,211,238,0.25)"),
                          dict(range=[q1, q2], color="rgba(167,139,250,0.25)"),
                          dict(range=[q2, top], color="rgba(244,114,182,0.25)")],
                   threshold=dict(line=dict(color=AMBER, width=3), value=float(cost.mean())))))
    return style(fig, 380)


# ---- page -------------------------------------------------------------------
st.markdown("""
<style>
.block-container {max-width: 1150px; padding-top: 2rem;}
.hero {padding: 28px 30px; border-radius: 20px; margin-bottom: 14px;
       background: linear-gradient(135deg, #1b1f4b 0%, #3b1d5e 55%, #5a1f55 100%);
       border: 1px solid rgba(255,255,255,0.1);}
.hero h1 {margin: 0 0 4px 0; font-size: 2.1rem; color: #fff; padding: 0;}
.hero p {margin: 0 0 18px 0; color: #b7bfe0;}
.chips {display: flex; gap: 12px; flex-wrap: wrap;}
.chip {flex: 1; min-width: 150px; padding: 12px 16px; border-radius: 14px; background: rgba(255,255,255,0.07);}
.chip b {display: block; font-size: 1.6rem; color: #fff;}
.chip span {font-size: 0.8rem; color: #aab3d6; text-transform: uppercase; letter-spacing: .06em;}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero">
  <h1>🌍 Cost of Living Explorer</h1>
  <p>Numbeo 2026 · rank countries, compare them side by side, and predict living costs.</p>
  <div class="chips">
    <div class="chip"><b>{len(df)}</b><span>Countries</span></div>
    <div class="chip"><b>{cost.mean():.1f}</b><span>Average index</span></div>
    <div class="chip"><b>{df.loc[cost.idxmax(), 'Country']}</b><span>Most expensive</span></div>
    <div class="chip"><b>{df.loc[cost.idxmin(), 'Country']}</b><span>Cheapest</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

tab_rank, tab_cmp, tab_pred = st.tabs(["🏆 Rankings", "⚔️ Compare countries", "🔮 Cost predictor"])

# tab 1 · rankings
with tab_rank:
    c1, c2, c3 = st.columns([2, 1, 2])
    metric = c1.selectbox("Metric", list(METRICS), index=0)
    order = c2.radio("Show", ["Highest", "Lowest"], horizontal=True)
    n = c3.slider("Countries", 5, 30, 10)
    st.plotly_chart(rankings_fig(metric, n, order), use_container_width=True)

# tab 2 · compare countries
with tab_cmp:
    countries_all = sorted(df["Country"])
    default_cmp = [c for c in ["Switzerland", "Japan", "India"] if c in countries_all] or countries_all[:3]
    picks = st.multiselect("Pick up to 4 countries", countries_all, default=default_cmp, max_selections=4)
    if picks:
        st.plotly_chart(compare_fig(picks), use_container_width=True)
        table = df[df["Country"].isin(picks)][["Country"] + RADAR_COLS].round(1)
        table.columns = ["Country"] + RADAR_LABELS
        st.caption("Raw index values")
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.info("Select at least one country to see the comparison.")

# tab 3 · predictor
with tab_pred:
    st.write("Move the sliders — the best multiple-regression model from Part H predicts "
             "the overall cost-of-living index.")
    left, right = st.columns([1, 1])
    with left:
        def slider(label, col):
            s = df[col]
            return st.slider(label, float(int(s.min())), float(int(s.max()) + 1),
                             float(s.median()), step=0.5)
        rent_v = slider("Rent index", "Rent_Index")
        groc_v = slider("Groceries index", "Groceries_Index")
        rest_v = slider("Restaurant price index", "Restaurant_Price_Index")

        row = scaler.transform(pd.DataFrame([[rent_v, groc_v, rest_v]], columns=FEATURES))
        p = float(model.predict(row)[0])
        tier = "💚 Low-cost" if p < q1 else ("💜 Mid-cost" if p < q2 else "🔥 High-cost")
        gap = p - cost.mean()
        st.markdown(
            f"### {tier} country\n"
            f"That is **{abs(gap):.1f} points {'above' if gap >= 0 else 'below'}** the world average "
            f"of {cost.mean():.1f} (amber line). Model used: **{best_name}**.")
    with right:
        st.plotly_chart(gauge_fig(p), use_container_width=True)
