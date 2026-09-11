import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Crop Price Crash Predictor",
    page_icon="🌾",
    layout="centered"
)

APP_DIR = Path(__file__).parent

TRAINED_CROPS = {"Banana", "Brinjal", "Cabbage", "Garlic", "Green Chilli"}

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: linear-gradient(160deg, #0f2818 0%, #1c3d24 35%, #2f5233 65%, #4a7c40 100%);
        background-attachment: fixed;
    }

    .block-container {
        max-width: 780px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .app-header {
        text-align: center;
        padding: 1.6rem 1.2rem;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(6px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        margin-bottom: 1.6rem;
    }

    .app-header h1 {
        color: #f4f9ec;
        font-weight: 700;
        margin-bottom: 0.3rem;
        font-size: 2.1rem;
    }

    .app-header p {
        color: #d9e8cd;
        font-size: 0.95rem;
        margin: 0;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(16px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    @keyframes shimmer {
        0% { background-position: -400px 0; }
        100% { background-position: 400px 0; }
    }

    .app-header {
        animation: fadeInUp 0.6s ease-out;
    }

    /* Style every bordered container (st.container(border=True)) as a glass card */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.94) !important;
        border-radius: 16px !important;
        border: none !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25) !important;
        padding: 0.4rem 0.4rem !important;
        animation: fadeInUp 0.55s ease-out both;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }

    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 14px 36px rgba(0, 0, 0, 0.32) !important;
    }

    /* stagger the card entrance animations */
    div.element-container:nth-of-type(1) [data-testid="stVerticalBlockBorderWrapper"] { animation-delay: 0.05s; }
    div.element-container:nth-of-type(2) [data-testid="stVerticalBlockBorderWrapper"] { animation-delay: 0.15s; }
    div.element-container:nth-of-type(3) [data-testid="stVerticalBlockBorderWrapper"] { animation-delay: 0.25s; }
    div.element-container:nth-of-type(4) [data-testid="stVerticalBlockBorderWrapper"] { animation-delay: 0.35s; }

    .badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        animation: fadeIn 0.8s ease-out;
    }

    .badge-supported { background: #dcf5e0; color: #1e7d32; }
    .badge-limited { background: #fff2d9; color: #a15c00; }

    .recommend-sell {
        background: linear-gradient(135deg, #fde3e3, #ffd4d4);
        color: #a11212;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        font-weight: 600;
        font-size: 1.05rem;
        animation: fadeInUp 0.5s ease-out;
        box-shadow: 0 4px 14px rgba(161, 18, 18, 0.15);
    }

    .recommend-hold {
        background: linear-gradient(135deg, #e0f5e4, #cdf0d6);
        color: #1e7d32;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        font-weight: 600;
        font-size: 1.05rem;
        animation: fadeInUp 0.5s ease-out;
        box-shadow: 0 4px 14px rgba(30, 125, 50, 0.15);
    }

    div[data-testid="stMetric"] {
        background: rgba(0,0,0,0.03);
        border-radius: 12px;
        padding: 0.6rem 0.5rem;
        transition: background 0.2s ease;
    }

    div[data-testid="stMetric"]:hover {
        background: rgba(74, 124, 64, 0.1);
    }

    div[data-testid="stMetricValue"] {
        animation: fadeIn 0.7s ease-out;
    }

    div[data-baseweb="select"] > div {
        border-radius: 10px !important;
    }

    footer, header { visibility: hidden; }

    .app-footer {
        text-align: center;
        color: #d9e8cd;
        font-size: 0.8rem;
        margin-top: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="app-header">
        <h1>🌾 Crop Price Crash Predictor</h1>
        <p>Live price trends and crash-risk predictions across Indian mandis, powered by Agmarknet data.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# Load data + model
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load(APP_DIR / "crash_model_lite.pkl")
    features = joblib.load(APP_DIR / "model_features.pkl")
    threshold = joblib.load(APP_DIR / "crash_threshold.pkl")
    return model, features, threshold

@st.cache_data
def load_full_snapshot():
    df = pd.read_csv(APP_DIR / "latest_snapshot.csv")
    df["price_date"] = pd.to_datetime(df["price_date"])
    return df

@st.cache_data
def load_all_crops_snapshot():
    df = pd.read_csv(APP_DIR / "all_crops_snapshot.csv")
    df["latest_date"] = pd.to_datetime(df["latest_date"])
    return df

@st.cache_data
def load_price_history():
    df = pd.read_csv(APP_DIR / "price_history.csv")
    df["price_date"] = pd.to_datetime(df["price_date"])
    return df

model, FEATURES, THRESHOLD = load_model()
full_snapshot = load_full_snapshot()
all_snapshot = load_all_crops_snapshot()
history = load_price_history()

# ---------------------------------------------------------------------------
# Selection controls
# ---------------------------------------------------------------------------

with st.container(border=True):
    col1, col2 = st.columns(2)

    with col1:
        commodity = st.selectbox("Crop", sorted(all_snapshot["commodity"].unique()))

    filtered = all_snapshot[all_snapshot["commodity"] == commodity]

    with col2:
        state = st.selectbox("State", sorted(filtered["state"].unique()))

    filtered = filtered[filtered["state"] == state]

    market = st.selectbox("Market (Mandi)", sorted(filtered["market_name"].unique()))

row = filtered[filtered["market_name"] == market]

if row.empty:
    st.warning("No data available for this combination.")
    st.stop()

row = row.sort_values("latest_date").iloc[-1]
is_supported = commodity in TRAINED_CROPS

# ---------------------------------------------------------------------------
# Current price snapshot
# ---------------------------------------------------------------------------

with st.container(border=True):
    badge_class = "badge-supported" if is_supported else "badge-limited"
    badge_text = "✅ Crash prediction supported" if is_supported else "⚠️ Limited data — price trend only"
    st.markdown(f'<span class="badge {badge_class}">{badge_text}</span>', unsafe_allow_html=True)

    st.subheader("Latest known price data")

    m1, m2, m3 = st.columns(3)
    m1.metric("Modal Price (₹/Quintal)", f"{row['latest_price']:.0f}")

    if pd.notna(row["pct_change_7d"]):
        m2.metric("~7-day change", f"{row['pct_change_7d']*100:.1f}%")
    else:
        m2.metric("~7-day change", "N/A")

    m3.metric("As of", row["latest_date"].strftime("%d %b %Y"))

# ---------------------------------------------------------------------------
# Price history chart (stock-style: min-max band + modal price line)
# ---------------------------------------------------------------------------

hist = history[
    (history["commodity"] == commodity)
    & (history["state"] == state)
    & (history["market_name"] == market)
].sort_values("price_date")

with st.container(border=True):
    st.subheader("90-day price trend")

    if hist.empty or len(hist) < 2:
        st.caption("Not enough history to chart this crop/market yet.")
    else:
        fig = go.Figure()

        # min-max daily range band
        fig.add_trace(go.Scatter(
            x=pd.concat([hist["price_date"], hist["price_date"][::-1]]),
            y=pd.concat([hist["max_price"], hist["min_price"][::-1]]),
            fill="toself",
            fillcolor="rgba(74, 124, 64, 0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False
        ))

        # gradient area under the modal price line
        fig.add_trace(go.Scatter(
            x=hist["price_date"],
            y=hist["modal_price"],
            mode="lines",
            line=dict(color="#2f5233", width=3, shape="spline", smoothing=0.6),
            fill="tozeroy",
            fillcolor="rgba(74, 124, 64, 0.25)",
            name="Modal Price",
            hovertemplate="₹%{y:,.0f}<br>%{x|%d %b %Y}<extra></extra>"
        ))

        # highlight the latest point
        fig.add_trace(go.Scatter(
            x=[hist["price_date"].iloc[-1]],
            y=[hist["modal_price"].iloc[-1]],
            mode="markers",
            marker=dict(size=10, color="#e8871e", line=dict(width=2, color="white")),
            showlegend=False,
            hoverinfo="skip"
        ))

        y_min = hist["min_price"].min()
        y_max = hist["max_price"].max()
        pad = (y_max - y_min) * 0.1 if y_max > y_min else y_max * 0.1

        fig.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=False, title=None),
            yaxis=dict(
                title="₹ / Quintal",
                range=[max(0, y_min - pad), y_max + pad],
                showgrid=True,
                gridcolor="rgba(0,0,0,0.06)"
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            hovermode="x unified",
            transition=dict(duration=500, easing="cubic-in-out")
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("Shaded band shows the daily min–max price range; the line shows the modal price.")

# ---------------------------------------------------------------------------
# Prediction + sell/hold recommendation
# ---------------------------------------------------------------------------

with st.container(border=True):
    st.subheader("Prediction")

    probability = None

    if is_supported:
        full_row = full_snapshot[
            (full_snapshot["commodity"] == commodity)
            & (full_snapshot["state"] == state)
            & (full_snapshot["market_name"] == market)
        ]

        if full_row.empty:
            st.info(
                "This crop/market combination doesn't have enough recent price "
                "history (needs at least 30 days of records) to compute a "
                "reliable prediction."
            )
        else:
            full_row = full_row.sort_values("price_date").iloc[-1]
            X = pd.DataFrame([full_row[FEATURES]])
            probability = model.predict_proba(X)[0, 1]
            will_crash = probability >= THRESHOLD

            if will_crash:
                st.error(
                    f"⚠️ **High crash risk** — estimated {probability*100:.1f}% probability "
                    f"of a 15%+ price drop within 7 days."
                )
            else:
                st.success(
                    f"✅ **Low crash risk** — estimated {probability*100:.1f}% probability "
                    f"of a 15%+ price drop within 7 days."
                )

            gauge_color = "#c0392b" if will_crash else "#1e7d32"

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%", "font": {"size": 34, "color": gauge_color}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#888"},
                    "bar": {"color": gauge_color, "thickness": 0.3},
                    "bgcolor": "white",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 40], "color": "rgba(30,125,50,0.15)"},
                        {"range": [40, 70], "color": "rgba(230,170,30,0.18)"},
                        {"range": [70, 100], "color": "rgba(192,57,43,0.18)"}
                    ],
                    "threshold": {
                        "line": {"color": gauge_color, "width": 3},
                        "thickness": 0.8,
                        "value": THRESHOLD * 100
                    }
                }
            ))
            gauge.update_layout(
                height=220,
                margin=dict(l=20, r=20, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#2f3e2a")
            )
            st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})

            st.markdown("<br>", unsafe_allow_html=True)
            if will_crash:
                st.markdown(
                    '<div class="recommend-sell">🔴 High risk — consider selling the crop now, '
                    'before the price drops.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="recommend-hold">🟢 Low risk — it looks reasonably safe to hold '
                    'the crop for now.</div>',
                    unsafe_allow_html=True
                )
    else:
        st.warning(
            f"**Crash prediction isn't available for {commodity} yet.** "
            "The model was trained only on Banana, Brinjal, Cabbage, Garlic, and "
            "Green Chilli — the crops with enough historical price history in "
            "this dataset to train a reliable model. Other crops are shown here "
            "with their latest available price trend only, as a limited resource "
            "until more training data is added."
        )

with st.expander("How this works"):
    st.write(
        """
        The crash-risk model is a Random Forest trained on historical
        Agmarknet mandi price records for five crops (Banana, Brinjal,
        Cabbage, Garlic, Green Chilli), using recent price momentum —
        lags, rolling averages, and volatility — to estimate the
        probability of a sharp (15%+) price drop within the following week.

        All other crops and states from the Agmarknet dataset are still
        browsable here for their latest price trend, but don't yet have a
        trained crash model behind them.

        **Note:** predictions are based on the most recent price data
        available in the training set for this crop/market combination,
        not live real-time prices. The sell/hold message is a simple guide,
        not financial advice.
        """
    )

# ---------------------------------------------------------------------------
# AI assistant — answers questions about this app, this crop, prices, etc.
# ---------------------------------------------------------------------------

st.markdown("### 💬 Ask the assistant")

api_key = st.secrets.get("ANTHROPIC_API_KEY", None)

if not api_key:
    st.info(
        "The AI assistant isn't configured yet. To enable it, add your "
        "Anthropic API key as `ANTHROPIC_API_KEY` in this app's "
        "**Settings → Secrets** on Streamlit Cloud, then reload the app."
    )
else:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    context_note = (
        f"Currently selected on the dashboard: crop={commodity}, state={state}, "
        f"market={market}, latest modal price={row['latest_price']:.0f} INR/quintal, "
        f"as of {row['latest_date'].strftime('%Y-%m-%d')}. "
        + (
            f"Crash-risk model estimate: {probability*100:.1f}% probability of a 15%+ "
            f"price drop in the next 7 days."
            if probability is not None
            else "No crash-risk model is available for this crop."
        )
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("Ask about this crop, this app, or crop prices in general...")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = client.messages.create(
                    model="claude-sonnet-5",
                    max_tokens=600,
                    system=(
                        "You are a helpful assistant embedded in a Crop Price Crash "
                        "Predictor web app for Indian agricultural markets (mandis). "
                        "Answer questions about crop prices, market trends, farming "
                        "economics, and this app's data and predictions. Be concise "
                        "and practical. Here is the current app context: " + context_note
                    ),
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ]
                )
                answer = response.content[0].text
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown(
    '<p class="app-footer">Data source: Agmarknet, Government of India · Model: Random Forest</p>',
    unsafe_allow_html=True
)
