import streamlit as st
import pandas as pd
import joblib
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

    .glass-card {
        background: rgba(255, 255, 255, 0.92);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
        margin-bottom: 1.2rem;
    }

    .badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }

    .badge-supported {
        background: #dcf5e0;
        color: #1e7d32;
    }

    .badge-limited {
        background: #fff2d9;
        color: #a15c00;
    }

    div[data-testid="stMetric"] {
        background: rgba(0,0,0,0.03);
        border-radius: 12px;
        padding: 0.6rem 0.5rem;
    }

    div[data-baseweb="select"] > div {
        border-radius: 10px !important;
    }

    footer, header {
        visibility: hidden;
    }

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
    # Full feature set, available only for the 5 crops the crash model was trained on
    df = pd.read_csv(APP_DIR / "latest_snapshot.csv")
    df["price_date"] = pd.to_datetime(df["price_date"])
    return df

@st.cache_data
def load_all_crops_snapshot():
    # Covers every crop and state in the raw Agmarknet dataset
    df = pd.read_csv(APP_DIR / "all_crops_snapshot.csv")
    df["latest_date"] = pd.to_datetime(df["latest_date"])
    return df

model, FEATURES, THRESHOLD = load_model()
full_snapshot = load_full_snapshot()
all_snapshot = load_all_crops_snapshot()

# ---------------------------------------------------------------------------
# Selection controls
# ---------------------------------------------------------------------------

st.markdown('<div class="glass-card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    commodity = st.selectbox("Crop", sorted(all_snapshot["commodity"].unique()))

filtered = all_snapshot[all_snapshot["commodity"] == commodity]

with col2:
    state = st.selectbox("State", sorted(filtered["state"].unique()))

filtered = filtered[filtered["state"] == state]

market = st.selectbox("Market (Mandi)", sorted(filtered["market_name"].unique()))

row = filtered[filtered["market_name"] == market]

st.markdown('</div>', unsafe_allow_html=True)

if row.empty:
    st.warning("No data available for this combination.")
    st.stop()

row = row.sort_values("latest_date").iloc[-1]
is_supported = commodity in TRAINED_CROPS

# ---------------------------------------------------------------------------
# Current price snapshot (available for every crop/state)
# ---------------------------------------------------------------------------

st.markdown('<div class="glass-card">', unsafe_allow_html=True)

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

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Prediction (only for the 5 crops the model was trained on)
# ---------------------------------------------------------------------------

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Prediction")

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

        st.progress(min(max(probability, 0.0), 1.0))
else:
    st.warning(
        f"**Crash prediction isn't available for {commodity} yet.** "
        "The model was trained only on Banana, Brinjal, Cabbage, Garlic, and "
        "Green Chilli — the crops with enough historical price history in "
        "this dataset to train a reliable model. Other crops are shown here "
        "with their latest available price trend only, as a limited resource "
        "until more training data is added."
    )

st.markdown('</div>', unsafe_allow_html=True)

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
        not live real-time prices.
        """
    )

st.markdown(
    '<p class="app-footer">Data source: Agmarknet, Government of India · Model: Random Forest</p>',
    unsafe_allow_html=True
)
