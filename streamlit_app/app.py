import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Crop Price Crash Predictor",
    page_icon="🌾",
    layout="centered"
)

# ---------- Load model + data (cached so it only loads once) ----------

@st.cache_resource
def load_model():
    model = joblib.load("crash_model_lite.pkl")
    features = joblib.load("model_features.pkl")
    threshold = joblib.load("crash_threshold.pkl")
    return model, features, threshold

@st.cache_data
def load_snapshot():
    df = pd.read_csv("latest_snapshot.csv")
    df["price_date"] = pd.to_datetime(df["price_date"])
    return df

model, FEATURES, THRESHOLD = load_model()
snapshot = load_snapshot()

# ---------- Header ----------

st.title("🌾 Crop Price Crash Predictor")
st.caption(
    "Predicts the probability that a crop's price will drop 15% or more "
    "within the next 7 days, based on recent Agmarknet mandi price trends."
)

st.divider()

# ---------- Selection controls ----------

col1, col2 = st.columns(2)

with col1:
    commodity = st.selectbox(
        "Crop",
        sorted(snapshot["commodity"].unique())
    )

filtered = snapshot[snapshot["commodity"] == commodity]

with col2:
    state = st.selectbox(
        "State",
        sorted(filtered["state"].unique())
    )

filtered = filtered[filtered["state"] == state]

market = st.selectbox(
    "Market (Mandi)",
    sorted(filtered["market_name"].unique())
)

row = filtered[filtered["market_name"] == market]

if row.empty:
    st.warning("No data available for this combination.")
    st.stop()

row = row.sort_values("price_date").iloc[-1]

# ---------- Show current snapshot ----------

st.divider()
st.subheader("Latest known price data")

m1, m2, m3 = st.columns(3)
m1.metric("Modal Price (₹/Quintal)", f"{row['modal_price']:.0f}")
m2.metric("7-day change", f"{row['price_change_7d']*100:.1f}%")
m3.metric("As of", row["price_date"].strftime("%d %b %Y"))

# ---------- Predict ----------

X = pd.DataFrame([row[FEATURES]])
probability = model.predict_proba(X)[0, 1]
will_crash = probability >= THRESHOLD

st.divider()
st.subheader("Prediction")

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

with st.expander("How this works"):
    st.write(
        """
        This app uses a Random Forest model trained on historical Agmarknet
        mandi price records for Banana, Brinjal, Cabbage, Garlic, and Green
        Chilli. It looks at recent price momentum (lags, rolling averages,
        volatility) for the selected crop and market, and estimates the
        likelihood of a sharp (15%+) price crash in the following week.

        **Note:** predictions are based on the most recent price data
        available in the training set for this crop/market combination,
        not live real-time prices.
        """
    )

st.caption("Data source: Agmarknet, Government of India · Model: Random Forest")
