import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Crop Price Crash Predictor", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
TRAINED_CROPS = {"Banana", "Brinjal", "Cabbage", "Garlic", "Green Chilli"}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*{font-family:Inter,sans-serif}.stApp{background:radial-gradient(circle at 10% 10%,rgba(54,170,91,.18),transparent 28%),radial-gradient(circle at 90% 15%,rgba(245,177,66,.13),transparent 25%),linear-gradient(135deg,#07140d,#0d2417 42%,#102d1b);background-attachment:fixed}.block-container{max-width:1450px;padding:1.5rem 2rem 3rem}header,footer,#MainMenu{visibility:hidden}[data-testid="stSidebar"]{background:linear-gradient(180deg,#08170e,#102b19);border-right:1px solid rgba(255,255,255,.08)}[data-testid="stSidebar"] *{color:#eef8ee}.hero{position:relative;overflow:hidden;padding:34px 38px;border:1px solid rgba(255,255,255,.12);border-radius:28px;background:linear-gradient(135deg,rgba(255,255,255,.10),rgba(255,255,255,.035));backdrop-filter:blur(16px);box-shadow:0 25px 70px rgba(0,0,0,.28);animation:rise .8s ease-out}.hero:before{content:"";position:absolute;width:280px;height:280px;right:-80px;top:-130px;border-radius:50%;background:rgba(108,214,124,.15);filter:blur(8px);animation:pulse 5s infinite}.hero h1{margin:0;color:#f4fff1;font-size:clamp(2rem,4vw,4rem);font-weight:800;letter-spacing:-2px}.hero p{margin:9px 0 0;color:#b9d0bd;font-size:1.03rem}.pill{display:inline-block;padding:7px 13px;border-radius:999px;background:rgba(106,211,117,.13);border:1px solid rgba(106,211,117,.3);color:#a9efb1;font-size:.78rem;font-weight:700;margin-bottom:13px}.leaf{position:absolute;font-size:25px;opacity:.12;animation:float 8s ease-in-out infinite}.leaf.one{left:5%;bottom:12%}.leaf.two{right:18%;top:15%;animation-delay:2s}.leaf.three{right:5%;bottom:8%;animation-delay:4s}.card{background:rgba(255,255,255,.965);border-radius:22px;padding:22px;box-shadow:0 15px 45px rgba(0,0,0,.20);animation:rise .55s ease-out}.section-title{color:#eaffea;font-size:1.35rem;font-weight:800;margin:20px 0 10px}.metric{background:linear-gradient(145deg,#fff,#f0f7f0);border-radius:18px;padding:18px;min-height:105px;border:1px solid #dbe9dc;box-shadow:0 8px 25px rgba(0,0,0,.08);transition:.25s}.metric:hover{transform:translateY(-4px);box-shadow:0 15px 30px rgba(0,0,0,.14)}.metric .label{color:#66806b;font-size:.76rem;font-weight:700;text-transform:uppercase;letter-spacing:.6px}.metric .value{color:#173d22;font-size:1.8rem;font-weight:800;margin-top:6px}.risk-high{background:linear-gradient(135deg,#4b1114,#9e252b);color:white;border-radius:22px;padding:25px;box-shadow:0 12px 40px rgba(158,37,43,.32);animation:glowred 2.5s infinite}.risk-medium{background:linear-gradient(135deg,#533b08,#a26d0c);color:white;border-radius:22px;padding:25px}.risk-low{background:linear-gradient(135deg,#0d4522,#208a45);color:white;border-radius:22px;padding:25px}.risk-title{font-size:1.65rem;font-weight:800}.risk-sub{margin-top:5px;opacity:.88}.stButton>button{border-radius:12px;border:0;background:linear-gradient(135deg,#2d9b4a,#176b31);color:white;font-weight:700;padding:.65rem 1rem;transition:.25s}.stButton>button:hover{transform:translateY(-2px);box-shadow:0 8px 22px rgba(45,155,74,.28)}div[data-baseweb="select"]>div{border-radius:12px!important}[data-testid="stTabs"] button{color:#bcd2c0;font-weight:700}[data-testid="stTabs"] button[aria-selected="true"]{color:#9bea9f}@keyframes rise{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}@keyframes pulse{0%,100%{transform:scale(1);opacity:.5}50%{transform:scale(1.25);opacity:.9}}@keyframes float{0%,100%{transform:translateY(0) rotate(-8deg)}50%{transform:translateY(-25px) rotate(8deg)}}@keyframes glowred{0%,100%{box-shadow:0 12px 40px rgba(158,37,43,.28)}50%{box-shadow:0 12px 55px rgba(235,58,68,.48)}}.footer{text-align:center;color:#78947e;padding:28px 0 5px;font-size:.8rem}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return (joblib.load(APP_DIR / "crash_model_lite.pkl"), joblib.load(APP_DIR / "model_features.pkl"), joblib.load(APP_DIR / "crash_threshold.pkl"))

@st.cache_data
def load_csv(name, **kwargs):
    return pd.read_csv(APP_DIR / name, **kwargs)

model, FEATURES, THRESHOLD = load_model()
all_snapshot = load_csv("all_crops_snapshot.csv", parse_dates=["latest_date"])
full_snapshot = load_csv("latest_snapshot.csv", parse_dates=["price_date"])
history = load_csv("price_history.csv", parse_dates=["price_date"])

try:
    model_comparison = pd.read_csv(ROOT_DIR / "model_comparison.csv")
except Exception:
    model_comparison = pd.DataFrame()
try:
    threshold_comparison = pd.read_csv(ROOT_DIR / "threshold_comparison.csv")
except Exception:
    threshold_comparison = pd.DataFrame()

st.markdown("""
<div class="hero"><span class="pill">● ML MARKET INTELLIGENCE</span><h1>🌾 Crop Price Crash Predictor</h1><p>Explore Indian mandi prices, identify short-term crash risk, and understand the patterns behind the prediction.</p><span class="leaf one">🌿</span><span class="leaf two">🍃</span><span class="leaf three">🌱</span></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🌾 Market Console")
    st.caption("Choose a crop and mandi to explore the data.")
    commodity = st.selectbox("Crop", sorted(all_snapshot["commodity"].dropna().unique()))
    filtered = all_snapshot[all_snapshot["commodity"] == commodity]
    state = st.selectbox("State", sorted(filtered["state"].dropna().unique()))
    filtered = filtered[filtered["state"] == state]
    market = st.selectbox("Mandi", sorted(filtered["market_name"].dropna().unique()))
    st.divider()
    st.markdown("### Model status")
    if commodity in TRAINED_CROPS:
        st.success("Prediction available")
    else:
        st.warning("Trend analysis only")
    st.caption(f"Crash threshold: {THRESHOLD:.2f}")
    st.caption("Model focus: 15%+ drop within 7 days")

row = filtered[filtered["market_name"] == market].sort_values("latest_date").iloc[-1]
hist = history[(history["commodity"] == commodity) & (history["state"] == state) & (history["market_name"] == market)].sort_values("price_date")
latest_price = float(row["latest_price"])
pct7 = row.get("pct_change_7d", None)
is_supported = commodity in TRAINED_CROPS

st.markdown('<div class="section-title">Market snapshot</div>', unsafe_allow_html=True)
c1,c2,c3,c4 = st.columns(4)
for col,label,value in [(c1,"Modal Price",f"₹{latest_price:,.0f}"),(c2,"7-Day Change",f"{pct7*100:.1f}%" if pd.notna(pct7) else "N/A"),(c3,"Latest Date",row["latest_date"].strftime("%d %b %Y")),(c4,"Records in View",f"{len(hist):,}")]:
    col.markdown(f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Explore</div>', unsafe_allow_html=True)
tab1,tab2,tab3,tab4 = st.tabs(["📈 Live Analysis","🗺️ Market Explorer","🤖 Model Lab","ℹ️ About"])

with tab1:
    left,right = st.columns([1.8,1])
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("90-day price movement")
        if len(hist)>1:
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=hist["price_date"],y=hist["max_price"],line=dict(width=0),showlegend=False,hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=hist["price_date"],y=hist["min_price"],fill="tonexty",fillcolor="rgba(72,170,91,.12)",line=dict(width=0),name="Daily range",hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=hist["price_date"],y=hist["modal_price"],mode="lines",line=dict(color="#2e9146",width=3),name="Modal price",hovertemplate="₹%{y:,.0f}<br>%{x|%d %b %Y}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[hist["price_date"].iloc[-1]],y=[hist["modal_price"].iloc[-1]],mode="markers",marker=dict(size=12,color="#f3a63b",line=dict(width=3,color="white")),showlegend=False))
            fig.update_layout(height=390,margin=dict(l=5,r=5,t=10,b=5),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",hovermode="x unified",legend=dict(orientation="h"),xaxis=dict(showgrid=False),yaxis=dict(gridcolor="rgba(0,0,0,.06)",title="₹ / Quintal"))
            st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        else:
            st.info("Not enough history for this market.")
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        if is_supported:
            full_row=full_snapshot[(full_snapshot["commodity"]==commodity)&(full_snapshot["state"]==state)&(full_snapshot["market_name"]==market)]
            if not full_row.empty:
                full_row=full_row.sort_values("price_date").iloc[-1]
                X=pd.DataFrame([full_row[FEATURES]])
                probability=float(model.predict_proba(X)[0,1])
                crash=probability>=THRESHOLD
                if probability>=.55:
                    risk,cls,icon,text="HIGH","risk-high","🚨","Strong historical crash signal."
                elif probability>=.40:
                    risk,cls,icon,text="MEDIUM","risk-medium","⚠️","Market should be monitored closely."
                else:
                    risk,cls,icon,text="LOW","risk-low","✅","Lower historical crash risk."
                st.markdown(f'<div class="{cls}"><div class="risk-title">{icon} {risk} RISK</div><div class="risk-sub">Estimated probability of a 15%+ drop: <b>{probability*100:.1f}%</b></div><div class="risk-sub">{text}</div></div>',unsafe_allow_html=True)
                gauge=go.Figure(go.Indicator(mode="gauge+number",value=probability*100,number={"suffix":"%"},gauge={"axis":{"range":[0,100]},"bar":{"color":"#e34b51" if crash else "#2e9b4b"},"steps":[{"range":[0,40],"color":"#e6f5e9"},{"range":[40,55],"color":"#fff2cf"},{"range":[55,100],"color":"#fde2e2"}],"threshold":{"line":{"color":"#222","width":3},"value":THRESHOLD*100}}))
                gauge.update_layout(height=230,margin=dict(l=10,r=10,t=15,b=0),paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(gauge,use_container_width=True,config={"displayModeBar":False})
                st.caption("Historical-pattern signal, not a guaranteed price forecast.")
            else:
                st.info("This market does not have the required model features.")
        else:
            st.markdown('<div class="card"><h3>📊 Trend mode</h3><p>The current crash model was trained on Banana, Brinjal, Cabbage, Garlic and Green Chilli. Other commodities remain available for price exploration.</p></div>',unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("All-market view")
    crop_view=all_snapshot[all_snapshot["commodity"]==commodity].copy().sort_values("latest_price",ascending=False).head(20)
    fig=px.bar(crop_view,x="latest_price",y="market_name",color="latest_price",orientation="h",color_continuous_scale="Greens",labels={"latest_price":"Latest modal price (₹/Quintal)","market_name":"Mandi"})
    fig.update_layout(height=650,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",coloraxis_showscale=False,margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    st.dataframe(crop_view[["state","market_name","latest_price","latest_date"]].rename(columns={"state":"State","market_name":"Mandi","latest_price":"Modal Price","latest_date":"Date"}),use_container_width=True,hide_index=True)
    st.markdown('</div>',unsafe_allow_html=True)

with tab3:
    a,b=st.columns(2)
    with a:
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.subheader("Model comparison")
        if not model_comparison.empty:
            cols=[c for c in ["model","accuracy","precision","recall","f1","roc_auc"] if c in model_comparison.columns]
            st.dataframe(model_comparison[cols],use_container_width=True,hide_index=True)
            metric_col="f1" if "f1" in model_comparison.columns else model_comparison.columns[-1]
            fig=px.bar(model_comparison,x="model",y=metric_col,title="Model F1 comparison")
            fig.update_layout(height=300,margin=dict(l=5,r=5,t=45,b=5),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        else:
            st.info("Model comparison file not available.")
        st.markdown('</div>',unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.subheader("Threshold tuning")
        if not threshold_comparison.empty:
            st.dataframe(threshold_comparison,use_container_width=True,hide_index=True)
            if "threshold" in threshold_comparison.columns and "f1" in threshold_comparison.columns:
                fig=px.line(threshold_comparison,x="threshold",y="f1",markers=True,title="F1 score by threshold")
                fig.update_layout(height=300,margin=dict(l=5,r=5,t=45,b=5),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        else:
            st.info("Threshold comparison file not available.")
        st.markdown('</div>',unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="card">',unsafe_allow_html=True)
    st.subheader("About this project")
    st.write("This project uses historical Agmarknet mandi prices to identify short-term price-crash risk. A crash is defined as a 15% or larger fall in the minimum modal price observed during the following 7 calendar days.")
    x,y,z=st.columns(3)
    x.metric("Historical records",f"{len(history):,}")
    y.metric("Commodities",f"{all_snapshot['commodity'].nunique()}")
    z.metric("Markets",f"{all_snapshot['market_name'].nunique():,}")
    st.markdown("### What the model uses")
    st.write("Price lags, percentage changes, rolling averages, volatility, moving-average relationships, month and day-of-week features.")
    st.markdown("### Important")
    st.warning("The application is a student/research decision-support project. It does not account for weather, policy shocks, sudden supply changes, transport costs or real-time events, so its output should not be treated as a guaranteed selling instruction.")
    st.markdown('</div>',unsafe_allow_html=True)

st.markdown('<div class="footer">Built with Python · Pandas · Scikit-learn · Random Forest · Plotly · Streamlit · Agmarknet data</div>',unsafe_allow_html=True)
