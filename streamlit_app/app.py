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
html,body,[class*="css"]{font-family:Inter,sans-serif!important}
.stApp{background:#07120c;color:#e9f5eb;background-image:radial-gradient(circle at 15% 10%,rgba(41,145,72,.16),transparent 30%),radial-gradient(circle at 90% 20%,rgba(36,100,55,.13),transparent 28%);background-attachment:fixed}
.block-container{max-width:1450px;padding:1.4rem 2rem 3rem}
header,footer,#MainMenu{visibility:hidden}
[data-testid="stSidebar"]{background:#091a10;border-right:1px solid #1d3a26}
[data-testid="stSidebar"] *{color:#e8f4ea!important}
[data-testid="stHeader"]{background:transparent}
.hero{position:relative;overflow:hidden;padding:36px 40px;border:1px solid #234a2d;border-radius:28px;background:linear-gradient(135deg,#10291a,#0b1c12);box-shadow:0 25px 70px rgba(0,0,0,.45);animation:rise .7s ease-out}
.hero:before{content:"";position:absolute;width:320px;height:320px;right:-100px;top:-170px;border-radius:50%;background:rgba(74,190,91,.13);filter:blur(5px);animation:pulse 5s infinite}
.hero h1{margin:0;color:#f1fff3;font-size:clamp(2rem,4vw,4rem);font-weight:800;letter-spacing:-2px}.hero p{margin:10px 0 0;color:#9fbaa6;font-size:1.03rem}.pill{display:inline-block;padding:7px 13px;border-radius:999px;background:#12351e;border:1px solid #2b7040;color:#9bea9f;font-size:.78rem;font-weight:700;margin-bottom:13px}.leaf{position:absolute;font-size:25px;opacity:.13;animation:float 8s ease-in-out infinite}.leaf.one{left:5%;bottom:12%}.leaf.two{right:18%;top:15%;animation-delay:2s}.leaf.three{right:5%;bottom:8%;animation-delay:4s}
.section-title{color:#e8f7ea;font-size:1.35rem;font-weight:800;margin:22px 0 11px}.metric{background:linear-gradient(145deg,#10261a,#0c1d13);border:1px solid #21432a;border-radius:18px;padding:18px;min-height:105px;box-shadow:0 10px 28px rgba(0,0,0,.25);transition:.25s}.metric:hover{transform:translateY(-4px);border-color:#367448}.metric .label{color:#7f9d87;font-size:.76rem;font-weight:700;text-transform:uppercase;letter-spacing:.6px}.metric .value{color:#eaffec;font-size:1.8rem;font-weight:800;margin-top:6px}
.panel{background:linear-gradient(145deg,#0e2115,#0a1910);border:1px solid #1f4028;border-radius:22px;padding:22px;box-shadow:0 15px 45px rgba(0,0,0,.28);margin-bottom:12px}.panel h3,.panel h2{color:#eaf8ec!important}.panel p,.panel label{color:#a7beaD!important}
.risk-high{background:linear-gradient(135deg,#451418,#821f28);color:#fff;border:1px solid #b83a45;border-radius:22px;padding:25px;box-shadow:0 12px 45px rgba(170,35,45,.25);animation:glowred 2.5s infinite}.risk-medium{background:linear-gradient(135deg,#44330b,#74570d);color:#fff;border:1px solid #a17b20;border-radius:22px;padding:25px}.risk-low{background:linear-gradient(135deg,#0c3920,#176637);color:#fff;border:1px solid #287d47;border-radius:22px;padding:25px}.risk-title{font-size:1.65rem;font-weight:800}.risk-sub{margin-top:6px;opacity:.9}
.stButton>button{border-radius:12px;border:1px solid #347447;background:#12351e;color:#dff7e3;font-weight:700}.stButton>button:hover{background:#1a4928;border-color:#55a86a}.stSelectbox label{color:#9fbaa6!important}.stSelectbox>div>div{background:#10261a!important;color:#e8f5ea!important;border-color:#285333!important}.stTextInput>div>div{background:#10261a!important}.stDataFrame{border:1px solid #24472d;border-radius:14px;overflow:hidden}
[data-testid="stTabs"]{background:transparent}[data-testid="stTabs"] button{color:#829e89;font-weight:700}[data-testid="stTabs"] button[aria-selected="true"]{color:#9bea9f}.stAlert{background:#10251a!important;border:1px solid #285333!important;color:#dff3e2!important}
div[data-testid="stMetric"]{background:#0e2115!important;border:1px solid #21432a!important;border-radius:15px;padding:12px}div[data-testid="stMetric"] label{color:#8fa996!important}div[data-testid="stMetricValue"]{color:#eaffec!important}
@keyframes rise{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}@keyframes pulse{0%,100%{transform:scale(1);opacity:.45}50%{transform:scale(1.25);opacity:.85}}@keyframes float{0%,100%{transform:translateY(0) rotate(-8deg)}50%{transform:translateY(-25px) rotate(8deg)}}@keyframes glowred{0%,100%{box-shadow:0 12px 40px rgba(158,37,43,.22)}50%{box-shadow:0 12px 55px rgba(235,58,68,.38)}}
.footer{text-align:center;color:#64806c;padding:28px 0 5px;font-size:.8rem}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return (joblib.load(APP_DIR / "crash_model_lite.pkl"), joblib.load(APP_DIR / "model_features.pkl"), joblib.load(APP_DIR / "crash_threshold.pkl"))

@st.cache_data
def load_csv(name, **kwargs):
    return pd.read_csv(APP_DIR / name, **kwargs)

def load_optional_csv(path):
    try:
        if not path.exists() or path.stat().st_size < 100:
            return pd.DataFrame()
        df = pd.read_csv(path)
        if len(df.columns) <= 2 and any("git-lfs" in str(x).lower() for x in df.astype(str).values.ravel()[:10]):
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

model, FEATURES, THRESHOLD = load_model()
all_snapshot = load_csv("all_crops_snapshot.csv", parse_dates=["latest_date"])
full_snapshot = load_csv("latest_snapshot.csv", parse_dates=["price_date"])
history = load_csv("price_history.csv", parse_dates=["price_date"])
model_comparison = load_optional_csv(ROOT_DIR / "model_comparison.csv")
threshold_comparison = load_optional_csv(ROOT_DIR / "threshold_comparison.csv")

st.markdown("""
<div class="hero"><span class="pill">● ML MARKET INTELLIGENCE</span><h1>🌾 Crop Price Crash Predictor</h1><p>Explore Indian mandi prices, detect short-term crash risk and understand the historical patterns behind the prediction.</p><span class="leaf one">🌿</span><span class="leaf two">🍃</span><span class="leaf three">🌱</span></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🌾 Market Console")
    st.caption("Select a crop, state and mandi.")
    commodity = st.selectbox("Crop", sorted(all_snapshot["commodity"].dropna().unique()))
    filtered = all_snapshot[all_snapshot["commodity"] == commodity]
    state = st.selectbox("State", sorted(filtered["state"].dropna().unique()))
    filtered = filtered[filtered["state"] == state]
    market = st.selectbox("Mandi", sorted(filtered["market_name"].dropna().unique()))
    st.divider()
    if commodity in TRAINED_CROPS: st.success("Prediction available")
    else: st.info("Trend analysis mode")
    st.caption(f"Prediction threshold: {THRESHOLD:.2f}")
    st.caption("Crash definition: 15%+ fall within 7 days")

row = filtered[filtered["market_name"] == market].sort_values("latest_date").iloc[-1]
hist = history[(history["commodity"] == commodity) & (history["state"] == state) & (history["market_name"] == market)].sort_values("price_date")
latest_price = float(row["latest_price"])
pct7 = row.get("pct_change_7d", None)
is_supported = commodity in TRAINED_CROPS

st.markdown('<div class="section-title">Market Snapshot</div>', unsafe_allow_html=True)
c1,c2,c3,c4 = st.columns(4)
for col,label,value in [(c1,"Modal Price",f"₹{latest_price:,.0f}"),(c2,"7-Day Change",f"{pct7*100:.1f}%" if pd.notna(pct7) else "N/A"),(c3,"Latest Date",row["latest_date"].strftime("%d %b %Y")),(c4,"History Records",f"{len(hist):,}")]:
    col.markdown(f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div></div>',unsafe_allow_html=True)

st.markdown('<div class="section-title">Explore</div>', unsafe_allow_html=True)
tab1,tab2,tab3,tab4 = st.tabs(["📈 Live Analysis","🗺️ Market Explorer","🤖 Model Lab","ℹ️ About"])

with tab1:
    left,right=st.columns([1.8,1])
    with left:
        st.markdown('<div class="panel">',unsafe_allow_html=True)
        st.subheader("90-Day Price Movement")
        if len(hist)>1:
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=hist["price_date"],y=hist["max_price"],line=dict(width=0),showlegend=False,hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=hist["price_date"],y=hist["min_price"],fill="tonexty",fillcolor="rgba(61,145,79,.14)",line=dict(width=0),name="Daily range",hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=hist["price_date"],y=hist["modal_price"],mode="lines",line=dict(color="#62c878",width=3),name="Modal price"))
            fig.update_layout(height=390,margin=dict(l=5,r=5,t=10,b=5),paper_bgcolor="#0e2115",plot_bgcolor="#0e2115",font=dict(color="#b7cbb9"),hovermode="x unified",legend=dict(orientation="h"),xaxis=dict(showgrid=False),yaxis=dict(gridcolor="#1e3b26",title="₹ / Quintal"))
            st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        else: st.info("Not enough history for this market.")
        st.markdown('</div>',unsafe_allow_html=True)
    with right:
        if is_supported:
            full_row=full_snapshot[(full_snapshot["commodity"]==commodity)&(full_snapshot["state"]==state)&(full_snapshot["market_name"]==market)]
            if not full_row.empty:
                full_row=full_row.sort_values("price_date").iloc[-1]
                missing=[f for f in FEATURES if f not in full_row.index]
                if missing:
                    st.error("Model feature data is incomplete for this market.")
                else:
                    X=pd.DataFrame([full_row[FEATURES]])
                    probability=float(model.predict_proba(X)[0,1])
                    crash=probability>=THRESHOLD
                    if probability>=.55: risk,cls,icon,text="HIGH","risk-high","🚨","Strong historical crash signal."
                    elif probability>=.40: risk,cls,icon,text="MEDIUM","risk-medium","⚠️","Market should be monitored closely."
                    else: risk,cls,icon,text="LOW","risk-low","✅","Lower historical crash risk."
                    st.markdown(f'<div class="{cls}"><div class="risk-title">{icon} {risk} RISK</div><div class="risk-sub">Estimated probability of a 15%+ drop: <b>{probability*100:.1f}%</b></div><div class="risk-sub">{text}</div></div>',unsafe_allow_html=True)
                    gauge=go.Figure(go.Indicator(mode="gauge+number",value=probability*100,number={"suffix":"%","font":{"color":"#eaffec"}},gauge={"axis":{"range":[0,100],"tickcolor":"#829e89"},"bar":{"color":"#ef626b" if crash else "#62c878"},"bgcolor="#10261a","bordercolor":"#2b5135","steps":[{"range":[0,40],"color":"#16311f"},{"range":[40,55],"color":"#3b3217"},{"range":[55,100],"color":"#421b20"}],"threshold":{"line":{"color":"#d7e8da","width":3},"value":THRESHOLD*100}}))
                    gauge.update_layout(height=230,margin=dict(l=10,r=10,t=15,b=0),paper_bgcolor="#0e2115",font=dict(color="#dceade"))
                    st.plotly_chart(gauge,use_container_width=True,config={"displayModeBar":False})
                    st.caption("Historical-pattern signal, not a guaranteed price forecast.")
            else: st.info("This market does not have the required model features.")
        else:
            st.markdown('<div class="panel"><h3>📊 Trend Mode</h3><p>The crash model currently supports Banana, Brinjal, Cabbage, Garlic and Green Chilli. Other commodities remain available for price exploration.</p></div>',unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="panel">',unsafe_allow_html=True)
    st.subheader("All-Market View")
    crop_view=all_snapshot[all_snapshot["commodity"]==commodity].copy().sort_values("latest_price",ascending=False).head(20)
    fig=px.bar(crop_view,x="latest_price",y="market_name",orientation="h",labels={"latest_price":"Latest modal price (₹/Quintal)","market_name":"Mandi"})
    fig.update_traces(marker_color="#4eae63")
    fig.update_layout(height=650,paper_bgcolor="#0e2115",plot_bgcolor="#0e2115",font=dict(color="#b7cbb9"),margin=dict(l=10,r=10,t=10,b=10),xaxis=dict(gridcolor="#1e3b26"),yaxis=dict(gridcolor="#0e2115"))
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    display=crop_view[["state","market_name","latest_price","latest_date"]].rename(columns={"state":"State","market_name":"Mandi","latest_price":"Modal Price","latest_date":"Date"}).copy()
    display["Modal Price"]=display["Modal Price"].map(lambda x:f"₹{x:,.0f}")
    display["Date"]=display["Date"].dt.strftime("%d %b %Y")
    st.dataframe(display,use_container_width=True,hide_index=True)
    st.markdown('</div>',unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="panel">',unsafe_allow_html=True)
    st.subheader("Model Intelligence")
    st.write("The production predictor uses the trained Random Forest model and a probability threshold of 0.55.")
    m1,m2,m3,m4=st.columns(4)
    m1.metric("Algorithm","Random Forest")
    m2.metric("Threshold",f"{THRESHOLD:.2f}")
    m3.metric("ROC-AUC","80.56%")
    m4.metric("F1","51.89%")
    if not model_comparison.empty:
        st.markdown("#### Model comparison")
        numeric_candidates=[c for c in ["accuracy","precision","recall","f1","roc_auc"] if c in model_comparison.columns]
        if "model" in model_comparison.columns and numeric_candidates:
            metric_col="f1" if "f1" in numeric_candidates else numeric_candidates[0]
            mc=model_comparison[["model",metric_col]].copy()
            mc[metric_col]=pd.to_numeric(mc[metric_col],errors="coerce")
            mc=mc.dropna()
            if not mc.empty:
                fig=px.bar(mc,x="model",y=metric_col,title=f"{metric_col.upper()} by Model")
                fig.update_traces(marker_color="#4eae63")
                fig.update_layout(height=320,paper_bgcolor="#0e2115",plot_bgcolor="#0e2115",font=dict(color="#b7cbb9"),xaxis=dict(gridcolor="#1e3b26"),yaxis=dict(gridcolor="#1e3b26"))
                st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        else:
            st.info("Model comparison data is unavailable in the deployed repository build.")
    else:
        st.info("Model comparison CSV is stored through Git LFS and is not required for prediction. The production model remains fully available.")
    if not threshold_comparison.empty:
        st.markdown("#### Threshold analysis")
        st.dataframe(threshold_comparison,use_container_width=True,hide_index=True)
    else:
        st.caption("Threshold analysis file is optional for the live predictor.")
    st.markdown('</div>',unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="panel">',unsafe_allow_html=True)
    st.subheader("About this project")
    st.write("This project uses historical Agmarknet mandi prices to identify short-term price-crash risk. A crash is defined as a 15% or larger fall in the minimum modal price observed during the following 7 calendar days.")
    x,y,z=st.columns(3)
    x.metric("Historical records",f"{len(history):,}")
    y.metric("Commodities",f"{all_snapshot['commodity'].nunique()}")
    z.metric("Markets",f"{all_snapshot['market_name'].nunique():,}")
    st.markdown("### Features")
    st.write("Price lags, percentage changes, rolling averages, volatility, moving-average relationships, month and day-of-week features.")
    st.warning("This is a student/research decision-support project. It is not a guaranteed forecast or financial/agricultural selling instruction.")
    st.markdown('</div>',unsafe_allow_html=True)

st.markdown('<div class="footer">Built with Python · Pandas · Scikit-learn · Random Forest · Plotly · Streamlit · Agmarknet data</div>',unsafe_allow_html=True)
