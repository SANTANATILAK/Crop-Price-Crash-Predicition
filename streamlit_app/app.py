import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Crop Price Crash Predictor", page_icon="🌾", layout="wide")
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
TRAINED = {"Banana", "Brinjal", "Cabbage", "Garlic", "Green Chilli"}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:Inter,sans-serif!important}
.stApp{background:#06100a;color:#e9f7eb;background-image:radial-gradient(circle at 10% 5%,rgba(47,170,75,.16),transparent 28%),radial-gradient(circle at 90% 15%,rgba(30,100,50,.12),transparent 30%);background-attachment:fixed}
.block-container{max-width:1450px;padding:1.4rem 2rem 3rem}header,footer,#MainMenu{visibility:hidden}
[data-testid="stSidebar"]{background:#08170d;border-right:1px solid #1d3b25}[data-testid="stSidebar"] *{color:#e5f3e7!important}
.hero{position:relative;overflow:hidden;padding:38px 42px;border:1px solid #285333;border-radius:28px;background:linear-gradient(135deg,#10281a,#091a10);box-shadow:0 25px 70px #0008;animation:rise .7s ease-out}.hero:after{content:"";position:absolute;width:300px;height:300px;right:-90px;top:-170px;border-radius:50%;background:#54c66a22;filter:blur(5px);animation:pulse 5s infinite}.hero h1{margin:0;color:#f1fff3;font-size:clamp(2rem,4vw,4rem);font-weight:800;letter-spacing:-2px}.hero p{color:#9fbaa6;margin:10px 0 0;font-size:1.05rem}.pill{display:inline-block;padding:7px 13px;border-radius:99px;background:#12351e;border:1px solid #347447;color:#9bea9f;font-size:.75rem;font-weight:800;margin-bottom:13px}.leaf{position:absolute;font-size:25px;opacity:.14;animation:float 8s ease-in-out infinite}.one{left:5%;bottom:12%}.two{right:18%;top:15%;animation-delay:2s}.three{right:5%;bottom:8%;animation-delay:4s}
.title{color:#e9f8eb;font-size:1.35rem;font-weight:800;margin:22px 0 11px}.metric,.panel{background:linear-gradient(145deg,#10261a,#0a1910);border:1px solid #21452b;border-radius:20px;box-shadow:0 14px 40px #0006}.metric{padding:18px;min-height:105px;transition:.25s}.metric:hover{transform:translateY(-4px);border-color:#438e54}.metric .label{color:#7f9d87;font-size:.75rem;font-weight:700;text-transform:uppercase}.metric .value{color:#eaffec;font-size:1.75rem;font-weight:800;margin-top:7px}.panel{padding:22px;margin-bottom:12px}.panel h2,.panel h3{color:#eaf8ec!important}.panel p{color:#a7beac!important}
.risk-high,.risk-medium,.risk-low{color:white;border-radius:20px;padding:23px}.risk-high{background:linear-gradient(135deg,#451418,#821f28);border:1px solid #b83a45;animation:glow 2.5s infinite}.risk-medium{background:linear-gradient(135deg,#44330b,#74570d);border:1px solid #a17b20}.risk-low{background:linear-gradient(135deg,#0c3920,#176637);border:1px solid #287d47}.risk-title{font-size:1.55rem;font-weight:800}.risk-sub{margin-top:6px;opacity:.9}
.stButton>button{background:#12351e!important;color:#e8f8ea!important;border:1px solid #347447!important;border-radius:12px!important;font-weight:700}.stSelectbox label{color:#9fbaa6!important}.stSelectbox>div>div{background:#10261a!important;color:#e8f5ea!important;border-color:#285333!important}.stDataFrame{border:1px solid #24472d;border-radius:14px;overflow:hidden}[data-testid="stTabs"] button{color:#829e89;font-weight:700}[data-testid="stTabs"] button[aria-selected="true"]{color:#9bea9f}.stAlert{background:#10251a!important;border:1px solid #285333!important;color:#dff3e2!important}div[data-testid="stMetric"]{background:#0e2115!important;border:1px solid #21432a!important;border-radius:14px}div[data-testid="stMetricValue"]{color:#eaffec!important}
@keyframes rise{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}@keyframes pulse{0%,100%{transform:scale(1);opacity:.4}50%{transform:scale(1.25);opacity:.85}}@keyframes float{0%,100%{transform:translateY(0) rotate(-8deg)}50%{transform:translateY(-25px) rotate(8deg)}}@keyframes glow{0%,100%{box-shadow:0 12px 40px #b52f3822}50%{box-shadow:0 12px 55px #ed596044}}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return (joblib.load(APP_DIR / "crash_model_lite.pkl"), joblib.load(APP_DIR / "model_features.pkl"), joblib.load(APP_DIR / "crash_threshold.pkl"))

@st.cache_data
def load_data(name, dates=None):
    return pd.read_csv(APP_DIR / name, parse_dates=dates or [])

model, FEATURES, THRESHOLD = load_model()
all_snapshot = load_data("all_crops_snapshot.csv", ["latest_date"])
latest = load_data("latest_snapshot.csv", ["price_date"])
history = load_data("price_history.csv", ["price_date"])

st.markdown('<div class="hero"><span class="pill">● ML MARKET INTELLIGENCE</span><h1>🌾 Crop Price Crash Predictor</h1><p>Indian mandi price intelligence with historical crash-risk analysis.</p><span class="leaf one">🌿</span><span class="leaf two">🍃</span><span class="leaf three">🌱</span></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🌾 Market Console")
    crop = st.selectbox("Crop", sorted(all_snapshot.commodity.dropna().unique()))
    d = all_snapshot[all_snapshot.commodity == crop]
    state = st.selectbox("State", sorted(d.state.dropna().unique()))
    d = d[d.state == state]
    market = st.selectbox("Mandi", sorted(d.market_name.dropna().unique()))
    st.divider()
    st.success("Prediction available") if crop in TRAINED else st.info("Trend analysis mode")
    st.caption(f"Threshold: {THRESHOLD:.2f} · 15%+ drop / 7 days")

row = d[d.market_name == market].sort_values("latest_date").iloc[-1]
h = history[(history.commodity == crop) & (history.state == state) & (history.market_name == market)].sort_values("price_date")
price = float(row.latest_price)
change = row.get("pct_change_7d", float("nan"))

st.markdown('<div class="title">Market Snapshot</div>', unsafe_allow_html=True)
cols = st.columns(4)
items = [("Modal Price",f"₹{price:,.0f}"),("7-Day Change",f"{change*100:.1f}%" if pd.notna(change) else "N/A"),("Latest Date",row.latest_date.strftime("%d %b %Y")),("History Records",f"{len(h):,}")]
for c,(label,value) in zip(cols,items):
    c.markdown(f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div></div>',unsafe_allow_html=True)

st.markdown('<div class="title">Explore</div>', unsafe_allow_html=True)
t1,t2,t3,t4 = st.tabs(["📈 Live Analysis","🗺️ Market Explorer","🤖 Model Lab","ℹ️ About"])

with t1:
    a,b = st.columns([1.8,1])
    with a:
        st.markdown('<div class="panel">',unsafe_allow_html=True)
        st.subheader("90-Day Price Movement")
        if len(h)>1:
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=h.price_date,y=h.max_price,line=dict(width=0),showlegend=False,hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=h.price_date,y=h.min_price,fill="tonexty",fillcolor="rgba(70,160,90,.13)",line=dict(width=0),name="Daily range"))
            fig.add_trace(go.Scatter(x=h.price_date,y=h.modal_price,mode="lines",line=dict(color="#62c878",width=3),name="Modal price"))
            fig.update_layout(height=390,paper_bgcolor="#0e2115",plot_bgcolor="#0e2115",font=dict(color="#b7cbb9"),margin=dict(l=5,r=5,t=10,b=5),xaxis=dict(showgrid=False),yaxis=dict(gridcolor="#1e3b26",title="₹ / Quintal"),legend=dict(orientation="h"))
            st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        else: st.info("Not enough history for this market.")
        st.markdown('</div>',unsafe_allow_html=True)
    with b:
        if crop in TRAINED:
            r=latest[(latest.commodity==crop)&(latest.state==state)&(latest.market_name==market)]
            if not r.empty:
                r=r.sort_values("price_date").iloc[-1]
                missing=[f for f in FEATURES if f not in r.index]
                if missing:
                    st.error("Model feature data is incomplete for this market.")
                else:
                    probability=float(model.predict_proba(pd.DataFrame([r[FEATURES]]))[0,1])
                    crash=probability>=THRESHOLD
                    if probability>=.55: risk,cl="HIGH","risk-high"; icon="🚨"; msg="Strong historical crash signal."
                    elif probability>=.40: risk,cl="MEDIUM","risk-medium"; icon="⚠️"; msg="Market should be monitored closely."
                    else: risk,cl="LOW","risk-low"; icon="✅"; msg="Lower historical crash risk."
                    st.markdown(f'<div class="{cl}"><div class="risk-title">{icon} {risk} RISK</div><div class="risk-sub">Estimated probability: <b>{probability*100:.1f}%</b></div><div class="risk-sub">{msg}</div></div>',unsafe_allow_html=True)
                    gauge = go.Figure(go.Indicator(mode="gauge+number",value=probability*100,number={"suffix":"%","font":{"color":"#eaffec"}},gauge={"axis":{"range":[0,100],"tickcolor":"#829e89"},"bar":{"color":"#ef626b" if crash else "#62c878"},"bgcolor":"#10261a","bordercolor":"#2b5135","steps":[{"range":[0,40],"color":"#16311f"},{"range":[40,55],"color":"#3b3217"},{"range":[55,100],"color":"#421b20"}],"threshold":{"line":{"color":"#d7e8da","width":3},"value":THRESHOLD*100}}))
                    gauge.update_layout(height=230,margin=dict(l=10,r=10,t=15,b=0),paper_bgcolor="#0e2115",font=dict(color="#dceade"))
                    st.plotly_chart(gauge,use_container_width=True,config={"displayModeBar":False})
                    st.caption("Historical-pattern signal, not a guaranteed forecast.")
        else:
            st.markdown('<div class="panel"><h3>📊 Trend Mode</h3><p>The trained crash model currently supports Banana, Brinjal, Cabbage, Garlic and Green Chilli.</p></div>',unsafe_allow_html=True)

with t2:
    st.markdown('<div class="panel">',unsafe_allow_html=True)
    st.subheader("Top Mandi Prices")
    view=d.sort_values("latest_price",ascending=False).head(20).copy()
    fig=px.bar(view,x="latest_price",y="market_name",orientation="h",labels={"latest_price":"Modal Price (₹/Quintal)","market_name":"Mandi"})
    fig.update_traces(marker_color="#4eae63")
    fig.update_layout(height=620,paper_bgcolor="#0e2115",plot_bgcolor="#0e2115",font=dict(color="#b7cbb9"),margin=dict(l=10,r=10,t=10,b=10),xaxis=dict(gridcolor="#1e3b26"),yaxis=dict(gridcolor="#0e2115"))
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    table=view[["state","market_name","latest_price","latest_date"]].rename(columns={"state":"State","market_name":"Mandi","latest_price":"Modal Price","latest_date":"Date"}).copy()
    table["Modal Price"]=table["Modal Price"].map(lambda x:f"₹{x:,.0f}")
    table["Date"]=table["Date"].dt.strftime("%d %b %Y")
    st.dataframe(table,use_container_width=True,hide_index=True)
    st.markdown('</div>',unsafe_allow_html=True)

with t3:
    st.markdown('<div class="panel">',unsafe_allow_html=True)
    st.subheader("Model Information")
    m1,m2,m3=st.columns(3)
    m1.metric("Model","Random Forest")
    m2.metric("Threshold",f"{THRESHOLD:.2f}")
    m3.metric("Crash Definition","15% / 7 days")
    st.write("The application uses price lags, percentage changes, rolling averages, volatility, moving-average relationships, month and day-of-week features.")
    st.info("Model comparison files are optional. The prediction engine does not depend on the Git-LFS comparison CSVs.")
    st.markdown('</div>',unsafe_allow_html=True)

with t4:
    st.markdown('<div class="panel">',unsafe_allow_html=True)
    st.subheader("About the Project")
    x,y,z=st.columns(3)
    x.metric("Price Records",f"{len(history):,}")
    y.metric("Commodities",f"{all_snapshot.commodity.nunique()}")
    z.metric("Markets",f"{all_snapshot.market_name.nunique():,}")
    st.write("Crop Price Crash Predictor is a student ML decision-support application built from historical Agmarknet mandi prices. It estimates whether a significant short-term price fall is likely from historical patterns.")
    st.warning("This is not financial or agricultural advice. Weather, policy, supply shocks, transportation and other real-time factors can change prices unexpectedly.")
    st.markdown('</div>',unsafe_allow_html=True)

st.markdown('<p style="text-align:center;color:#64806c;margin-top:28px">Built with Python · Pandas · Scikit-learn · Random Forest · Plotly · Streamlit · Agmarknet</p>',unsafe_allow_html=True)
