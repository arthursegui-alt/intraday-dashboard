import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from entsoe import EntsoePandasClient
import time
import random

# ==========================================
# 1. DASHBOARD CONFIGURATION
# ==========================================
st.set_page_config(page_title="Trading Desk - Finland Grid", layout="wide")

st.title("🎛️ Finland Power Grid - Unified Trading Desk")

ENTSOE_KEY = "00f52b02-d674-4eab-af07-520c1673b3a7"
FINGRID_KEY = "5c21f14cbd3f47cabd205c60a4b5afa3"

# TIMEZONES CONFIGURATION
TZ_TARGET = 'Europe/Paris'     # Target timezone: CET / CEST
TZ_FINGRID = 'Europe/Helsinki' # Source timezone Fingrid: EET / EEST

# Dynamic current day target boundaries for layout reference
start_ts = pd.Timestamp.now(tz=TZ_TARGET).normalize()
end_ts = start_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

# String versions for Plotly axis boundaries
start_str_plotly = start_ts.strftime('%Y-%m-%d %H:%M:%S')
end_str_plotly = end_ts.strftime('%Y-%m-%d %H:%M:%S')

# ==========================================
# 2. UNIVERSAL FETCH ASSISTANT WITH RETRY & JITTER
# ==========================================
def fetch_fingrid_clean(dataset_id, lookback_hours=24, allow_future=False):
    now_utc = pd.Timestamp.now(tz='UTC')
    start_utc = now_utc - pd.Timedelta(hours=lookback_hours)
    
    if allow_future:
        end_utc = now_utc + pd.Timedelta(hours=12)
    else:
        end_utc = now_utc
    
    url = f"https://data.fingrid.fi/api/datasets/{dataset_id}/data"
    params = {
        "startTime": start_utc.strftime('%Y-%m-%dT%H:%M:%SZ'),
        "endTime": end_utc.strftime('%Y-%m-%dT%H:%M:%SZ'),
        "format": "json",
        "pageSize": 20000
    }
    
    max_retries = 3
    initial_backoff = 1.5
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers={"x-api-key": FINGRID_KEY}, params=params, timeout=10)
            
            if response.status_code == 429:
                sleep_duration = (initial_backoff * (2 ** attempt)) + random.uniform(0.4, 1.5)
                time.sleep(sleep_duration)
                continue
                
            if response.status_code != 200:
                return pd.Series(dtype='float64'), f"Error HTTP {response.status_code}"
                
            res = response.json()
            if isinstance(res, list): data_list = res
            elif isinstance(res, dict) and 'data' in res: data_list = res['data']
            else: data_list = []
            if len(data_list) == 0: return pd.Series(dtype='float64'), "Empty response"
            
            df = pd.DataFrame(data_list)
            df['startTime'] = pd.to_datetime(df['startTime'])
            
            if df['startTime'].dt.tz is None:
                df['startTime'] = df['startTime'].dt.tz_localize(TZ_FINGRID)
            
            series = df.set_index('startTime')['value'].sort_index().tz_convert(TZ_TARGET)
            return series, f"Connected ({len(series)} pts)"
            
        except Exception as e:
            return pd.Series(dtype='float64'), f"Exception: {str(e)}"
            
    return pd.Series(dtype='float64'), "Throttled (429)"

# ==========================================
# 3. CENTRALIZED DATA COLLECTION PIPELINE
# ==========================================
@st.cache_data(ttl=60)
def load_all_dispatch_data():
    client = EntsoePandasClient(api_key=ENTSOE_KEY)
    status_log = {}
    
    # --- 1. ENTSO-E: Physical Grid Operations ---
    try:
        df_gen = client.query_generation('FI', start=start_ts, end=end_ts).tz_convert(TZ_TARGET)
        ee_mix = df_gen.xs('Actual Aggregated', level=1, axis=1)
        ee_prod = ee_mix.sum(axis=1)
    except: ee_mix, ee_prod = pd.DataFrame(), pd.Series(dtype='float64')
    try: ee_prod_fc = client.query_generation_forecast('FI', start=start_ts, end=end_ts).tz_convert(TZ_TARGET)
    except: ee_prod_fc = pd.Series(dtype='float64')
    try:
        df_load = client.query_load('FI', start=start_ts, end=end_ts)
        ee_load = df_load.iloc[:, 0].tz_convert(TZ_TARGET) if isinstance(df_load, pd.DataFrame) else df_load.tz_convert(TZ_TARGET)
    except: ee_load = pd.Series(dtype='float64')
    try:
        df_load_fc = client.query_load_forecast('FI', start=start_ts, end=end_ts)
        ee_load_fc = df_load_fc.iloc[:, 0].tz_convert(TZ_TARGET) if isinstance(df_load_fc, pd.DataFrame) else df_load_fc.tz_convert(TZ_TARGET)
    except: ee_load_fc = pd.Series(dtype='float64')
    try: ee_day_ahead_prices = client.query_day_ahead_prices('FI', start=start_ts, end=end_ts).tz_convert(TZ_TARGET)
    except: ee_day_ahead_prices = pd.Series(dtype='float64')

    # Interconnections
    borders_map = {'Sweden (SE1)': 'SE_1', 'Sweden (SE3)': 'SE_3', 'Estonia (EE)': 'EE', 'Norway (NO4)': 'NO_4'}
    df_trade_actual, df_trade_forecast = pd.DataFrame(), pd.DataFrame()
    for name, code in borders_map.items():
        try:
            fwd_act = client.query_scheduled_exchanges(code, 'FI', start=start_ts, end=end_ts, dayahead=False)
            rev_act = client.query_scheduled_exchanges('FI', code, start=start_ts, end=end_ts, dayahead=False)
            df_trade_actual[name] = (rev_act - fwd_act).tz_convert(TZ_TARGET)
        except: pass
        try:
            fwd_fc = client.query_scheduled_exchanges(code, 'FI', start=start_ts, end=end_ts, dayahead=True)
            rev_fc = client.query_scheduled_exchanges('FI', code, start=start_ts, end=end_ts, dayahead=True)
            df_trade_forecast[name] = (rev_fc - fwd_fc).tz_convert(TZ_TARGET)
        except: pass
    total_actual = df_trade_actual.sum(axis=1) if not df_trade_actual.empty else pd.Series(dtype='float64')
    total_forecast = df_trade_forecast.sum(axis=1) if not df_trade_forecast.empty else pd.Series(dtype='float64')

    # --- 2. Fingrid API: Alpha Trading Logistics (With courtesy sleep offsets) ---
    mfrr_prices, status_log['mFRR Prices (400)'] = fetch_fingrid_clean(400, lookback_hours=24, allow_future=True)
    time.sleep(0.35)
    imbalance_prices, status_log['Imbalance Prices (319)'] = fetch_fingrid_clean(319, lookback_hours=24, allow_future=True)
    time.sleep(0.35)
    mfrr_up_vol, status_log['mFRR Up Realized Vol (375)'] = fetch_fingrid_clean(375, lookback_hours=24, allow_future=False)
    time.sleep(0.35)
    mfrr_down_vol, status_log['mFRR Down Realized Vol (376)'] = fetch_fingrid_clean(376, lookback_hours=24, allow_future=False)
    time.sleep(0.35)
    fi_balance, status_log['Live Net Balance (198)'] = fetch_fingrid_clean(198, lookback_hours=30, allow_future=False)
    time.sleep(0.35)
    mfrr_activated_mw, status_log['mFRR Live Activations MW (342)'] = fetch_fingrid_clean(342, lookback_hours=4, allow_future=False)
    time.sleep(0.35)
    wind_forecast, status_log['Wind Power Forecast (245)'] = fetch_fingrid_clean(245, lookback_hours=24, allow_future=True)
    time.sleep(0.35)
    mfrr_up_bids, status_log['OrderBook Up Bids Vol (373)'] = fetch_fingrid_clean(373, lookback_hours=12, allow_future=True)
    time.sleep(0.35)
    mfrr_down_bids, status_log['OrderBook Down Bids Vol (374)'] = fetch_fingrid_clean(374, lookback_hours=12, allow_future=True)
    time.sleep(0.35)
    afrr_up_price, status_log['aFRR PICASSO Price Up (352)'] = fetch_fingrid_clean(352, lookback_hours=12, allow_future=True)
    time.sleep(0.35)
    nuclear_prod, status_log['Nuclear Production TR (188)'] = fetch_fingrid_clean(188, lookback_hours=6, allow_future=False)
    time.sleep(0.35)
    wind_prod, status_log['Wind Production TR (181)'] = fetch_fingrid_clean(181, lookback_hours=12, allow_future=False)
    time.sleep(0.35)
    hydro_prod, status_log['Hydro Production TR (191)'] = fetch_fingrid_clean(191, lookback_hours=12, allow_future=False)
    time.sleep(0.35)
    electric_boilers, status_log['Electric Boilers TR (371)'] = fetch_fingrid_clean(371, lookback_hours=12, allow_future=False)
    
    return (ee_mix, ee_prod, ee_prod_fc, ee_load, ee_load_fc, ee_day_ahead_prices,
            df_trade_actual, df_trade_forecast, total_actual, total_forecast, 
            mfrr_prices, imbalance_prices, mfrr_up_vol, mfrr_down_vol, fi_balance,
            mfrr_activated_mw, wind_forecast, mfrr_up_bids, mfrr_down_bids, afrr_up_price,
            nuclear_prod, wind_prod, hydro_prod, electric_boilers, status_log)

# ==========================================
# 4. DATA INITIALIZATION & TOP HEADER
# ==========================================
(ee_mix, ee_prod, ee_prod_fc, ee_load, ee_load_fc, ee_da_prices, df_trade_actual, df_trade_forecast, 
 total_actual, total_forecast, mfrr, imbalance, mfrr_up, mfrr_down, fi_balance,
 mfrr_activated, wind_fc, up_bids, down_bids, afrr_price, nuclear_prod, wind_prod, 
 hydro_prod, electric_boilers, status_log) = load_all_dispatch_data()

# TACTICAL DIRECTION ASSISTANT
balance_latest = fi_balance.dropna().iloc[-1] if not fi_balance.empty and not fi_balance.dropna().empty else 0.0
head_col1, head_col2 = st.columns([3, 1])

if balance_latest < 0: 
    head_col1.error(f"🔴 LIVE LOCAL GRID DIRECTION (FI): UP (National Energy Deficit)")
    head_col2.metric("Instantaneous Balance FI", f"{balance_latest:,.1f} MW", "Requires upward bids activation", delta_color="inverse")
elif balance_latest > 0: 
    head_col1.info(f"🔵 LIVE LOCAL GRID DIRECTION (FI): DOWN (National Energy Surplus)")
    head_col2.metric("Instantaneous Balance FI", f"+{balance_latest:,.1f} MW", "Requires downward bids activation", delta_color="normal")
else:
    head_col1.success(f"🟢 LIVE LOCAL GRID DIRECTION (FI): STABLE")
    head_col2.metric("Instantaneous Balance FI", "0.0 MW", "Nominal equilibrium")

st.markdown("---")

# TWO STRATEGIC TABS IN ENGLISH
main_tabs = st.tabs(["🔮 Live Tools & Anticipation (Trading Desk)", "🌐 Global Situation & Macro Analysis"])

HOVER_LABEL_CONFIG = dict(bgcolor="rgba(255, 255, 255, 0.98)", font_size=15, font_family="Arial", font_color="#111111", bordercolor="rgba(0,0,0,0.15)")
CLEAN_MW_TEMPLATE = "%{y:,.1f} MW<extra></extra>"
CLEAN_EUR_TEMPLATE = "%{y:,.2f} €/MWh<extra></extra>"

# --------------------------------------------------------------------------------------
# TAB 1: LIVE TOOLS & ANTICIPATION (Trading Desk Execution Cockpit)
# --------------------------------------------------------------------------------------
with main_tabs[0]:
    @st.fragment(run_every=60)
    def render_trading_desk():
        (ee_mix, ee_prod, ee_prod_fc, ee_load, ee_load_fc, ee_da_prices, df_trade_actual, df_trade_forecast, 
         total_actual, total_forecast, mfrr, imbalance, mfrr_up, mfrr_down, fi_balance,
         mfrr_activated, wind_fc, up_bids, down_bids, afrr_price, nuclear_prod, wind_prod,
         hydro_prod, electric_boilers, status_log) = load_all_dispatch_data()
        
        now_ts = pd.Timestamp.now(tz=TZ_TARGET)
        now_str_plotly = now_ts.strftime('%Y-%m-%d %H:%M:%S')
        
        mfrr_price_latest = mfrr.dropna().iloc[-1] if not mfrr.dropna().empty else 0.0
        act_mw_latest = mfrr_activated.dropna().iloc[-1] if not mfrr_activated.dropna().empty else 0.0
        nuc_latest = nuclear_prod.dropna().iloc[-1] if not nuclear_prod.empty and not nuclear_prod.dropna().empty else 0.0
        wind_latest = wind_prod.dropna().iloc[-1] if not wind_prod.empty and not wind_prod.dropna().empty else 0.0
        hydro_latest = hydro_prod.dropna().iloc[-1] if not hydro_prod.empty and not hydro_prod.dropna().empty else 0.0
        boilers_latest = electric_boilers.dropna().iloc[-1] if not electric_boilers.empty and not electric_boilers.dropna().empty else 0.0
        balance_latest = fi_balance.dropna().iloc[-1] if not fi_balance.dropna().empty else 0.0
        balance_today = fi_balance[fi_balance.index >= start_ts] if not fi_balance.empty else pd.Series(dtype='float64')

        # KPI Panel (Tactical)
        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        t_col1.metric("mFRR Activation Price", f"{mfrr_price_latest:.2f} €/MWh")
        t_col2.metric("Live System Dispatch (342)", f"{act_mw_latest:,.1f} MW")
        t_col3.metric("Hydro Fleet Status (191)", f"{hydro_latest:,.0f} MW")
        t_col4.metric("Electric Boilers Absorption", f"{boilers_latest:,.1f} MW")

        st.markdown("---")
        
        # 1. Leading Price signals
        st.subheader("💶 Prices & aFRR Leading Indicators (PICASSO)")
        fig_market = go.Figure()
        if not ee_da_prices.empty: fig_market.add_trace(go.Scatter(x=ee_da_prices.index, y=ee_da_prices, name="Day-Ahead Spot Price", line=dict(color="#00bcd4", width=2.5), line_shape='hv'))
        if not mfrr.empty: fig_market.add_trace(go.Scatter(x=mfrr.index, y=mfrr, name="mFRR Price", line=dict(color="#ff5722", width=3), mode="lines+markers"))
        if not afrr_price.empty: fig_market.add_trace(go.Scatter(x=afrr_price.index, y=afrr_price, name="aFRR Picasso Price Up (Leading Signal)", line=dict(color="#9c27b0", width=1.5, dash="dot")))
        fig_market.add_vline(x=now_str_plotly, line_width=2, line_dash="dash", line_color="red")
        fig_market.update_layout(template="plotly_white", height=320, yaxis_title="€/MWh", hovermode="x unified")
        st.plotly_chart(fig_market, use_container_width=True)

        st.markdown("---")
        
        # 2. Interconnections
        st.subheader("🌐 Real-Time Interconnection Saturations & NTC Limits (>0 Export, <0 Import)")
        if not df_trade_actual.empty:
            fig_ntc = go.Figure()
            fig_ntc.add_trace(go.Scatter(x=df_trade_actual.index, y=df_trade_actual['Sweden (SE1)'], name="Flow FI-SE1 (North)", line=dict(color="#2ca02c", width=2.5), line_shape='hv'))
            fig_ntc.add_trace(go.Scatter(x=df_trade_actual.index, y=df_trade_actual['Sweden (SE3)'], name="Flow FI-SE3 (Central)", line=dict(color="#d62728", width=2.5), line_shape='hv'))
            fig_ntc.add_trace(go.Scatter(x=df_trade_actual.index, y=df_trade_actual['Estonia (EE)'], name="Flow FI-EE (Baltic)", line=dict(color="#9467bd", width=2.5), line_shape='hv'))
            
            fig_ntc.add_hline(y=-1500, line_dash="dash", line_color="#2ca02c", line_width=1.5, annotation_text="SE1 Import NTC Max (-1500MW)")
            fig_ntc.add_hline(y=1500, line_dash="dash", line_color="#2ca02c", line_width=1.5, annotation_text="SE1 Export NTC Max (+1500MW)")
            fig_ntc.add_hline(y=-1200, line_dash="dash", line_color="#d62728", line_width=1.5, annotation_text="SE3 Import NTC Max (-1200MW)")
            fig_ntc.add_hline(y=1200, line_dash="dash", line_color="#d62728", line_width=1.5, annotation_text="SE3 Export NTC Max (+1200MW)")
            fig_ntc.add_hline(y=-1000, line_dash="dash", line_color="#9467bd", line_width=1.5, annotation_text="EE Import NTC Max (-1000MW)")
            
            fig_ntc.add_hline(y=0, line_width=1, line_color="#222222")
            fig_ntc.add_vline(x=now_str_plotly, line_width=2, line_dash="dash", line_color="red")
            fig_ntc.update_layout(template="plotly_white", height=380, yaxis_title="Transmission Power (MW)", hovermode="x unified")
            st.plotly_chart(fig_ntc, use_container_width=True)

        st.markdown("---")
        
        # 3. Production anomalies & forecasts errors
        st.subheader("🚨 Live Generation Deltas & Deviations")
        alpha_col1, alpha_col2 = st.columns(2)
        with alpha_col1:
            st.markdown("**⚛️ Nuclear Outages Stability Watch (188)**")
            nuc_today = nuclear_prod[nuclear_prod.index >= start_ts] if not nuclear_prod.empty else pd.Series(dtype='float64')
            fig_nuc = go.Figure()
            if not nuc_today.empty: fig_nuc.add_trace(go.Scatter(x=nuc_today.index, y=nuc_today, name="Nuclear Power", line=dict(color="#ff9800", width=2.5)))
            fig_nuc.add_vline(x=now_str_plotly, line_width=2, line_dash="dash", line_color="red")
            fig_nuc.update_xaxes(range=[start_str_plotly, end_str_plotly])
            fig_nuc.update_layout(template="plotly_white", height=230, yaxis_title="MW", hovermode="x unified")
            st.plotly_chart(fig_nuc, use_container_width=True)
            
        with alpha_col2:
            st.markdown("**💨 Wind Fleet Imbalance Delta (Real 181 - Forecast 245)**")
            if not wind_prod.empty and not wind_fc.empty:
                df_wind = pd.DataFrame({'Real': wind_prod, 'Forecast': wind_fc}).dropna()
                df_wind['Delta'] = df_wind['Real'] - df_wind['Forecast']
                df_wind_today = df_wind[df_wind.index >= start_ts]
                fig_w_err = go.Figure()
                if not df_wind_today.empty: fig_w_err.add_trace(go.Scatter(x=df_wind_today.index, y=df_wind_today['Delta'], name="Wind Error", line=dict(color="#009688", width=2), fill='tozeroy'))
                fig_w_err.add_hline(y=0, line_width=1.5, line_color="#333333")
                fig_w_err.add_vline(x=now_str_plotly, line_width=2, line_dash="dash", line_color="red")
                fig_w_err.update_xaxes(range=[start_str_plotly, end_str_plotly])
                fig_w_err.update_layout(template="plotly_white", height=230, yaxis_title="Error Delta (MW)", hovermode="x unified")
                st.plotly_chart(fig_w_err, use_container_width=True)

        st.markdown("---")
        
        # 4. System logistics curves
        st.subheader("⚡ High-Frequency System Balance Dynamics")
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.markdown("**⚡ Live mFRR Balancing Energy Requests (342)**")
            mfrr_act_today = mfrr_activated[mfrr_activated.index >= start_ts] if not mfrr_activated.empty else pd.Series(dtype='float64')
            fig_act = go.Figure()
            if not mfrr_act_today.empty: fig_act.add_trace(go.Scatter(x=mfrr_act_today.index, y=mfrr_act_today, name="Active mFRR", line=dict(color="#d81b60", width=2), fill='tozeroy'))
            fig_act.add_hline(y=0, line_width=1.5, line_color="#333333")
            fig_act.add_vline(x=now_str_plotly, line_width=2, line_dash="dash", line_color="red")
            fig_act.update_xaxes(range=[start_str_plotly, end_str_plotly])
            fig_act.update_layout(template="plotly_white", height=230, yaxis_title="MW", hovermode="x unified")
            st.plotly_chart(fig_act, use_container_width=True)

        with v_col2:
            st.markdown("**⏱️ Real-Time Local Physical Balance (198)**")
            fig_bal = go.Figure()
            if not balance_today.empty: fig_bal.add_trace(go.Scatter(x=balance_today.index, y=balance_today, name="Local Balance", line=dict(color="#9c27b0", width=2), fill='tozeroy'))
            fig_bal.add_hline(y=0, line_width=1.5, line_color="#333333")
            fig_bal.add_vline(x=now_str_plotly, line_width=2, line_dash="dash", line_color="red")
            fig_bal.update_xaxes(range=[start_str_plotly, end_str_plotly])
            fig_bal.update_layout(template="plotly_white", height=230, yaxis_title="MW", hovermode="x unified")
            st.plotly_chart(fig_bal, use_container_width=True)

        # 5. Data Sheets (Logs & Predictions)
        st.markdown("---")
        st.subheader("📋 Order Book Depth & Full Intraday Logs")
        b_col1, b_col2 = st.columns([4, 3])
        with b_col1:
            st.markdown("🔮 **Upcoming Order Book Liquidity Window (Next 2 Hours Bids)**")
            if not up_bids.empty or not down_bids.empty:
                df_bids = pd.DataFrame(index=up_bids.index.union(down_bids.index))
                df_bids['AVAILABLE UP BIDS (MW)'] = up_bids
                df_bids['AVAILABLE DOWN BIDS (MW)'] = down_bids
                df_bids = df_bids[df_bids.index >= now_ts.floor('15min')].head(8)
                if not df_bids.empty:
                    df_bids['ORDERBOOK STATUS'] = ["THIN (Volatility Risk)" if x < 400 else "LIQUID" for x in df_bids['AVAILABLE UP BIDS (MW)'].fillna(0)]
                    df_bids.index = df_bids.index.strftime('%H:%M')
                    df_bids.index.name = 'PTU (CET)'
                    st.dataframe(df_bids, use_container_width=True)
        with b_col2:
            st.markdown("📋 **Full Daily Balance History Log (Dataset 198)**")
            if not balance_today.empty:
                df_bal_log = pd.DataFrame(balance_today).sort_index(ascending=False)
                df_bal_log.columns = ['BALANCE (MW)']
                df_bal_log.index = df_bal_log.index.strftime('%H:%M')
                st.dataframe(df_bal_log.style.format(precision=1), use_container_width=True, height=220)

    render_trading_desk()

# --------------------------------------------------------------------------------------
# TAB 2: GLOBAL SITUATION & MACRO ANALYSIS (Network Supervision & Adequacy)
# --------------------------------------------------------------------------------------
with main_tabs[1]:
    @st.fragment(run_every=60)
    def render_live_control_room():
        (ee_mix, ee_prod, ee_prod_fc, ee_load, ee_load_fc, ee_da_prices, df_trade_actual, df_trade_forecast, 
         total_actual, total_forecast, mfrr, imbalance, mfrr_up, mfrr_down, fi_balance,
         mfrr_activated, wind_fc, up_bids, down_bids, afrr_price, nuclear_prod, wind_prod,
         hydro_prod, electric_boilers, status_log) = load_all_dispatch_data()
        
        now_ts = pd.Timestamp.now(tz=TZ_TARGET)
        now_str_plotly = now_ts.strftime('%Y-%m-%d %H:%M:%S')
        
        with st.expander("🔌 Connection Telemetry & System Status logs"):
            st.json(status_log)

        prod_val = ee_prod.dropna().iloc[-1] if not ee_prod.dropna().empty else 0.0
        load_val = ee_load.dropna().iloc[-1] if not ee_load.dropna().empty else 0.0
        net_val = total_actual.dropna().iloc[-1] if not total_actual.dropna().empty else 0.0

        # Structural Metrics Panels
        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        kpi_col1.metric("Total Generation (FI)", f"{prod_val:,.1f} MW")
        kpi_col2.metric("Total Load (FI)", f"{load_val:,.1f} MW")
        kpi_col3.metric("Net Commercial Position", f"{net_val:,.1f} MW")

        st.markdown("---")
        
        # Macro Chart 1: Global Trade
        st.subheader("🌐 Global Net Commercial Position (BZN|FI) — ENTSO-E")
        fig_net = go.Figure()
        if not total_actual.empty: fig_net.add_trace(go.Scatter(x=total_actual.index, y=total_actual, name="Net Position", line=dict(color="#1f77b4", width=3), fill='tozeroy', line_shape='hv'))
        fig_net.add_hline(y=0, line_width=2, line_color="#111111")
        fig_net.add_vline(x=now_str_plotly, line_width=2, line_dash="dash", line_color="red")
        fig_net.update_layout(template="plotly_white", height=320, yaxis_title="MW (>0 Export, <0 Import)", hovermode="x unified")
        st.plotly_chart(fig_net, use_container_width=True)

        st.markdown("---")
        
        # Macro Chart 2: Consumption Profiles
        st.subheader("📉 National Consumption Curve (Load actual vs forecast)")
        if not ee_load.empty or not ee_load_fc.empty:
            fig_load = go.Figure()
            if not ee_load_fc.empty: fig_load.add_trace(go.Scatter(x=ee_load_fc.index, y=ee_load_fc, name="Load Forecast", line=dict(color="#009688", width=2), line_shape='hv'))
            if not ee_load.empty: fig_load.add_trace(go.Scatter(x=ee_load.index, y=ee_load, name="Actual Load", line=dict(color="#d81b60", width=2.5), line_shape='hv'))
            fig_load.add_vline(x=now_str_plotly, line_width=2, line_dash="dash", line_color="red")
            fig_load.update_layout(template="plotly_white", height=350, yaxis_title="MW", hovermode="x unified")
            st.plotly_chart(fig_load, use_container_width=True)

        st.markdown("---")
        
        # Macro Chart 3: Generation Profiles
        st.subheader("📈 National Production Curve (Generation actual vs forecast)")
        if not ee_prod.empty or not ee_prod_fc.empty:
            fig_gen = go.Figure()
            if not ee_prod_fc.empty: fig_gen.add_trace(go.Scatter(x=ee_prod_fc.index, y=ee_prod_fc, name="Forecast Generation", line=dict(color="#ffa600", width=2), line_shape='hv'))
            if not ee_prod.empty: fig_gen.add_trace(go.Scatter(x=ee_prod.index, y=ee_prod, name="Actual Production", line=dict(color="#003f5c", width=2.5), line_shape='hv'))
            fig_gen.add_vline(x=now_str_plotly, line_width=2, line_dash="dash", line_color="red")
            fig_gen.update_layout(template="plotly_white", height=350, yaxis_title="MW", hovermode="x unified")
            st.plotly_chart(fig_gen, use_container_width=True)

        st.markdown("---")
        
        # Macro Chart 4: Fuel Mix
        st.subheader("📊 National Generation Technological Mix Breakdown")
        if not ee_mix.empty:
            fig_mix = px.area(ee_mix, x=ee_mix.index, y=ee_mix.columns)
            fig_mix.update_layout(template="plotly_white", height=320, xaxis_title="Time (CET)", yaxis_title="MW", hovermode="x unified")
            st.plotly_chart(fig_mix, use_container_width=True)

        # Border Exchanges Spreadsheet
        st.markdown("📋 **Exhaustive Hourly Exchanges Log Matrix (Values in MW)**")
        if not df_trade_actual.empty:
            df_matrix = pd.DataFrame(index=df_trade_actual.index)
            if not total_actual.empty: df_matrix['NET POSITION (TOTAL)'] = total_actual
            for country in ['Sweden (SE1)', 'Sweden (SE3)', 'Estonia (EE)', 'Norway (NO4)']:
                if country in df_trade_actual.columns: df_matrix[f"{country.upper()}"] = df_trade_actual[country]
            df_matrix.index = df_matrix.index.strftime('%H:%M')
            st.dataframe(df_matrix, height=180, use_container_width=True)

    render_live_control_room()