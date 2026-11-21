import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date

st.set_page_config(
    page_title="Global Market Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Index universe
# -----------------------------
INDEX_MASTER = {
    "NIFTY 50": {"symbol": "^NSEI", "region": "India"},
    "BANK NIFTY": {"symbol": "^NSEBANK", "region": "India"},
    "SENSEX": {"symbol": "^BSESN", "region": "India"},
    "S&P 500": {"symbol": "^GSPC", "region": "USA"},
    "NASDAQ 100": {"symbol": "^NDX", "region": "USA"},
    "DOW JONES": {"symbol": "^DJI", "region": "USA"},
    "FTSE 100": {"symbol": "^FTSE", "region": "Europe"},
    "DAX 40": {"symbol": "^GDAXI", "region": "Europe"},
    "CAC 40": {"symbol": "^FCHI", "region": "Europe"},
    "IBEX 35": {"symbol": "^IBEX", "region": "Europe"},
    "SWISS MARKET INDEX": {"symbol": "^SSMI", "region": "Europe"},
    "NIKKEI 225": {"symbol": "^N225", "region": "Asia Pacific"},
    "HANG SENG": {"symbol": "^HSI", "region": "Asia Pacific"},
    "SHANGHAI COMPOSITE": {"symbol": "000001.SS", "region": "Asia Pacific"},
    "KOSPI": {"symbol": "^KS11", "region": "Asia Pacific"},
    "TAIWAN WEIGHTED": {"symbol": "^TWII", "region": "Asia Pacific"},
    "STRAITS TIMES": {"symbol": "^STI", "region": "Asia Pacific"},
    "ASX 200": {"symbol": "^AXJO", "region": "Asia Pacific"},
    "BRAZIL IBOVESPA": {"symbol": "^BVSP", "region": "Latin America"},
    "MEXICO IPC": {"symbol": "^MXX", "region": "Latin America"},
}

@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_data(period="1y"):
    rows = []
    histories = {}
    errors = []
    for name, info in INDEX_MASTER.items():
        symbol = info["symbol"]
        try:
            raw = yf.download(
                symbol, period=period, interval="1d",
                auto_adjust=True, progress=False, threads=False,
                timeout=25
            )
            if raw.empty:
                errors.append({"Index": name, "Symbol": symbol, "Issue": "No data returned"})
                continue

            close_data = raw["Close"]
            if isinstance(close_data, pd.DataFrame):
                close_data = close_data.iloc[:, 0]
            close = pd.to_numeric(close_data, errors="coerce").dropna()
            close = close[~close.index.duplicated(keep="last")].sort_index()

            if len(close) < 22:
                errors.append({"Index": name, "Symbol": symbol, "Issue": "Insufficient history"})
                continue

            latest = float(close.iloc[-1])
            prev = float(close.iloc[-2])

            def ret(n):
                if len(close) <= n:
                    return np.nan
                old = float(close.iloc[-(n + 1)])
                return (latest / old - 1) * 100 if old else np.nan

            sma20 = close.rolling(20).mean().iloc[-1]
            sma50 = close.rolling(50).mean().iloc[-1]
            sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan

            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
            rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50)
            rsi14 = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else np.nan

            if any(pd.isna(x) for x in [sma20, sma50, sma200]):
                trend = "INSUFFICIENT HISTORY"
            elif latest > sma20 > sma50 > sma200:
                trend = "STRONG BULLISH"
            elif latest < sma20 < sma50 < sma200:
                trend = "STRONG BEARISH"
            elif latest > sma50 and latest > sma200:
                trend = "BULLISH"
            elif latest < sma50 and latest < sma200:
                trend = "BEARISH"
            else:
                trend = "MIXED / SIDEWAYS"

            if pd.isna(rsi14):
                momentum = "UNAVAILABLE"
            elif rsi14 >= 70:
                momentum = "OVERBOUGHT"
            elif rsi14 <= 30:
                momentum = "OVERSOLD"
            else:
                momentum = "NEUTRAL"

            latest_date = pd.Timestamp(close.index[-1]).date()
            age_days = (date.today() - latest_date).days
            daily_return = (latest / prev - 1) * 100 if prev else np.nan

            quality = "NORMAL"
            if abs(daily_return) >= 3:
                quality = "VERIFY LARGE MOVE"
            if age_days > 7:
                quality = "CHECK STALE DATA"

            rows.append({
                "Index": name,
                "Region": info["region"],
                "Symbol": symbol,
                "Last Price": latest,
                "Daily %": daily_return,
                "Weekly %": ret(5),
                "Monthly %": ret(21),
                "Quarterly %": ret(63),
                "SMA 20": sma20,
                "SMA 50": sma50,
                "SMA 200": sma200,
                "RSI 14": rsi14,
                "Trend": trend,
                "Momentum": momentum,
                "Data Date": str(latest_date),
                "Data Age Days": age_days,
                "Quality Status": quality
            })
            histories[name] = close.rename(name)
        except Exception as exc:
            errors.append({"Index": name, "Symbol": symbol, "Issue": str(exc)[:180]})

    return pd.DataFrame(rows), histories, pd.DataFrame(errors)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("🌍 Market Scanner")
st.sidebar.caption("Global indices | End-of-day analysis")
period_label = st.sidebar.selectbox(
    "Historical data window",
    ["1 year", "2 years"],
    index=0
)
period = "1y" if period_label == "1 year" else "2y"
regions = sorted(set(v["region"] for v in INDEX_MASTER.values()))
selected_regions = st.sidebar.multiselect("Regions", regions, default=regions)
selected_trends = st.sidebar.multiselect(
    "Trend classifications",
    ["STRONG BULLISH", "BULLISH", "MIXED / SIDEWAYS", "BEARISH", "STRONG BEARISH", "INSUFFICIENT HISTORY"],
    default=["STRONG BULLISH", "BULLISH", "MIXED / SIDEWAYS", "BEARISH", "STRONG BEARISH", "INSUFFICIENT HISTORY"]
)
refresh = st.sidebar.button("↻ Refresh market data", use_container_width=True)
if refresh:
    st.cache_data.clear()

st.title("🌍 Global Market Intelligence")
st.caption("Global stock-index monitoring, technical indicators and regional comparisons")
st.info(
    "Data source: Yahoo Finance via the unofficial yfinance library. "
    "Data may be delayed, incomplete or unavailable for some indices. "
    "This dashboard is for research and education, not trading advice."
)

with st.spinner("Collecting available market data..."):
    data, histories, errors = fetch_market_data(period)

if data.empty:
    st.error("No index data was retrieved. Try refreshing later or check data-source availability.")
    st.stop()

filtered = data[
    data["Region"].isin(selected_regions) &
    data["Trend"].isin(selected_trends)
].copy()

# -----------------------------
# Overview metrics
# -----------------------------
latest_date = data["Data Date"].max()
bullish_count = data["Trend"].isin(["STRONG BULLISH", "BULLISH"]).sum()
bearish_count = data["Trend"].isin(["STRONG BEARISH", "BEARISH"]).sum()
best = data.loc[data["Daily %"].idxmax()]
worst = data.loc[data["Daily %"].idxmin()]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Indices retrieved", f"{len(data)} / {len(INDEX_MASTER)}")
c2.metric("Bullish", int(bullish_count))
c3.metric("Bearish", int(bearish_count))
c4.metric("Best daily", best["Index"], f"{best['Daily %']:+.2f}%")
c5.metric("Weakest daily", worst["Index"], f"{worst['Daily %']:+.2f}%")

st.caption(f"Latest observation date in retrieved data: {latest_date} | Dashboard cache: 15 minutes")

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_heatmap, tab_regions, tab_index, tab_quality, tab_export = st.tabs([
    "Overview", "Heatmap", "Regions", "Index Analysis", "Data Quality", "Export"
])

with tab_overview:
    st.subheader("Global index performance")
    sort_by = st.selectbox(
        "Sort table by",
        ["Daily %", "Weekly %", "Monthly %", "Quarterly %", "RSI 14", "Index"],
        key="overview_sort"
    )
    table = filtered.sort_values(sort_by, ascending=(sort_by != "Index"))
    display_cols = [
        "Index", "Region", "Last Price", "Daily %", "Weekly %",
        "Monthly %", "Quarterly %", "RSI 14", "Trend", "Momentum",
        "Data Date", "Quality Status"
    ]
    st.dataframe(
        table[display_cols].style.format({
            "Last Price": "{:,.2f}",
            "Daily %": "{:+.2f}%",
            "Weekly %": "{:+.2f}%",
            "Monthly %": "{:+.2f}%",
            "Quarterly %": "{:+.2f}%",
            "RSI 14": "{:.2f}"
        }, na_rep="—"),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Daily performance ranking")
    chart_df = filtered.sort_values("Daily %", ascending=True)
    fig = px.bar(
        chart_df, x="Daily %", y="Index", color="Daily %",
        color_continuous_scale="RdYlGn", orientation="h",
        hover_data=["Region", "Weekly %", "Monthly %", "Trend"]
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=650, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

with tab_heatmap:
    st.subheader("Global performance heatmap")
    heat = filtered.set_index("Index")[["Daily %", "Weekly %", "Monthly %", "Quarterly %"]]
    if not heat.empty:
        fig = px.imshow(
            heat, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
            labels={"color": "Return %"}
        )
        fig.update_layout(height=max(450, len(heat) * 32))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Returns are based on trading-session observations, not calendar-day intervals.")

with tab_regions:
    st.subheader("Regional comparison")
    regional = data.groupby("Region").agg(
        Indices=("Index", "count"),
        Median_Daily=("Daily %", "median"),
        Median_Weekly=("Weekly %", "median"),
        Median_Monthly=("Monthly %", "median"),
        Bullish=("Trend", lambda s: s.isin(["BULLISH", "STRONG BULLISH"]).sum()),
        Bearish=("Trend", lambda s: s.isin(["BEARISH", "STRONG BEARISH"]).sum())
    ).reset_index()
    st.dataframe(
        regional.style.format({
            "Median_Daily": "{:+.2f}%",
            "Median_Weekly": "{:+.2f}%",
            "Median_Monthly": "{:+.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )
    fig = px.bar(
        regional.sort_values("Median_Daily"),
        x="Median_Daily", y="Region", orientation="h",
        color="Median_Daily", color_continuous_scale="RdYlGn",
        labels={"Median_Daily": "Median daily return (%)"}
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

with tab_index:
    st.subheader("Individual index analysis")
    available = sorted(histories.keys())
    if available:
        chosen = st.selectbox("Choose an index", available)
        series = histories[chosen]
        detail = data[data["Index"] == chosen].iloc[0]

        a, b, c, d = st.columns(4)
        a.metric("Last price", f"{detail['Last Price']:,.2f}")
        b.metric("Daily change", f"{detail['Daily %']:+.2f}%")
        c.metric("RSI (14)", "—" if pd.isna(detail["RSI 14"]) else f"{detail['RSI 14']:.2f}")
        d.metric("Trend", detail["Trend"])

        chart = pd.DataFrame({"Close": series})
        chart["SMA 20"] = chart["Close"].rolling(20).mean()
        chart["SMA 50"] = chart["Close"].rolling(50).mean()
        chart["SMA 200"] = chart["Close"].rolling(200).mean()

        fig = go.Figure()
        for col in ["Close", "SMA 20", "SMA 50", "SMA 200"]:
            fig.add_trace(go.Scatter(
                x=chart.index, y=chart[col], name=col, mode="lines"
            ))
        fig.update_layout(
            title=f"{chosen} — closing price and moving averages",
            height=500, xaxis_title="Date", yaxis_title="Index level",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        ag = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        al = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = ag / al.replace(0, np.nan)
        rsi_series = 100 - 100 / (1 + rs)
        rsi_series = rsi_series.mask((al == 0) & (ag > 0), 100)
        rsi_series = rsi_series.mask((al == 0) & (ag == 0), 50)

        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=rsi_series.index, y=rsi_series, name="RSI 14"))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_layout(title="Relative Strength Index (14)", height=300, yaxis_range=[0, 100])
        st.plotly_chart(fig_rsi, use_container_width=True)

with tab_quality:
    st.subheader("Data quality and coverage")
    quality = data[[
        "Index", "Symbol", "Data Date", "Data Age Days",
        "Daily %", "Quality Status"
    ]].copy()
    st.dataframe(quality, use_container_width=True, hide_index=True)

    flagged = quality[quality["Quality Status"] != "NORMAL"]
    if not flagged.empty:
        st.warning("Some indices need manual verification.")
        st.dataframe(flagged, use_container_width=True, hide_index=True)
    else:
        st.success("No automatic quality flags were triggered. This does not independently verify market data.")

    if not errors.empty:
        st.subheader("Download issues")
        st.dataframe(errors, use_container_width=True, hide_index=True)
    else:
        st.success("No download errors were returned by the data provider.")

with tab_export:
    st.subheader("Download market data")
    csv_data = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download scanner results (CSV)",
        data=csv_data,
        file_name="global_market_scanner.csv",
        mime="text/csv"
    )

    history_frame = pd.concat(histories, axis=1) if histories else pd.DataFrame()
    if not history_frame.empty:
        history_csv = history_frame.to_csv().encode("utf-8")
        st.download_button(
            "Download historical prices (CSV)",
            data=history_csv,
            file_name="global_index_price_history.csv",
            mime="text/csv"
        )

    st.caption("For an Excel workbook, export the CSV files and open them in Excel. A multi-sheet Excel export can be added in a later iteration.")

st.divider()
st.caption(
    "Global Market Intelligence V2.0 | Educational and research tool. "
    "Not investment advice. Verify significant price moves with primary exchange data."
)
