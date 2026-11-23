import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
from io import BytesIO

st.set_page_config(
    page_title="Rakesh | Global Market Intelligence V2.2",
    page_icon="🌍",
    layout="wide"
)

APP_VERSION = "2.2"
STALE_AFTER_BUSINESS_DAYS = 5
LARGE_MOVE_THRESHOLD = 5.0

INDEXES = {
    "NIFTY 50": ("^NSEI", "India"),
    "BANK NIFTY": ("^NSEBANK", "India"),
    "SENSEX": ("^BSESN", "India"),
    "S&P 500": ("^GSPC", "USA"),
    "NASDAQ 100": ("^NDX", "USA"),
    "DOW JONES": ("^DJI", "USA"),
    "FTSE 100": ("^FTSE", "Europe"),
    "DAX 40": ("^GDAXI", "Europe"),
    "CAC 40": ("^FCHI", "Europe"),
    "IBEX 35": ("^IBEX", "Europe"),
    "SWISS MARKET INDEX": ("^SSMI", "Europe"),
    "NIKKEI 225": ("^N225", "Asia Pacific"),
    "HANG SENG": ("^HSI", "Asia Pacific"),
    "SHANGHAI COMPOSITE": ("000001.SS", "Asia Pacific"),
    "KOSPI": ("^KS11", "Asia Pacific"),
    "TAIWAN WEIGHTED": ("^TWII", "Asia Pacific"),
    "STRAITS TIMES": ("^STI", "Asia Pacific"),
    "ASX 200": ("^AXJO", "Asia Pacific"),
    "BRAZIL IBOVESPA": ("^BVSP", "Latin America"),
    "MEXICO IPC": ("^MXX", "Latin America"),
}

TRENDS = [
    "STRONG BULLISH", "BULLISH", "MIXED / SIDEWAYS",
    "BEARISH", "STRONG BEARISH", "INSUFFICIENT HISTORY"
]


def rsi_calc(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result = result.mask((avg_down == 0) & (avg_up > 0), 100)
    result = result.mask((avg_down == 0) & (avg_up == 0), 50)
    return result


def period_return(series, sessions):
    if len(series) <= sessions:
        return np.nan
    base = float(series.iloc[-sessions - 1])
    latest = float(series.iloc[-1])
    if base == 0 or not np.isfinite(base):
        return np.nan
    return (latest / base - 1) * 100


def classify_trend(price, sma20, sma50, sma200):
    if any(pd.isna(value) for value in [sma20, sma50, sma200]):
        return "INSUFFICIENT HISTORY"
    if price > sma20 > sma50 > sma200:
        return "STRONG BULLISH"
    if price < sma20 < sma50 < sma200:
        return "STRONG BEARISH"
    if price > sma50 and price > sma200:
        return "BULLISH"
    if price < sma50 and price < sma200:
        return "BEARISH"
    return "MIXED / SIDEWAYS"


def get_ohlcv_column(raw, field):
    """Handle both ordinary and yfinance MultiIndex column layouts."""
    if isinstance(raw.columns, pd.MultiIndex):
        matches = [column for column in raw.columns if field.lower() in [str(x).lower() for x in column]]
        if not matches:
            return pd.Series(index=raw.index, dtype="float64")
        values = raw[matches[0]]
        if isinstance(values, pd.DataFrame):
            values = values.iloc[:, 0]
        return pd.to_numeric(values, errors="coerce")
    if field not in raw.columns:
        return pd.Series(index=raw.index, dtype="float64")
    values = raw[field]
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]
    return pd.to_numeric(values, errors="coerce")


def business_days_since(observation_date):
    """Weekdays elapsed; exchange-specific public holidays are not modelled."""
    try:
        observed = pd.Timestamp(observation_date).normalize()
        today = pd.Timestamp(date.today()).normalize()
        if observed >= today:
            return 0
        return max(0, len(pd.bdate_range(observed + pd.Timedelta(days=1), today)))
    except Exception:
        return np.nan


def quality_score(available, duplicate_count, ohlc_invalid, close_invalid,
                  business_age, daily_return, history_length):
    """Transparent 100-point heuristic; not an independent price verification."""
    score = 0
    score += 25 if available else 0
    score += 15 if duplicate_count == 0 else 0
    score += 20 if ohlc_invalid == 0 else 0
    score += 15 if close_invalid == 0 else 0
    score += 15 if pd.notna(business_age) and business_age <= STALE_AFTER_BUSINESS_DAYS else 0
    score += 5 if pd.notna(daily_return) and abs(daily_return) < LARGE_MOVE_THRESHOLD else 0
    score += 5 if history_length >= 200 else (3 if history_length >= 50 else 0)
    return int(score)


@st.cache_data(ttl=900, show_spinner=False)
def get_data(period):
    rows = []
    histories = {}
    issues = []

    for name, (symbol, region) in INDEXES.items():
        try:
            raw = yf.download(
                symbol,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
                timeout=25
            )
            if raw is None or raw.empty:
                raise ValueError("No data returned by Yahoo Finance")

            close_raw = get_ohlcv_column(raw, "Close")
            open_raw = get_ohlcv_column(raw, "Open")
            high_raw = get_ohlcv_column(raw, "High")
            low_raw = get_ohlcv_column(raw, "Low")

            duplicate_count = int(raw.index.duplicated(keep=False).sum())
            raw_dates = pd.to_datetime(raw.index, errors="coerce")
            valid_date_mask = ~pd.isna(raw_dates)

            frame = pd.DataFrame({
                "Open": open_raw,
                "High": high_raw,
                "Low": low_raw,
                "Close": close_raw,
            }, index=raw.index)
            frame = frame.loc[valid_date_mask].copy()
            frame = frame[~frame.index.duplicated(keep="last")].sort_index()

            close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
            if len(close) < 22:
                raise ValueError(f"Insufficient usable close history: {len(close)} rows")

            close_invalid = int((close <= 0).sum())
            ohlc_frame = frame[["Open", "High", "Low", "Close"]].dropna()
            if ohlc_frame.empty:
                ohlc_invalid = 0
            else:
                invalid_ohlc = (
                    (ohlc_frame[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
                    | (ohlc_frame["High"] < ohlc_frame["Low"])
                    | (ohlc_frame["High"] < ohlc_frame[["Open", "Close"]].max(axis=1))
                    | (ohlc_frame["Low"] > ohlc_frame[["Open", "Close"]].min(axis=1))
                )
                ohlc_invalid = int(invalid_ohlc.sum())

            latest_price = float(close.iloc[-1])
            sma20 = close.rolling(20).mean().iloc[-1]
            sma50 = close.rolling(50).mean().iloc[-1]
            sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
            rsi_value = rsi_calc(close).iloc[-1]
            rsi_value = float(rsi_value) if pd.notna(rsi_value) else np.nan

            daily = period_return(close, 1)
            weekly = period_return(close, 5)
            monthly = period_return(close, 21)
            quarterly = period_return(close, 63)

            observation_date = pd.Timestamp(close.index[-1]).date()
            calendar_age = max(0, (date.today() - observation_date).days)
            business_age = business_days_since(observation_date)
            history_length = len(close)

            momentum = (
                "UNAVAILABLE" if pd.isna(rsi_value)
                else "OVERBOUGHT" if rsi_value >= 70
                else "OVERSOLD" if rsi_value <= 30
                else "NEUTRAL"
            )

            flags = []
            if duplicate_count:
                flags.append(f"DUPLICATE DATE ROWS: {duplicate_count}")
            if close_invalid:
                flags.append(f"NON-POSITIVE CLOSES: {close_invalid}")
            if ohlc_invalid:
                flags.append(f"INVALID OHLC ROWS: {ohlc_invalid}")
            if pd.isna(daily):
                flags.append("DAILY RETURN UNAVAILABLE")
            elif abs(daily) >= LARGE_MOVE_THRESHOLD:
                flags.append("VERIFY LARGE DAILY MOVE")
            if pd.isna(business_age) or business_age > STALE_AFTER_BUSINESS_DAYS:
                flags.append("POTENTIALLY STALE DATA")
            if history_length < 200:
                flags.append("LESS THAN 200 SESSIONS")
            if history_length < 50:
                flags.append("INSUFFICIENT HISTORY FOR SMA 50")
            if pd.isna(rsi_value):
                flags.append("RSI 14 UNAVAILABLE")

            score = quality_score(
                True, duplicate_count, ohlc_invalid, close_invalid,
                business_age, daily, history_length
            )
            if score >= 90:
                status = "GOOD"
            elif score >= 70:
                status = "REVIEW"
            else:
                status = "POOR"

            rows.append({
                "Index": name,
                "Region": region,
                "Symbol": symbol,
                "Last Price": latest_price,
                "Daily %": daily,
                "Weekly %": weekly,
                "Monthly %": monthly,
                "Quarterly %": quarterly,
                "SMA 20": sma20,
                "SMA 50": sma50,
                "SMA 200": sma200,
                "RSI 14": rsi_value,
                "Trend": classify_trend(latest_price, sma20, sma50, sma200),
                "Momentum": momentum,
                "Data Date": str(observation_date),
                "Calendar Age Days": calendar_age,
                "Business Days Since Update": business_age,
                "History Rows": history_length,
                "Duplicate Date Rows": duplicate_count,
                "Invalid OHLC Rows": ohlc_invalid,
                "Non-positive Close Rows": close_invalid,
                "Quality Score": score,
                "Quality Status": status,
                "Quality Flags": "; ".join(flags) if flags else "No automatic flags",
            })
            histories[name] = close.rename(name)

        except Exception as error:
            issues.append({
                "Index": name,
                "Symbol": symbol,
                "Issue": str(error)[:250]
            })

    return pd.DataFrame(rows), histories, pd.DataFrame(issues)


def build_regional_summary(data):
    if data.empty:
        return pd.DataFrame()
    return (
        data.groupby("Region")
        .agg(
            Indices=("Index", "count"),
            Median_Daily=("Daily %", "median"),
            Median_Weekly=("Weekly %", "median"),
            Median_Monthly=("Monthly %", "median"),
            Bullish=("Trend", lambda values: values.isin(["BULLISH", "STRONG BULLISH"]).sum()),
            Bearish=("Trend", lambda values: values.isin(["BEARISH", "STRONG BEARISH"]).sum()),
            Average_Quality=("Quality Score", "mean"),
        )
        .reset_index()
    )


def make_excel(data, regional, weekly, monthly, quality, issues, history):
    output = BytesIO()
    sheets = {
        "Global Scanner": data,
        "Regional Comparison": regional,
        "Weekly Ranking": weekly,
        "Monthly Ranking": monthly,
        "Data Quality": quality,
        "Download Issues": issues,
        "Price History": history,
    }
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            safe_frame = frame.copy()
            safe_frame.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.book[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column in worksheet.columns:
                width = max(
                    (len(str(cell.value or "")) for cell in column),
                    default=10
                )
                worksheet.column_dimensions[column[0].column_letter].width = min(max(width + 2, 12), 38)
    output.seek(0)
    return output.getvalue()


# Sidebar controls
st.sidebar.markdown("## 🌍 Rakesh's Market Scanner")
st.sidebar.caption("Global indices • Investors • Traders")
period_label = st.sidebar.selectbox("Historical window", ["1 year", "2 years"])
period = "1y" if period_label == "1 year" else "2y"
all_regions = sorted({value[1] for value in INDEXES.values()})
selected_regions = st.sidebar.multiselect("Regions", all_regions, default=all_regions)
selected_trends = st.sidebar.multiselect("Trend filters", TRENDS, default=TRENDS)
search_text = st.sidebar.text_input("Find an index", placeholder="e.g. NASDAQ")
if st.sidebar.button("↻ Refresh data now", use_container_width=True):
    get_data.clear()

st.title("🌍 Rakesh Parkar | Global Market Intelligence")
st.caption(f"Version {APP_VERSION} • Global index research workspace")
st.info(
    "Data source: Yahoo Finance through the unofficial yfinance library. "
    "Data may be delayed, incomplete or revised and is not guaranteed real-time. "
    "Freshness checks exclude weekends but do not model each exchange's public holidays. "
    "Verify significant moves and important figures with exchange or licensed sources. "
    "Research tool only; not investment advice."
)

with st.spinner("Loading global index data..."):
    data, histories, issues = get_data(period)

if data.empty:
    st.error("No market data retrieved. Try refreshing later. Check the Download Issues section if available.")
    if not issues.empty:
        st.dataframe(issues, use_container_width=True, hide_index=True)
    st.stop()

view = data[
    data["Region"].isin(selected_regions)
    & data["Trend"].isin(selected_trends)
].copy()
if search_text.strip():
    view = view[view["Index"].str.contains(search_text.strip(), case=False, na=False)]

bull_count = int(data["Trend"].isin(["STRONG BULLISH", "BULLISH"]).sum())
bear_count = int(data["Trend"].isin(["STRONG BEARISH", "BEARISH"]).sum())
advancers = int((data["Daily %"] > 0).sum())
decliners = int((data["Daily %"] < 0).sum())
valid_daily = data.dropna(subset=["Daily %"])
best = valid_daily.loc[valid_daily["Daily %"].idxmax()] if not valid_daily.empty else None
worst = valid_daily.loc[valid_daily["Daily %"].idxmin()] if not valid_daily.empty else None
latest_observation = data["Data Date"].max()
average_quality = data["Quality Score"].mean()
above_sma = [
    int((data["Last Price"] > data[f"SMA {window}"]).sum())
    for window in [20, 50, 200]
]

st.markdown("### Global market snapshot")
snapshot = st.columns(5)
snapshot[0].metric("Indices retrieved", f"{len(data)} / {len(INDEXES)}")
snapshot[1].metric("Bullish trend", bull_count)
snapshot[2].metric("Bearish trend", bear_count)
snapshot[3].metric(
    "Best daily",
    best["Index"] if best is not None else "—",
    f"{best['Daily %']:+.2f}%" if best is not None else None
)
snapshot[4].metric(
    "Weakest daily",
    worst["Index"] if worst is not None else "—",
    f"{worst['Daily %']:+.2f}%" if worst is not None else None
)
st.caption(
    f"Latest observation date across retrieved indices: {latest_observation} | "
    f"Average heuristic quality score: {average_quality:.1f}/100 | Cache: 15 minutes"
)

tabs = st.tabs([
    "Market Overview",
    "Global Heatmap",
    "Regional View",
    "Bullish & Bearish",
    "Index Deep Dive",
    "Data Quality",
    "Research Export",
])

with tabs[0]:
    st.subheader("Global index performance")
    sort_column = st.selectbox(
        "Rank table by",
        ["Daily %", "Weekly %", "Monthly %", "Quarterly %", "RSI 14", "Quality Score", "Index"]
    )
    sorted_view = view.sort_values(
        sort_column,
        ascending=(sort_column == "Index"),
        na_position="last"
    )
    visible_columns = [
        "Index", "Region", "Last Price", "Daily %", "Weekly %", "Monthly %",
        "Quarterly %", "RSI 14", "Trend", "Momentum", "Data Date",
        "Quality Score", "Quality Status"
    ]
    st.dataframe(
        sorted_view[visible_columns].style.format({
            "Last Price": "{:,.2f}",
            "Daily %": "{:+.2f}%",
            "Weekly %": "{:+.2f}%",
            "Monthly %": "{:+.2f}%",
            "Quarterly %": "{:+.2f}%",
            "RSI 14": "{:.1f}",
            "Quality Score": "{:.0f}",
        }, na_rep="—"),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Global market breadth")
    breadth_columns = st.columns(5)
    breadth_columns[0].metric("Advancers", advancers)
    breadth_columns[1].metric("Decliners", decliners)
    for position, window in enumerate([20, 50, 200]):
        breadth_columns[position + 2].metric(
            f"Above SMA {window}",
            f"{above_sma[position]}/{len(data)}"
        )
    breadth_frame = pd.DataFrame({
        "Measure": ["Above SMA 20", "Above SMA 50", "Above SMA 200"],
        "Indices": above_sma
    })
    figure = px.bar(
        breadth_frame, x="Measure", y="Indices", text="Indices",
        title="Index participation above moving averages"
    )
    figure.update_layout(yaxis_range=[0, len(data)])
    st.plotly_chart(figure, use_container_width=True)

with tabs[1]:
    st.subheader("Global market heatmap")
    heatmap = view.set_index("Index")[["Daily %", "Weekly %", "Monthly %", "Quarterly %"]]
    if not heatmap.empty:
        figure = px.imshow(
            heatmap,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            labels={"color": "Return %"}
        )
        figure.update_layout(height=max(480, len(heatmap) * 30))
        st.plotly_chart(figure, use_container_width=True)
    else:
        st.warning("No indices match the current filters.")

with tabs[2]:
    st.subheader("Regional comparison")
    regional = build_regional_summary(data)
    st.dataframe(
        regional.style.format({
            "Median_Daily": "{:+.2f}%",
            "Median_Weekly": "{:+.2f}%",
            "Median_Monthly": "{:+.2f}%",
            "Average_Quality": "{:.1f}",
        }, na_rep="—"),
        use_container_width=True,
        hide_index=True
    )
    if not regional.empty:
        figure = px.bar(
            regional.sort_values("Median_Weekly"),
            x="Median_Weekly",
            y="Region",
            orientation="h",
            color="Median_Weekly",
            color_continuous_scale="RdYlGn",
            title="Regional median weekly return"
        )
        figure.add_vline(x=0, line_dash="dash", line_color="gray")
        figure.update_layout(coloraxis_showscale=False)
        st.plotly_chart(figure, use_container_width=True)

with tabs[3]:
    st.subheader("Strongest and weakest indices")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Strongest weekly")
        st.dataframe(
            data.sort_values("Weekly %", ascending=False)[
                ["Index", "Region", "Weekly %", "Monthly %", "Trend"]
            ].head(10),
            use_container_width=True,
            hide_index=True
        )
        st.markdown("#### Strongest monthly")
        st.dataframe(
            data.sort_values("Monthly %", ascending=False)[
                ["Index", "Region", "Monthly %", "Trend"]
            ].head(10),
            use_container_width=True,
            hide_index=True
        )
    with right:
        st.markdown("#### Weakest weekly")
        st.dataframe(
            data.sort_values("Weekly %")[
                ["Index", "Region", "Weekly %", "Monthly %", "Trend"]
            ].head(10),
            use_container_width=True,
            hide_index=True
        )
        st.markdown("#### Weakest monthly")
        st.dataframe(
            data.sort_values("Monthly %")[
                ["Index", "Region", "Monthly %", "Trend"]
            ].head(10),
            use_container_width=True,
            hide_index=True
        )

with tabs[4]:
    st.subheader("Individual index analysis")
    available_indexes = sorted(histories.keys())
    if available_indexes:
        selected_index = st.selectbox("Select an index", available_indexes)
        series = histories[selected_index]
        selected_row = data[data["Index"] == selected_index].iloc[0]
        metrics = st.columns(5)
        metrics[0].metric("Last price", f"{selected_row['Last Price']:,.2f}")
        metrics[1].metric("Daily return", f"{selected_row['Daily %']:+.2f}%")
        metrics[2].metric(
            "RSI 14",
            "—" if pd.isna(selected_row["RSI 14"]) else f"{selected_row['RSI 14']:.1f}"
        )
        metrics[3].metric("Trend", selected_row["Trend"])
        metrics[4].metric("Quality", f"{selected_row['Quality Score']}/100")

        chart_data = pd.DataFrame({"Close": series})
        for window in [20, 50, 200]:
            chart_data[f"SMA {window}"] = chart_data["Close"].rolling(window).mean()
        figure = go.Figure()
        for column in ["Close", "SMA 20", "SMA 50", "SMA 200"]:
            figure.add_trace(go.Scatter(
                x=chart_data.index, y=chart_data[column], name=column
            ))
        figure.update_layout(
            title=f"{selected_index}: closing price and moving averages",
            height=480,
            hovermode="x unified"
        )
        st.plotly_chart(figure, use_container_width=True)

        rsi_series = rsi_calc(series)
        rsi_figure = go.Figure(go.Scatter(
            x=rsi_series.index, y=rsi_series, name="RSI 14"
        ))
        rsi_figure.add_hline(y=70, line_dash="dash", line_color="red")
        rsi_figure.add_hline(y=30, line_dash="dash", line_color="green")
        rsi_figure.update_layout(
            title="Relative Strength Index (14)",
            height=280,
            yaxis_range=[0, 100]
        )
        st.plotly_chart(rsi_figure, use_container_width=True)
    else:
        st.warning("No index history is available for charting.")

with tabs[5]:
    st.subheader("Data quality and reliability monitoring")
    st.caption(
        "The score is a transparent automated screening heuristic, not a probability "
        "that a quote is correct and not independent verification against an exchange."
    )

    quality_columns = st.columns(4)
    quality_columns[0].metric("Average quality score", f"{average_quality:.1f}/100")
    quality_columns[1].metric(
        "Good",
        int((data["Quality Status"] == "GOOD").sum())
    )
    quality_columns[2].metric(
        "Needs review",
        int((data["Quality Status"] == "REVIEW").sum())
    )
    quality_columns[3].metric(
        "Poor",
        int((data["Quality Status"] == "POOR").sum())
    )

    quality_view = data[[
        "Index", "Symbol", "Data Date", "Calendar Age Days",
        "Business Days Since Update", "History Rows",
        "Duplicate Date Rows", "Invalid OHLC Rows",
        "Non-positive Close Rows", "Daily %", "Quality Score",
        "Quality Status", "Quality Flags"
    ]].copy()

    st.dataframe(
        quality_view.style.format({
            "Daily %": "{:+.2f}%",
            "Quality Score": "{:.0f}",
        }, na_rep="—"),
        use_container_width=True,
        hide_index=True
    )

    flagged = quality_view[
        (quality_view["Quality Status"] != "GOOD")
        | (quality_view["Quality Flags"] != "No automatic flags")
    ]
    if not flagged.empty:
        st.warning(
            "Review the following records. A warning is a prompt for investigation, "
            "not proof that the market quote is wrong."
        )
        st.dataframe(flagged, use_container_width=True, hide_index=True)
    else:
        st.success("No automatic quality warnings were generated.")

    if not issues.empty:
        st.subheader("Download or retrieval failures")
        st.dataframe(issues, use_container_width=True, hide_index=True)
    else:
        st.success("No download failures were returned for this refresh.")

    with st.expander("How the quality score is calculated"):
        st.markdown(
            """
            The 100-point heuristic is composed of:

            - **25 points:** price history was retrieved and processed.
            - **15 points:** no duplicate date rows were detected.
            - **20 points:** no invalid OHLC relationships were detected.
            - **15 points:** no non-positive closing prices were detected.
            - **15 points:** latest observation is within five weekdays.
            - **5 points:** latest daily move is below the large-move review threshold.
            - **5 points:** at least 200 observations are available (3 points for 50–199).

            Status bands: **GOOD = 90–100**, **REVIEW = 70–89**, **POOR = below 70**.

            Weekends are excluded from freshness age. Exchange-specific holidays are not
            modelled, so holiday periods can still create a false stale-data warning.
            A large move is a verification prompt, not an automatic data error.
            """
        )

with tabs[6]:
    st.subheader("Download complete research workbook")
    regional = build_regional_summary(data)
    weekly = data.sort_values("Weekly %", ascending=False, na_position="last").reset_index(drop=True)
    weekly.insert(0, "Weekly Rank", range(1, len(weekly) + 1))
    monthly = data.sort_values("Monthly %", ascending=False, na_position="last").reset_index(drop=True)
    monthly.insert(0, "Monthly Rank", range(1, len(monthly) + 1))
    quality = data[[
        "Index", "Symbol", "Data Date", "Calendar Age Days",
        "Business Days Since Update", "History Rows",
        "Duplicate Date Rows", "Invalid OHLC Rows",
        "Non-positive Close Rows", "Daily %", "Quality Score",
        "Quality Status", "Quality Flags"
    ]]
    history = pd.concat(histories, axis=1) if histories else pd.DataFrame()
    workbook = make_excel(
        data,
        regional,
        weekly,
        monthly,
        quality,
        issues,
        history.reset_index() if not history.empty else history
    )
    st.download_button(
        "⬇ Download complete Excel research report",
        workbook,
        file_name=f"Rakesh_Global_Market_Intelligence_V2_2_{date.today().isoformat()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
    st.caption(
        "Seven worksheets: Global Scanner, Regional Comparison, Weekly Ranking, "
        "Monthly Ranking, Data Quality, Download Issues and Price History."
    )

st.divider()
st.subheader("Daily market pulse")
if best is not None and worst is not None:
    st.write(
        f"Across {len(data)} retrieved indices, {advancers} advanced and {decliners} declined. "
        f"{bull_count} meet the bullish trend rules and {bear_count} meet the bearish rules. "
        f"{above_sma[2]} of {len(data)} indices are above their 200-session moving average. "
        f"Best daily performer: {best['Index']} ({best['Daily %']:+.2f}%). "
        f"Weakest: {worst['Index']} ({worst['Daily %']:+.2f}%)."
    )
else:
    st.write(
        f"Across {len(data)} retrieved indices, {advancers} advanced and {decliners} declined. "
        "Daily best/weakest ranking is unavailable because daily returns are missing."
    )

st.caption(
    f"Rakesh Parkar | Global Market Intelligence V{APP_VERSION} • "
    "Rule-based analytics, not AI-generated advice. Yahoo Finance/yfinance coverage "
    "and data quality are not guaranteed."
)
