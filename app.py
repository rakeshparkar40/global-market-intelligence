import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
from io import BytesIO

st.set_page_config(page_title="Rakesh | Global Market Intelligence", page_icon="🌍", layout="wide")

INDEXES = {
"NIFTY 50":("^NSEI","India"), "BANK NIFTY":("^NSEBANK","India"), "SENSEX":("^BSESN","India"),
"S&P 500":("^GSPC","USA"), "NASDAQ 100":("^NDX","USA"), "DOW JONES":("^DJI","USA"),
"FTSE 100":("^FTSE","Europe"), "DAX 40":("^GDAXI","Europe"), "CAC 40":("^FCHI","Europe"),
"IBEX 35":("^IBEX","Europe"), "SWISS MARKET INDEX":("^SSMI","Europe"),
"NIKKEI 225":("^N225","Asia Pacific"), "HANG SENG":("^HSI","Asia Pacific"),
"SHANGHAI COMPOSITE":("000001.SS","Asia Pacific"), "KOSPI":("^KS11","Asia Pacific"),
"TAIWAN WEIGHTED":("^TWII","Asia Pacific"), "STRAITS TIMES":("^STI","Asia Pacific"),
"ASX 200":("^AXJO","Asia Pacific"), "BRAZIL IBOVESPA":("^BVSP","Latin America"),
"MEXICO IPC":("^MXX","Latin America")
}
TRENDS = ["STRONG BULLISH","BULLISH","MIXED / SIDEWAYS","BEARISH","STRONG BEARISH","INSUFFICIENT HISTORY"]

def rsi_calc(s, n=14):
    d=s.diff(); up=d.clip(lower=0); down=-d.clip(upper=0)
    au=up.ewm(alpha=1/n,min_periods=n,adjust=False).mean()
    ad=down.ewm(alpha=1/n,min_periods=n,adjust=False).mean()
    rs=au/ad.replace(0,np.nan); r=100-100/(1+rs)
    r=r.mask((ad==0)&(au>0),100).mask((ad==0)&(au==0),50)
    return r

def ret(s,n):
    if len(s)<=n: return np.nan
    return (float(s.iloc[-1])/float(s.iloc[-n-1])-1)*100

def trend(p,a,b,c):
    if any(pd.isna(x) for x in [a,b,c]): return "INSUFFICIENT HISTORY"
    if p>a>b>c: return "STRONG BULLISH"
    if p<a<b<c: return "STRONG BEARISH"
    if p>b and p>c: return "BULLISH"
    if p<b and p<c: return "BEARISH"
    return "MIXED / SIDEWAYS"

@st.cache_data(ttl=900, show_spinner=False)
def get_data(period):
    rows=[]; hist={}; issues=[]
    for name,(symbol,region) in INDEXES.items():
        try:
            raw=yf.download(symbol,period=period,interval="1d",auto_adjust=True,progress=False,threads=False,timeout=25)
            if raw.empty: raise ValueError("No data returned")
            s=raw["Close"]
            if isinstance(s,pd.DataFrame): s=s.iloc[:,0]
            s=pd.to_numeric(s,errors="coerce").dropna()
            s=s[~s.index.duplicated(keep="last")].sort_index()
            if len(s)<22: raise ValueError("Insufficient price history")
            if (s<=0).any(): raise ValueError("Invalid non-positive price")
            p=float(s.iloc[-1]); a=s.rolling(20).mean().iloc[-1]; b=s.rolling(50).mean().iloc[-1]
            c=s.rolling(200).mean().iloc[-1] if len(s)>=200 else np.nan
            rv=rsi_calc(s).iloc[-1]; rv=float(rv) if pd.notna(rv) else np.nan
            d=ret(s,1); dt=pd.Timestamp(s.index[-1]).date(); age=(date.today()-dt).days
            mom="UNAVAILABLE" if pd.isna(rv) else ("OVERBOUGHT" if rv>=70 else "OVERSOLD" if rv<=30 else "NEUTRAL")
            flags=[]
            if pd.notna(d) and abs(d)>=3: flags.append("VERIFY LARGE MOVE")
            if age>7: flags.append("CHECK STALE DATA")
            if len(s)<200: flags.append("LESS THAN 200 SESSIONS")
            rows.append({"Index":name,"Region":region,"Symbol":symbol,"Last Price":p,"Daily %":d,
             "Weekly %":ret(s,5),"Monthly %":ret(s,21),"Quarterly %":ret(s,63),
             "SMA 20":a,"SMA 50":b,"SMA 200":c,"RSI 14":rv,"Trend":trend(p,a,b,c),
             "Momentum":mom,"Data Date":str(dt),"Data Age Days":age,
             "Quality Status":"; ".join(flags) if flags else "NORMAL"})
            hist[name]=s.rename(name)
        except Exception as e: issues.append({"Index":name,"Symbol":symbol,"Issue":str(e)[:180]})
    return pd.DataFrame(rows),hist,pd.DataFrame(issues)

def excel_bytes(data,regional,weekly,monthly,quality,issues,history):
    out=BytesIO()
    with pd.ExcelWriter(out,engine="openpyxl") as w:
        for name,df in {"Global Scanner":data,"Regional Comparison":regional,
         "Weekly Ranking":weekly,"Monthly Ranking":monthly,"Data Quality":quality,
         "Download Issues":issues,"Price History":history}.items():
            df.to_excel(w,sheet_name=name,index=False)
            ws=w.book[name]; ws.freeze_panes="A2"; ws.auto_filter.ref=ws.dimensions
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width=min(max(max(len(str(x.value or "")) for x in col)+2,12),34)
    out.seek(0); return out.getvalue()

st.sidebar.markdown("## 🌍 Rakesh's Market Scanner")
st.sidebar.caption("Global indices • Investors • Traders")
period_label=st.sidebar.selectbox("Historical window",["1 year","2 years"])
period="1y" if period_label=="1 year" else "2y"
regions_all=sorted({v[1] for v in INDEXES.values()})
regions=st.sidebar.multiselect("Regions",regions_all,default=regions_all)
trends=st.sidebar.multiselect("Trend filters",TRENDS,default=TRENDS)
search=st.sidebar.text_input("Find an index",placeholder="e.g. NASDAQ")
if st.sidebar.button("↻ Refresh data now",use_container_width=True): get_data.clear()

st.title("🌍 Rakesh Parkar | Global Market Intelligence")
st.caption("A global market research workspace for investors and traders")
st.info("Data source: Yahoo Finance through the unofficial yfinance library. This is historical/end-of-day data, not a guaranteed real-time feed. Verify significant moves with exchange or licensed data. Research tool only; not investment advice.")

with st.spinner("Loading global index data..."): data,hist,issues=get_data(period)
if data.empty:
    st.error("No market data retrieved. Try refreshing later."); st.stop()

view=data[data.Region.isin(regions)&data.Trend.isin(trends)].copy()
if search.strip(): view=view[view.Index.str.contains(search.strip(),case=False,na=False)]
bull=int(data.Trend.isin(["STRONG BULLISH","BULLISH"]).sum())
bear=int(data.Trend.isin(["STRONG BEARISH","BEARISH"]).sum())
adv=int((data["Daily %"]>0).sum()); dec=int((data["Daily %"]<0).sum())
best=data.loc[data["Daily %"].idxmax()]; worst=data.loc[data["Daily %"].idxmin()]
obs=max(data["Data Date"])
st.markdown("### Global market snapshot")
cols=st.columns(5)
cols[0].metric("Indices retrieved",f"{len(data)} / {len(INDEXES)}")
cols[1].metric("Bullish trend",bull); cols[2].metric("Bearish trend",bear)
cols[3].metric("Best daily",best.Index,f"{best['Daily %']:+.2f}%")
cols[4].metric("Weakest daily",worst.Index,f"{worst['Daily %']:+.2f}%")
st.caption(f"Latest observation date: {obs} | Data cache: 15 minutes")

tabs=st.tabs(["Market Overview","Global Heatmap","Regional View","Bullish & Bearish","Index Deep Dive","Data Quality","Research Export"])
with tabs[0]:
    st.subheader("Global index performance")
    sort=st.selectbox("Rank table by",["Daily %","Weekly %","Monthly %","Quarterly %","RSI 14","Index"])
    t=view.sort_values(sort,ascending=(sort=="Index"))
    cols_show=["Index","Region","Last Price","Daily %","Weekly %","Monthly %","Quarterly %","RSI 14","Trend","Momentum","Data Date","Quality Status"]
    st.dataframe(t[cols_show].style.format({"Last Price":"{:,.2f}","Daily %":"{:+.2f}%","Weekly %":"{:+.2f}%","Monthly %":"{:+.2f}%","Quarterly %":"{:+.2f}%","RSI 14":"{:.1f}"},na_rep="—"),use_container_width=True,hide_index=True)
    st.subheader("Global market breadth")
    above=[int((data["Last Price"]>data[f"SMA {n}"]).sum()) for n in [20,50,200]]
    b=st.columns(5); b[0].metric("Advancers",adv); b[1].metric("Decliners",dec)
    for i,n in enumerate([20,50,200]): b[i+2].metric(f"Above SMA {n}",f"{above[i]}/{len(data)}")
    breadth=pd.DataFrame({"Measure":["Above SMA 20","Above SMA 50","Above SMA 200"],"Indices":above})
    fig=px.bar(breadth,x="Measure",y="Indices",text="Indices",title="Index participation above moving averages")
    fig.update_layout(yaxis_range=[0,len(data)]); st.plotly_chart(fig,use_container_width=True)

with tabs[1]:
    st.subheader("Global market heatmap")
    heat=view.set_index("Index")[["Daily %","Weekly %","Monthly %","Quarterly %"]]
    if not heat.empty:
        fig=px.imshow(heat,text_auto=".2f",aspect="auto",color_continuous_scale="RdYlGn",color_continuous_midpoint=0,labels={"color":"Return %"})
        fig.update_layout(height=max(480,len(heat)*30)); st.plotly_chart(fig,use_container_width=True)
    else: st.warning("No indices match the current filters.")

with tabs[2]:
    st.subheader("Regional comparison")
    regional=data.groupby("Region").agg(Indices=("Index","count"),Median_Daily=("Daily %","median"),Median_Weekly=("Weekly %","median"),Median_Monthly=("Monthly %","median"),Bullish=("Trend",lambda x:x.isin(["BULLISH","STRONG BULLISH"]).sum()),Bearish=("Trend",lambda x:x.isin(["BEARISH","STRONG BEARISH"]).sum())).reset_index()
    st.dataframe(regional.style.format({"Median_Daily":"{:+.2f}%","Median_Weekly":"{:+.2f}%","Median_Monthly":"{:+.2f}%"}),use_container_width=True,hide_index=True)
    fig=px.bar(regional.sort_values("Median_Weekly"),x="Median_Weekly",y="Region",orientation="h",color="Median_Weekly",color_continuous_scale="RdYlGn",title="Regional median weekly return")
    fig.add_vline(x=0,line_dash="dash",line_color="gray"); fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig,use_container_width=True)

with tabs[3]:
    st.subheader("Strongest and weakest indices")
    left,right=st.columns(2)
    with left:
        st.markdown("#### Strongest weekly")
        st.dataframe(data.sort_values("Weekly %",ascending=False)[["Index","Region","Weekly %","Monthly %","Trend"]].head(10),use_container_width=True,hide_index=True)
        st.markdown("#### Strongest monthly")
        st.dataframe(data.sort_values("Monthly %",ascending=False)[["Index","Region","Monthly %","Trend"]].head(10),use_container_width=True,hide_index=True)
    with right:
        st.markdown("#### Weakest weekly")
        st.dataframe(data.sort_values("Weekly %")[["Index","Region","Weekly %","Monthly %","Trend"]].head(10),use_container_width=True,hide_index=True)
        st.markdown("#### Weakest monthly")
        st.dataframe(data.sort_values("Monthly %")[["Index","Region","Monthly %","Trend"]].head(10),use_container_width=True,hide_index=True)

with tabs[4]:
    st.subheader("Individual index analysis")
    name=st.selectbox("Select an index",sorted(hist))
    s=hist[name]; row=data[data.Index==name].iloc[0]
    k=st.columns(4); k[0].metric("Last price",f"{row['Last Price']:,.2f}"); k[1].metric("Daily return",f"{row['Daily %']:+.2f}%")
    k[2].metric("RSI 14","—" if pd.isna(row["RSI 14"]) else f"{row['RSI 14']:.1f}"); k[3].metric("Trend",row.Trend)
    chart=pd.DataFrame({"Close":s})
    for n in [20,50,200]: chart[f"SMA {n}"]=chart.Close.rolling(n).mean()
    fig=go.Figure()
    for c in ["Close","SMA 20","SMA 50","SMA 200"]: fig.add_trace(go.Scatter(x=chart.index,y=chart[c],name=c))
    fig.update_layout(title=f"{name}: closing price and moving averages",height=480,hovermode="x unified")
    st.plotly_chart(fig,use_container_width=True)
    rr=rsi_calc(s); rf=go.Figure(go.Scatter(x=rr.index,y=rr,name="RSI 14"))
    rf.add_hline(y=70,line_dash="dash",line_color="red"); rf.add_hline(y=30,line_dash="dash",line_color="green")
    rf.update_layout(title="Relative Strength Index (14)",height=280,yaxis_range=[0,100]); st.plotly_chart(rf,use_container_width=True)

with tabs[5]:
    st.subheader("Data quality monitoring")
    quality=data[["Index","Symbol","Data Date","Data Age Days","Daily %","Quality Status"]]
    st.dataframe(quality,use_container_width=True,hide_index=True)
    flagged=quality[quality["Quality Status"]!="NORMAL"]
    if not flagged.empty:
        st.warning("Review flagged rows manually. A flag is not proof of an incorrect quote.")
        st.dataframe(flagged,use_container_width=True,hide_index=True)
    else: st.success("No automatic flags. This is not independent verification.")
    if not issues.empty: st.dataframe(issues,use_container_width=True,hide_index=True)
    else: st.success("No download errors were returned.")

with tabs[6]:
    st.subheader("Download complete research workbook")
    regional=data.groupby("Region").agg(Indices=("Index","count"),Median_Daily=("Daily %","median"),Median_Weekly=("Weekly %","median"),Median_Monthly=("Monthly %","median"),Bullish=("Trend",lambda x:x.isin(["BULLISH","STRONG BULLISH"]).sum()),Bearish=("Trend",lambda x:x.isin(["BEARISH","STRONG BEARISH"]).sum())).reset_index()
    weekly=data.sort_values("Weekly %",ascending=False).reset_index(drop=True); weekly.insert(0,"Weekly Rank",range(1,len(weekly)+1))
    monthly=data.sort_values("Monthly %",ascending=False).reset_index(drop=True); monthly.insert(0,"Monthly Rank",range(1,len(monthly)+1))
    quality=data[["Index","Symbol","Data Date","Data Age Days","Daily %","Quality Status"]]
    history=pd.concat(hist,axis=1) if hist else pd.DataFrame()
    out=BytesIO()
    with pd.ExcelWriter(out,engine="openpyxl") as w:
        for sheet,frame in {"Global Scanner":data,"Regional Comparison":regional,"Weekly Ranking":weekly,"Monthly Ranking":monthly,"Data Quality":quality,"Download Issues":issues,"Price History":history.reset_index()}.items():
            frame.to_excel(w,sheet_name=sheet,index=False)
            ws=w.book[sheet]; ws.freeze_panes="A2"; ws.auto_filter.ref=ws.dimensions
            for col in ws.columns: ws.column_dimensions[col[0].column_letter].width=min(max(max(len(str(x.value or "")) for x in col)+2,12),34)
    st.download_button("⬇ Download complete Excel research report",out.getvalue(),file_name=f"Rakesh_Global_Market_Intelligence_{date.today().isoformat()}.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",use_container_width=True)
    st.caption("Seven sheets: scanner, regional comparison, weekly ranking, monthly ranking, data quality, download issues and price history.")

st.divider()
st.subheader("Daily market pulse")
st.write(f"Across {len(data)} retrieved indices, {adv} advanced and {dec} declined. {bull} meet the bullish trend rules and {bear} meet the bearish rules. {above[2]} of {len(data)} indices are above their 200-session moving average. Best daily performer: {best.Index} ({best['Daily %']:+.2f}%). Weakest: {worst.Index} ({worst['Daily %']:+.2f}%).")
st.caption("Rakesh Parkar | Global Market Intelligence V2.1 • Rule-based analytics, not AI-generated advice. Yahoo Finance/yfinance coverage and data quality are not guaranteed.")
