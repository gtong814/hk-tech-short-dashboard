import time
from datetime import datetime, timedelta
from io import StringIO
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 頁面設定與觀察名單
# ==========================================
st.set_page_config(page_title="港股科技股：沽空與淡倉雙重確認儀表板", layout="wide")

BENCHMARK_TICKER = "3033.HK"

DEFAULT_WATCHLIST = {
    "3033.HK": ("03033", "南方恒生科技ETF(基準)"),
    "1810.HK": ("01810", "小米集團-W"),
    "0700.HK": ("00700", "騰訊控股"),
    "9988.HK": ("09988", "阿里巴巴-W"),
    "3690.HK": ("03690", "美團-W"),
    "9618.HK": ("09618", "京東集團-SW"),
    "1024.HK": ("01024", "快手-W"),
    "9888.HK": ("09888", "百度集團-SW"),
    "0981.HK": ("00981", "中芯國際"),
    "1347.HK": ("01347", "華虹半導體"),
    "0992.HK": ("00992", "聯想集團"),
    "0020.HK": ("00020", "商湯-W"),
    "9660.HK": ("09660", "地平線機器人-W"),
    "9999.HK": ("09999", "網易-S"),
    "9626.HK": ("09626", "嗶哩嗶哩-W"),
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-HK,zh-TW;q=0.9,zh;q=0.8,en-US;q=0.7,en;q=0.6",
    "Referer": "https://www.aastocks.com/",
}

# ==========================================
# 1. 直連抓取證監會 (SFC) 每周未平倉淡倉 CSV
# ==========================================
@st.cache_data(ttl=3600)
def fetch_sfc_weekly_short_positions(num_weeks=6):
    """
    直接按日期推算 SFC 官方 CSV 網址 (避開網頁 HTML 爬蟲攔截)。
    格式: https://www.sfc.hk/-/media/EN/pdf/spr/YYYY/MM/DD/Short_Position_Reporting_Aggregated_Data_YYYYMMDD.csv
    """
    dfs = []
    today = datetime.now()
    # 找出最近一個星期五
    days_since_friday = (today.weekday() - 4) % 7
    last_friday = today - timedelta(days=days_since_friday)

    checked_weeks = 0
    week_offset = 0

    while len(dfs) < num_weeks and week_offset < num_weeks + 6:
        target_fri = last_friday - timedelta(weeks=week_offset)
        week_offset += 1

        # 嘗試星期五，若遇香港公眾假期則順延檢查星期四、星期三
        for day_shift in [0, 1, 2]:
            d = target_fri - timedelta(days=day_shift)
            url = (
                f"https://www.sfc.hk/-/media/EN/pdf/spr/"
                f"{d.strftime('%Y')}/{d.strftime('%m')}/{d.strftime('%d')}/"
                f"Short_Position_Reporting_Aggregated_Data_{d.strftime('%Y%m%d')}.csv"
            )
            try:
                r = requests.get(url, headers={"User-Agent": HEADERS["User-Agent"]}, timeout=10)
                if r.status_code == 200 and "Stock Code" in r.text:
                    df = pd.read_csv(StringIO(r.text))
                    df.columns = [c.strip() for c in df.columns]
                    df["Stock_Code_5"] = df["Stock Code"].astype(str).str.strip().str.zfill(5)
                    df["Report_Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

                    shares_col = [c for c in df.columns if "Shares" in c][0]
                    hkd_col = [c for c in df.columns if "HK$" in c][0]
                    df["Short_Shares"] = pd.to_numeric(df[shares_col].astype(str).str.replace(",", ""), errors="coerce")
                    df["Short_HKD"] = pd.to_numeric(df[hkd_col].astype(str).str.replace(",", ""), errors="coerce")

                    dfs.append(df[["Report_Date", "Stock_Code_5", "Stock Name", "Short_Shares", "Short_HKD"]])
                    break
            except Exception:
                continue

    if not dfs:
        return pd.DataFrame(), []

    sfc_all = pd.concat(dfs, ignore_index=True).dropna(subset=["Report_Date"])
    dates = sorted(sfc_all["Report_Date"].unique(), reverse=True)
    return sfc_all, dates

# ==========================================
# 2. 抓取 AAStocks 單股每日沽空數據 (HTTPS + 容錯)
# ==========================================
@st.cache_data(ttl=1800)
def fetch_aastocks_short(aa_symbol):
    url = f"https://www.aastocks.com/tc/stocks/analysis/stock-short-selling-ratio.aspx?symbol={aa_symbol}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=12)
        if resp.status_code != 200:
            return pd.DataFrame(), f"HTTP {resp.status_code}"

        tables = pd.read_html(StringIO(resp.text))
        short_df = None
        for df in tables:
            cols_str = " ".join([str(c) for c in df.columns])
            if "日期" in cols_str and "沽空比率" in cols_str:
                short_df = df.copy()
                break

        if short_df is None:
            return pd.DataFrame(), "網頁結構變更或被驗證攔截"

        short_df = short_df.iloc[:, :6]
        short_df.columns = ["Date", "Close_AA", "Short_Amt", "Total_Amt", "Short_Ratio", "Short_Avg_Price"]
        short_df["Date"] = pd.to_datetime(short_df["Date"], format="%Y/%m/%d", errors="coerce")
        short_df = short_df.dropna(subset=["Date"]).set_index("Date").sort_index()
        short_df["Short_Ratio"] = pd.to_numeric(
            short_df["Short_Ratio"].astype(str).str.replace("%", "").str.replace(",", ""),
            errors="coerce"
        )
        return short_df[["Short_Ratio", "Short_Amt", "Total_Amt"]], "OK"
    except Exception as e:
        return pd.DataFrame(), str(e)[:60]

# ==========================================
# 3. 批量掃描核心邏輯 (支援降級運作，永不崩潰)
# ==========================================
@st.cache_data(ttl=1800)
def run_full_scan(watchlist, sfc_df, sfc_dates):
    tickers = list(watchlist.keys())
    raw_prices = yf.download(tickers, period="3mo", group_by="ticker", auto_adjust=False, progress=False)

    bm_df = raw_prices[BENCHMARK_TICKER].dropna(subset=["Close"]).copy()
    bm_df["MA20"] = bm_df["Close"].rolling(20).mean()
    bm_close = float(bm_df["Close"].iloc[-1])
    bm_ma20 = float(bm_df["MA20"].iloc[-1])
    bm_5d_chg = float(bm_df["Close"].pct_change(5).iloc[-1] * 100)
    bm_above_ma20 = bm_close >= bm_ma20

    summary_rows = []
    history_dict = {}
    aa_status_log = []

    latest_sfc_date = sfc_dates[0] if len(sfc_dates) >= 1 else None
    prev_sfc_date = sfc_dates[1] if len(sfc_dates) >= 2 else None

    for yf_ticker, (aa_symbol, name) in watchlist.items():
        try:
            price_df = raw_prices[yf_ticker].dropna(subset=["Close"]).copy()
            if price_df.empty or len(price_df) < 10:
                continue
            price_df.index = pd.to_datetime(price_df.index).tz_localize(None)

            # 嘗試抓取每日沽空，若失敗仍保留股價數據
            short_df, aa_msg = fetch_aastocks_short(aa_symbol)
            if aa_msg != "OK":
                aa_status_log.append(f"{yf_ticker}: {aa_msg}")

            if not short_df.empty:
                df = price_df.join(short_df, how="left")
            else:
                df = price_df.copy()
                df["Short_Ratio"] = np.nan

            # 計算技術與沽空指標
            df["Vol_MA20"] = df["Volume"].rolling(20, min_periods=5).mean()
            df["Vol_Ratio"] = df["Volume"] / df["Vol_MA20"]
            df["SR_MA10"] = df["Short_Ratio"].rolling(10, min_periods=3).mean()
            df["SR_Std10"] = df["Short_Ratio"].rolling(10, min_periods=3).std()
            df["SR_ZScore"] = (df["Short_Ratio"] - df["SR_MA10"]) / df["SR_Std10"].replace(0, np.nan)
            df["Price_Chg_5D"] = df["Close"].pct_change(5) * 100
            history_dict[yf_ticker] = df

            latest = df.iloc[-1]
            sr = latest["Short_Ratio"]
            sr_z = latest["SR_ZScore"] if pd.notna(latest["SR_ZScore"]) else 0.0
            vol_r = float(latest["Vol_Ratio"])
            chg_5d = float(latest["Price_Chg_5D"])
            rs_vs_bm = chg_5d - bm_5d_chg
            is_green = latest["Close"] >= latest["Open"]

            # 匹配 SFC 未平倉淡倉數據
            sfc_hkd_m = np.nan
            sfc_wow_pct = np.nan
            if not sfc_df.empty and latest_sfc_date is not None:
                stock_sfc = sfc_df[sfc_df["Stock_Code_5"] == aa_symbol]
                cur_row = stock_sfc[stock_sfc["Report_Date"] == latest_sfc_date]
                if not cur_row.empty:
                    cur_shares = cur_row["Short_Shares"].iloc[0]
                    sfc_hkd_m = cur_row["Short_HKD"].iloc[0] / 1e6
                    if prev_sfc_date is not None:
                        prev_row = stock_sfc[stock_sfc["Report_Date"] == prev_sfc_date]
                        if not prev_row.empty and prev_row["Short_Shares"].iloc[0] > 0:
                            prev_shares = prev_row["Short_Shares"].iloc[0]
                            sfc_wow_pct = (cur_shares - prev_shares) / prev_shares * 100

            # 雙重確認訊號判定
            if yf_ticker == BENCHMARK_TICKER:
                signal = "⭐ 板塊基準 (3033)"
            elif pd.isna(sr):
                if vol_r >= 1.4 and is_green and rs_vs_bm > 0:
                    signal = "📈 放量強勢 (無日沽空)"
                else:
                    signal = "⚪ 正常震盪 (無日沽空)"
            elif vol_r >= 1.4 and is_green and (sr <= 25 or sr_z <= -0.8):
                if pd.notna(sfc_wow_pct) and sfc_wow_pct < -2.0:
                    signal = "🚀 真逼空 (淡倉實質減倉)"
                else:
                    signal = "🔥 放量稀釋逼空"
            elif (sr >= 38 or sr_z >= 1.2) and is_green:
                signal = "🟢 空頭擁擠築底"
            elif vol_r <= 0.85 and (sr >= 32 or sr_z >= 0.8) and not is_green:
                if pd.notna(sfc_wow_pct) and sfc_wow_pct > 2.0:
                    signal = "⚠️ 雙重壓盤 (淡倉持續增倉)"
                else:
                    signal = "🔴 縮量壓盤警戒"
            else:
                signal = "⚪ 正常震盪"

            summary_rows.append({
                "代號": yf_ticker,
                "名稱": name,
                "收市價": round(float(latest["Close"]), 2),
                "5日漲跌(%)": round(chg_5d, 2),
                "相對3033(%)": round(rs_vs_bm, 2),
                "量比": round(vol_r, 2),
                "日沽空(%)": round(float(sr), 2) if pd.notna(sr) else None,
                "沽空Z-Score": round(float(sr_z), 2) if pd.notna(sr) else 0.0,
                "SFC淡倉(百萬HKD)": round(float(sfc_hkd_m), 1) if pd.notna(sfc_hkd_m) else None,
                "SFC淡倉周變動(%)": round(float(sfc_wow_pct), 2) if pd.notna(sfc_wow_pct) else None,
                "雙重確認訊號": signal
            })
            time.sleep(0.3)
        except Exception:
            continue

    res_df = pd.DataFrame(summary_rows)
    if not res_df.empty:
        bm_row = res_df[res_df["代號"] == BENCHMARK_TICKER]
        others = res_df[res_df["代號"] != BENCHMARK_TICKER].sort_values(by="相對3033(%)", ascending=False)
        res_df = pd.concat([bm_row, others], ignore_index=True)

    bm_info = {
        "close": bm_close, "ma20": bm_ma20,
        "chg_5d": bm_5d_chg, "above_ma20": bm_above_ma20
    }
    return res_df, history_dict, bm_info, aa_status_log

# ==========================================
# 4. Streamlit 介面佈局
# ==========================================
st.title("📊 港股科技股：日線沽空流量 × SFC 周度淡倉 雙重確認儀表板")

with st.sidebar:
    st.header("⚙️ 參數與控制")
    sfc_weeks = st.slider("抓取 SFC 歷史淡倉周數", min_value=4, max_value=12, value=6)
    if st.button("🔄 強制重新抓取最新數據", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with st.spinner("正在從 SFC 官網直連抓取每周未平倉淡倉 CSV..."):
    sfc_df, sfc_dates = fetch_sfc_weekly_short_positions(num_weeks=sfc_weeks)

with st.spinner("正在批量掃描港股科技股價量與每日沽空數據..."):
    scan_df, history_dict, bm_info, aa_status_log = run_full_scan(DEFAULT_WATCHLIST, sfc_df, sfc_dates)

# 在側邊欄顯示數據源連線診斷
with st.sidebar:
    st.markdown("---")
    st.subheader("🔌 數據源連線狀態")
    st.write(f"- **yfinance 股價**：✅ 正常 ({len(history_dict)} 隻)")
    st.write(f"- **SFC 淡倉 CSV**：{'✅ 正常' if len(sfc_dates) > 0 else '❌ 被攔截'} ({len(sfc_dates)} 周)")
    if len(aa_status_log) == 0:
        st.write("- **AAStocks 日沽空**：✅ 正常")
    elif len(aa_status_log) < len(DEFAULT_WATCHLIST):
        st.write(f"- **AAStocks 日沽空**：⚠️ 部分失敗 ({len(aa_status_log)} 隻)")
    else:
        st.write("- **AAStocks 日沽空**：❌ 雲端 IP 被攔截（已自動降級為價量+SFC模式）")

# 頂部大市濾網 KPI 橫幅
c1, c2, c3, c4 = st.columns(4)
regime_str = "🟢 偏多 (站穩 MA20)" if bm_info["above_ma20"] else "🔴 偏弱 (受制 MA20)"
c1.metric("大市濾網 (3033.HK)", f"{bm_info['close']:.2f} HKD", f"{bm_info['chg_5d']:+.2f}% (5日)")
c2.metric("3033.HK 趨勢狀態", regime_str, f"MA20: {bm_info['ma20']:.2f}")
latest_sfc_str = pd.to_datetime(sfc_dates[0]).strftime("%Y-%m-%d") if sfc_dates else "暫無數據"
c3.metric("SFC 最新淡倉報告日", latest_sfc_str, f"已載入 {len(sfc_dates)} 周數據")
squeeze_cnt = scan_df["雙重確認訊號"].str.contains("逼空|強勢").sum() if not scan_df.empty else 0
c4.metric("今日強勢/逼空訊號數", f"{squeeze_cnt} 隻")

# 分頁設計
tab1, tab2 = st.tabs(["🌐 板塊四象限掃描與雙重確認總表", "🔍 單股深度診斷 (日線沽空 + SFC 淡倉趨勢)"])

with tab1:
    if scan_df.empty:
        st.warning("暫時無法取得掃描數據，請稍後重試。")
    else:
        has_short_ratio = scan_df["日沽空(%)"].notna().any()
        x_col = "沽空Z-Score" if has_short_ratio else "相對3033(%)"
        chart_title = (
            "香港科技股 vs 3033.HK：沽空偏離度 (Z-Score) 與量比四象限"
            if has_short_ratio
            else "香港科技股 vs 3033.HK：相對強弱 (RS%) 與量比四象限 (日沽空降級模式)"
        )
        plot_df = scan_df.copy()
        plot_df["點大小"] = plot_df["日沽空(%)"].fillna(20)

        fig_quad = px.scatter(
            plot_df,
            x=x_col,
            y="量比",
            size="點大小",
            color="雙重確認訊號",
            text="名稱",
            hover_data=["代號", "收市價", "相對3033(%)", "日沽空(%)", "SFC淡倉(百萬HKD)", "SFC淡倉周變動(%)"],
            template="plotly_dark",
            height=560,
            title=chart_title
        )
        fig_quad.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.6)
        fig_quad.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.6)
        fig_quad.update_traces(textposition="top center", textfont=dict(size=12))
        st.plotly_chart(fig_quad, use_container_width=True)

        st.subheader("📋 板塊雙重確認數據總表")
        st.dataframe(scan_df, use_container_width=True, hide_index=True)

with tab2:
    stock_options = [f"{k} - {v[1]}" for k, v in DEFAULT_WATCHLIST.items() if k in history_dict]
    if not stock_options:
        st.warning("目前沒有可供分析的個股歷史數據，請點擊側邊欄重新抓取。")
    else:
        selected_label = st.selectbox("選擇要深度分析的科技股", stock_options, index=1 if len(stock_options) > 1 else 0)
        sel_ticker = selected_label.split(" - ")[0]
        sel_aa_code, sel_name = DEFAULT_WATCHLIST[sel_ticker]

        df_single = history_dict[sel_ticker]

        # 圖 1：日線 K線 + 每日沽空比率 (如有) + 量比
        fig_daily = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            row_heights=[0.68, 0.32], specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )
        fig_daily.add_trace(
            go.Candlestick(x=df_single.index, open=df_single["Open"], high=df_single["High"],
                           low=df_single["Low"], close=df_single["Close"], name="股價 (HKD)"),
            row=1, col=1, secondary_y=False
        )
        if df_single["Short_Ratio"].notna().any():
            fig_daily.add_trace(
                go.Scatter(x=df_single.index, y=df_single["Short_Ratio"], name="日沽空比率 (%)",
                           line=dict(color="orange", width=2, dash="dot")),
                row=1, col=1, secondary_y=True
            )
        vol_colors = [
            "#FF5252" if (pd.notna(r) and r >= 35) else "rgba(100, 149, 237, 0.65)"
            for r in df_single["Short_Ratio"]
        ]
        fig_daily.add_trace(
            go.Bar(x=df_single.index, y=df_single["Vol_Ratio"], name="量比 (vs 20D)", marker_color=vol_colors),
            row=2, col=1
        )
        fig_daily.add_hline(y=1.5, line_dash="dash", line_color="cyan", row=2, col=1)
        fig_daily.update_layout(
            title=f"{sel_name} ({sel_ticker})：日線股價與量比分析",
            template="plotly_dark", height=520, xaxis_rangeslider_visible=False, hovermode="x unified"
        )
        st.plotly_chart(fig_daily, use_container_width=True)

        # 圖 2：SFC 每周未平倉淡倉趨勢
        if not sfc_df.empty:
            stock_sfc_hist = sfc_df[sfc_df["Stock_Code_5"] == sel_aa_code].sort_values("Report_Date")
            if not stock_sfc_hist.empty:
                fig_sfc = make_subplots(specs=[[{"secondary_y": True}]])
                fig_sfc.add_trace(
                    go.Bar(x=stock_sfc_hist["Report_Date"], y=stock_sfc_hist["Short_Shares"] / 1e6,
                           name="未平倉淡倉股數 (百萬股)", marker_color="#7C4DFF", opacity=0.75),
                    secondary_y=False
                )
                fig_sfc.add_trace(
                    go.Scatter(x=stock_sfc_hist["Report_Date"], y=stock_sfc_hist["Short_HKD"] / 1e6,
                               name="淡倉市值 (百萬港元)", line=dict(color="#FFD600", width=2.5)),
                    secondary_y=True
                )
                fig_sfc.update_layout(
                    title=f"{sel_name} ({sel_ticker})：SFC 每周未平倉淡倉總餘額趨勢",
                    template="plotly_dark", height=360, hovermode="x unified"
                )
                fig_sfc.update_yaxes(title_text="淡倉股數 (百萬股)", secondary_y=False)
                fig_sfc.update_yaxes(title_text="淡倉市值 (百萬港元)", secondary_y=True)
                st.plotly_chart(fig_sfc, use_container_width=True)
            else:
                st.info(f"SFC 最新淡倉申報清單中未包含 {sel_name} ({sel_ticker})。")
