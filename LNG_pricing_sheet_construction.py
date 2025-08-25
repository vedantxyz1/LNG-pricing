### Libraries we require (ensure its all at the top)
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from time import sleep
import re
import numpy as np
import requests
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import smtplib
from email.message import EmailMessage
import os

# Month code to name mapping used in futures contracts (standardisation)
month_map = {
    "F": "Jan", "G": "Feb", "H": "Mar", "J": "Apr", "K": "May", "M": "Jun",
    "N": "Jul", "Q": "Aug", "U": "Sep", "V": "Oct", "X": "Nov", "Z": "Dec"
}
month_codes = list(month_map)  # ['F', 'G', ..., 'Z']

# Parse symbols
# default settings
# Generate futures contract symbols for a given base code from start_date forward.
def parse_barchart_symbols(base_code: str, start_date: datetime, years_forward: int):

    # Args:
    #     base_code (str): e.g., "NG", "QA", "INK"
    #     start_date (datetime): Starting date (usually today)
    #     years_forward (int): Number of years into the future to pull

    #Returns: List of contract symbols like ["NGU25", "NGV25", ...]
    symbols = []
    current = start_date.replace(day=1)
    end_date = current + relativedelta(years=years_forward)
    end_year = end_date.year

    while current.year < end_year:
        y = str(current.year)[-2:]
        m_code = list(month_map.keys())[current.month - 1]
        symbols.append(f"{base_code}{m_code}{y}")
        current += relativedelta(months=1)

    # Add the **full final year** (Jan to Dec)
    for m_idx, m_code in enumerate(month_map.keys()):
        symbols.append(f"{base_code}{m_code}{str(end_year)[-2:]}")

    return symbols

## Fetch futures prices from Barchart symbol pages.
def fetch_barchart_prices(symbols, base_url="https://www.barchart.com/futures/quotes/{}/overview"):
    # Args:
    #     symbols (list): List of contract symbols like ["NGU25", "NGV25"]
    #     base_url (str): URL format string with `{}` for symbol insertion

    # Returns:
    #     pd.DataFrame: Cleaned dataframe with columns ["symbol", "last_price"]
    headers = {"User-Agent": "Mozilla/5.0"}
    records = []

    for symbol in symbols:
        url = base_url.format(symbol)
        try:
            # Throttle request rate (Barchart rate-limits aggressively)
            sleep(0.5) # remove if too slow

            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Find the container with init() JSON
            div = soup.find("div", class_="symbol-header-info")
            if not div:
                print(f"[WARN] Missing div for {symbol}")
                continue

            ng_init = div.get("data-ng-init")
            match = re.search(r'init\((\{.*?\})\)', ng_init)
            if not match:
                print(f"[WARN] No JSON init block found for {symbol}")
                continue

            data = json.loads(match.group(1).replace(r'\/', '/'))
            raw_price = data.get("lastPrice")

            if raw_price in [None, "N/A", "-"]:
                print(f"[SKIP] Invalid price for {symbol}")
                continue

            clean_price = re.sub(r"[^\d.]+$", "", str(raw_price))
            records.append({
                "symbol": symbol,
                "last_price": round(float(clean_price), 4) if clean_price else None
            })

        except Exception as e:
            print(f"[ERROR] Failed to fetch {symbol}: {e}")
            continue

    if not records:
        print("[ERROR] No valid prices fetched.")
        return pd.DataFrame(columns=["symbol", "last_price"])

    return pd.DataFrame(records)

## Abit of housekeeping/cleaning at the end to tie it all together
def clean_and_format_df(df_raw, label):
    df = df_raw.copy()

    # Extract month code and map to month name
    df["month_code"] = df["symbol"].str.extract(r'[A-Z]{2,3}([FGHJKMNQUVXZ])\d{2}')
    df["month"] = df["month_code"].map(month_map)

    # Extract year
    df["year"] = 2000 + df["symbol"].str.extract(r'(\d{2})$').astype(int)

    # Dynamically map current month and year for cash contracts
    cash_mask = df["symbol"].str.endswith("Y00")
    if cash_mask.any():
        df.loc[cash_mask, "month"] = datetime.today().strftime("%b")
        df.loc[cash_mask, "year"] = datetime.today().year

    # Label construction: Cash for Y00 contracts, Month'YY otherwise
    df["label"] = np.where(
        cash_mask,
        "Cash",
        df["month"].str[:3] + "'" + df["year"].astype(str).str[-2:]
    )
    
    df = df.rename(columns={"last_price": label})
    return df[["label", label]].reset_index(drop=True)

### Part to start pulling/scraping the data from barchart.com
today = datetime.today()
start_date = today

# Pull configuration for each commodity
commodity_config = {
    'NG': {'label': "Henry Hub ($/MMBtu)", 'base_code': 'NG', 'years': 12, 'has_cash': True, 'cash_symbol': 'NGY00'},
    'QA': {'label': "Brent ($/bbl)", 'base_code': 'QA', 'years': 8, 'has_cash': True, 'cash_symbol': 'QAY00'},
    'INK': {'label': "TTF ($/MMBtu)", 'base_code': 'INK', 'years': 3, 'has_cash': False},
    'JKM': {'label': "JKM ($/MMBtu)", 'base_code': 'JKM', 'years': 5, 'has_cash': False},
    'NF': {'label': "NBP (p/th)", 'base_code': 'NF', 'years': 7, 'has_cash': False}
}

# Container to store clean DFs
commodity_data = {}

for code, cfg in commodity_config.items():
    symbols = []

    # Add cash contract if exists
    if cfg.get('has_cash'):
        symbols.append(cfg['cash_symbol'])

    # Add futures symbols
    symbols += parse_barchart_symbols(
        base_code=cfg['base_code'],
        start_date=start_date,
        years_forward=cfg['years']
    )

    # Fetch prices
    raw_df = fetch_barchart_prices(symbols)

    # Clean and format
    clean_df = clean_and_format_df(raw_df, cfg['label'])

    # Drop current front-month label dynamcally
    front_month_label = today.strftime("%b'%y")
    clean_df = clean_df[clean_df["label"] != front_month_label]

    commodity_data[code] = clean_df

# cross checker - ensures the right tickers come out
#for code, df in commodity_data.items():
#    if not df.empty:
#        print(f"[INFO] {code} curve ends at:", df['label'].iloc[-1])

## Individual flat price dataframes (to use later for the spread calculations)
df_ng = commodity_data["NG"] # Henry df 
df_qa = commodity_data["QA"] # Brent df
df_ink = commodity_data["INK"] # TTF df
df_jkm = commodity_data["JKM"] # JKM df
df_nbp = commodity_data["NF"]  # NBP df

## Pulling the cable fwd curve - we require this to obtain more accurate marks on NBP ($/mmbtu) 
# NBP flat prices originally quoted in pence/therm
# Cable fwd curve extraction
def fetch_forward_fx_rates(url, tenor_list):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    data = []

    for tenor in tenor_list:##
        td = soup.find('td', string=lambda s: s and tenor in s)
        if not td:
            print(f"[WARN] {tenor} not found.")
            continue 

        siblings = td.find_next_siblings("td", limit=3)
        if len(siblings) < 3:
            print(f"[WARN] Not enough data for {tenor}.")
            continue

        try:
            bid = float(siblings[0].text.strip())
            ask = float(siblings[1].text.strip())
            mid = float(siblings[2].text.strip())
            data.append({
                "tenor": tenor,
                "bid": round(bid, 4),
                "ask": round(ask, 4),
                "mid": round(mid, 4)
            })
        except Exception as e:
            print(f"[ERROR] FX parsing failed for {tenor}: {e}")
            continue
    
    return pd.DataFrame(data)

# tenors to pull from

tenors = [
    # "One Week", "Two Week", "Three Week", decided to omit the first 3 weeks as it has on average no more than a 2 pip diff to the 1 month contract
    "One Month", "Two Month", "Three Month", "Four Month", "Five Month", "Six Month",
    "Seven Month", "Eight Month", "Nine Month", "Ten Month", "Eleven Month",
    "One Year", "Two Year", "Three Year", "Four Year", "Five Year"
]

fx_url = "https://www.fxempire.com/currencies/gbp-usd/forward-rates"
cable_fwd_data = fetch_forward_fx_rates(fx_url, tenors)
df_fx = cable_fwd_data[["tenor", "mid"]].rename(columns={"mid": "Cable Fwd Rate"})

## Applying the fwd rate curve to the correct portions of the NBP 
# Step 1: Parse label into date components
df_nbp["parsed_date"] = pd.to_datetime(df_nbp["label"], format="%b'%y")
df_nbp["month"] = df_nbp["parsed_date"].dt.month
df_nbp["year"] = df_nbp["parsed_date"].dt.year

# Step 2: Identify front month and year (first row of NBP)
front_year = df_nbp.iloc[0]["year"]
front_month = df_nbp.iloc[0]["month"]

# Step 3: Map each row to correct FX tenor
def map_fx_tenor(row):
    # calculate how many months ahead this NBP contract is from the front (earliest) contract
    # This assumes df_nbp is already sorted by date and the front_year/front_month were extracted from the first row
    month_diff = (row["year"] - front_year) * 12 + (row["month"] - front_month)

    # for contracts within the first 11 months of the curve (0 to 10 months ahead),
    # use month-specific tenors: "One-month", "Two-month", ... up to "Eleven Month"
    if month_diff < 11:
        return df_fx.iloc[month_diff]["tenor"]

    # For contracts 11 - 22 months ahead, use the 1-year forward FX rate
    elif month_diff < 23:
        return "One Year"
    
    # for contracrts 23 - 34 months ahead, use the 2-year FX rate
    elif month_diff < 35:
        return "Two Year"
    
    # for contracts 35 - 46 months ahead, use the 3-year FX rate
    elif month_diff < 47:
        return "Three Year"
    
    # for contracts 47 - 58 months ahead, use the 4-year FX rate
    elif month_diff < 59:
        return "Four Year"
    
    # for contracts 59 months and beyond, use the 5-year FX rate
    else:
        return "Five Year"

df_nbp["FX Tenor"] = df_nbp.apply(map_fx_tenor, axis=1)

# Step 4: Merge FX rates
df_nbp = df_nbp.merge(df_fx, left_on="FX Tenor", right_on="tenor", how="left")

# Step 5: Convert to $/MMBtu
df_nbp["NBP ($/MMBtu)"] = round(df_nbp["NBP (p/th)"] / 10 * df_nbp["Cable Fwd Rate"], 3)

# Step 6: Keeping the relevant columns
df_nbp = df_nbp[["label", "NBP (p/th)", "Cable Fwd Rate", "NBP ($/MMBtu)"]]
commodity_data["NF"] = df_nbp

# Merge all flat price curves on 'label' (left-to-right)
df_merged = (
    df_ng
    .merge(df_qa, on="label", how="outer")
    .merge(df_ink, on="label", how="outer")
    .merge(df_jkm, on="label", how="outer")
    .merge(df_nbp, on="label", how="outer")  # already includes Cable + $/MMBtu
)

# Chronological sort — keep 'Cash' first if present
df_merged["sort_key"] = df_merged["label"].apply(
    lambda x: pd.Timestamp("1900-01-01") if x == "Cash" else pd.to_datetime(x, format="%b'%y", errors="coerce")
)

df_merged = (
    df_merged
    .sort_values("sort_key")
    .drop(columns="sort_key")
    .reset_index(drop=True)
)

## Brent conversion to MMbtu from bbls and computing the geographical arbs
# Options to include certain spreads refer below

# Convert Brent to $/MMBtu
df_merged["Brent ($/MMBtu)"] = df_merged["Brent ($/bbl)"] / 5.8

# Geographical Spreads in $/MMBtu
df_merged["JKM v TTF"] = df_merged["JKM ($/MMBtu)"] - df_merged["TTF ($/MMBtu)"]

# Optional spreads: uncomment (#) if needed
# df_merged["HH v TTF"] = df_merged["Henry Hub ($/MMBtu)"] - df_merged["TTF ($/MMBtu)"]
# df_merged["JKM v HH"] = df_merged["JKM ($/MMBtu)"] - df_merged["Henry Hub ($/MMBtu)"]
# df_merged["JKM v NBP"] = df_merged["JKM ($/MMBtu)"] - df_merged["NBP ($/MMBtu)"]
# df_merged["TTF v NBP"] = df_merged["TTF ($/MMBtu)"] - df_merged["NBP ($/MMBtu)"]

df_merged = df_merged[[  
    "label",
    "JKM ($/MMBtu)",
    "TTF ($/MMBtu)",
    "NBP ($/MMBtu)",
    "Henry Hub ($/MMBtu)",
    "JKM v TTF",                   # core geographical spread

    # Optional spreads for Miguel: Uncomment to include in pricing Sheet
    # Check to ensure the "Optional Spread" computations above are uncommented as well  
    # "HH v TTF",                   # Henry Hub v TTF
    # "JKM v HH",                   # JKM v Henry Hub
    # "JKM v NBP",                  # JKM v NBP
    # "TTF v NBP",                  # TTF v NBP
    # "Brent ($/MMBtu)",           # crude oil in $/MMBtu

    "Brent ($/bbl)",              # Brent in original format
    "NBP (p/th)",
    "Cable Fwd Rate"
]]

# Copy final flat price table for export
df_merged_flat_price = df_merged.copy() 

# df_merged_flat_price is the df that will 
# be exported to sheet 1 in the workbook

# =========================== END of Sheet 1 construction =========================== 
### Start building Time Spreads and strips:

## i) M1/M2 spreads across the curve

# Dynamically find the last usable row for spreads
# Remove "Cash" if it's present at the top (M1/M2 logic assumes monthly structure)
df_spreads_base = df_merged[~df_merged["label"].str.lower().eq("cash")].copy()

# Ensure it's sorted properly (should already be, but safety)
df_spreads_base["sort_key"] = pd.to_datetime(df_spreads_base["label"], format="%b'%y", errors="coerce")
df_spreads_base = df_spreads_base.sort_values("sort_key").drop(columns="sort_key").reset_index(drop=True)

# Make sure there's at least 2 rows to do M1/M2
if len(df_spreads_base) < 2:
    raise ValueError("Not enough data to compute time spreads")

# Spread targets
spread_targets = [
    "JKM ($/MMBtu)",
    "TTF ($/MMBtu)",
    "NBP ($/MMBtu)",
    "Henry Hub ($/MMBtu)",
    "JKM v TTF"
]

# Build spread labels: "Aug/Sep'25", "Sep/Oct'25", etc.
labels_n = df_spreads_base["label"].values[:-1]
labels_np1 = df_spreads_base["label"].values[1:]
spread_labels = [f"{a}/{b}" for a, b in zip(labels_n, labels_np1)]

df_time_spreads = pd.DataFrame()
df_time_spreads["Spread Label"] = spread_labels

# Calculate M1-M2 spreads
for col in spread_targets:
    col_values = df_spreads_base[col].values
    spreads = [
        col_values[i] - col_values[i + 1]
        if pd.notna(col_values[i]) and pd.notna(col_values[i + 1])
        else np.nan
        for i in range(len(col_values) - 1)
    ]
    df_time_spreads[f"{col} Spread"] = spreads

# Optional: round spreads at the end
# df_time_spreads = df_time_spreads.round(3)

# Preview
#df_time_spreads.head()

## ii) Summer, Winter legs and the seasonal spreads between 
# Prep: Decompose contract labels
df_merged["contract_date"] = pd.to_datetime(df_merged["label"], format="%b'%y", errors='coerce')
df_merged = df_merged.dropna(subset=["contract_date"]).sort_values("contract_date").reset_index(drop=True)
df_merged["month"] = df_merged["contract_date"].dt.month
df_merged["year"] = df_merged["contract_date"].dt.year

# Define seasonal columns
price_cols = [
    "JKM ($/MMBtu)",
    "TTF ($/MMBtu)",
    "NBP ($/MMBtu)",
    "Henry Hub ($/MMBtu)",
    "JKM v TTF"
]
df_filtered = df_merged[["label", "contract_date", "month", "year"] + price_cols].copy()

# Seasonal averaging logic
def seasonal_avg(df, season, year):
    if season == "Win":
        mask = ((df["year"] == year) & df["month"].isin([10, 11, 12])) | \
               ((df["year"] == year + 1) & df["month"].isin([1, 2, 3]))
    else:  # Summer
        mask = (df["year"] == year) & df["month"].isin([4, 5, 6, 7, 8, 9])

    df_season = df[mask]
    if df_season.empty:
        return None

    out = {"Spread Label": f"{season}'{str(year)[-2:]}"}
    for col in price_cols:
        out[f"{col} Spread"] = df_season[col].mean(skipna=True)
    return out

# ---- Generate dynamic seasonal ladder in alternating order ----
today = datetime.today()
current_year = today.year
n_winters = 4
n_summers = 5
seasonal_ladder = []

# Always start with the first available Winter
for i in range(n_winters + n_summers):
    if i % 2 == 0:  # Even index → Winter
        y = current_year + (i // 2)
        expiry = datetime(y, 9, 30)
        if today <= expiry:
            win = seasonal_avg(df_filtered, "Win", y)
            if win:
                seasonal_ladder.append(win)
    else:  # Odd index → Summer
        y = current_year + ((i + 1) // 2)
        expiry = datetime(y, 3, 31)
        if today <= expiry:
            sumr = seasonal_avg(df_filtered, "Sum", y)
            if sumr:
                seasonal_ladder.append(sumr)

df_seasons = pd.DataFrame(seasonal_ladder)

# ---- Compute alternating spreads ----
spread_rows = []
for i in range(len(df_seasons) - 1):
    label_1 = df_seasons.iloc[i]["Spread Label"]
    label_2 = df_seasons.iloc[i + 1]["Spread Label"]

    spread_label = f"{label_1} - {label_2}"
    spread_row = {"Spread Label": spread_label}

    for col in price_cols:
        col_spread = f"{col} Spread"
        val_1 = df_seasons.iloc[i][col_spread]
        val_2 = df_seasons.iloc[i + 1][col_spread]
        spread_row[col_spread] = val_1 - val_2 if pd.notnull(val_1) and pd.notnull(val_2) else None

    spread_rows.append(spread_row)

df_season_spreads = pd.DataFrame(spread_rows)

# ---- Final seasonal + spread block ----
df_seasonal_strips = pd.concat([df_seasons, df_season_spreads], ignore_index=True).reset_index(drop=True)

## iii) Quarterly Legs and spreads between

# PREP
df = df_merged.copy()
df["contract_date"] = pd.to_datetime(df["label"], format="%b'%y", errors="coerce")
df = df.dropna(subset=["contract_date"]).copy()

df["quarter"] = df["contract_date"].dt.to_period("Q")
df["month"] = df["contract_date"].dt.month
df["year"] = df["contract_date"].dt.year

# EXPIRE QUARTERS LOGIC
# A quarter is expired if its *first month* has settled (settles end of previous month)
today = datetime.today()
expiry_map = {
    1: (12, -1),   # Jan → expires Dec (last day of dec) last year
    4: (3, 0),     # Apr → expires Mar
    7: (6, 0),     # Jul → expires Jun
    10: (9, 0),    # Oct → expires Sep
}

expired_quarters = []
# Introducing a Rolling 4-year horizon from today
cutoff_year = today.year + 4 

for q in df["quarter"].unique():
    q_year = q.year
    first_month = q.start_time.month
    expiry_month, offset_year = expiry_map[first_month]
    
    # Use day=28 as safe fallback for end-of-month
    expiry_date = datetime(q_year + offset_year, expiry_month, 28)
    
    # Drop quarters that have:
    # - Already expired, OR
    # - Have an expiry date beyond our 4-year horizon
    if today > expiry_date or q_year > cutoff_year:
        expired_quarters.append(q)

# Remove expired quarters
df = df[~df["quarter"].isin(expired_quarters)]

# VALID QUARTERS (3 months of data) 
quarter_counts = df.groupby("quarter")["label"].count()
valid_quarters = quarter_counts[quarter_counts == 3].index
df = df[df["quarter"].isin(valid_quarters)]

# QUARTERLY AVERAGES
quarterly_avg = df.groupby("quarter").agg({
    "JKM ($/MMBtu)": "mean",
    "TTF ($/MMBtu)": "mean",
    "NBP ($/MMBtu)": "mean",
    "Henry Hub ($/MMBtu)": "mean",
    "JKM v TTF": "mean"
}).reset_index()

# Format labels: Q1'27, etc.
def format_quarter_label(q): return f"Q{q.quarter}'{str(q.year)[-2:]}"
quarterly_avg["Spread Label"] = quarterly_avg["quarter"].apply(format_quarter_label)
quarterly_avg = quarterly_avg.drop(columns="quarter")

# Reorder & Rename
quarterly_avg = quarterly_avg[[  
    "Spread Label",
    "JKM ($/MMBtu)",
    "TTF ($/MMBtu)",
    "NBP ($/MMBtu)",
    "Henry Hub ($/MMBtu)",
    "JKM v TTF"
]].rename(columns={
    "JKM ($/MMBtu)": "JKM ($/MMBtu) Spread",
    "TTF ($/MMBtu)": "TTF ($/MMBtu) Spread",
    "NBP ($/MMBtu)": "NBP ($/MMBtu) Spread",
    "Henry Hub ($/MMBtu)": "Henry Hub ($/MMBtu) Spread",
    "JKM v TTF": "JKM v TTF Spread"
})

# Drop rows where all spreads are NaN (to avoid breaking diffs)
quarterly_avg = quarterly_avg.dropna(subset=[
    "JKM ($/MMBtu) Spread",
    "TTF ($/MMBtu) Spread",
    "NBP ($/MMBtu) Spread",
    "Henry Hub ($/MMBtu) Spread",
    "JKM v TTF Spread"
], how="all").reset_index(drop=True)

# ROLLING Q-on-Q SPREADS
rolling_spreads = quarterly_avg.copy()
rolling_spreads["Spread Label"] = (
    quarterly_avg["Spread Label"].shift() + " - " + quarterly_avg["Spread Label"]
)
rolling_spreads.iloc[0, rolling_spreads.columns.get_loc("Spread Label")] = None

for col in [
    "JKM ($/MMBtu) Spread",
    "TTF ($/MMBtu) Spread",
    "NBP ($/MMBtu) Spread",
    "Henry Hub ($/MMBtu) Spread",
    "JKM v TTF Spread"
]:
    rolling_spreads[col] = quarterly_avg[col].shift() - quarterly_avg[col]

rolling_spreads = rolling_spreads.dropna(subset=[
    "JKM ($/MMBtu) Spread",
    "TTF ($/MMBtu) Spread",
    "NBP ($/MMBtu) Spread",
    "Henry Hub ($/MMBtu) Spread",
    "JKM v TTF Spread"
], how="all").reset_index(drop=True)
# FINAL COMBINED OUTPUT
df_quarterly_strips = pd.concat([quarterly_avg, rolling_spreads], ignore_index=True)

## iv) Cal legs and the spreads between 
# --- PREP ---
df = df_merged.copy()
df["contract_date"] = pd.to_datetime(df["label"], format="%b'%y", errors="coerce")
df = df.dropna(subset=["contract_date"]).copy()
df["month"] = df["contract_date"].dt.month
df["year"] = df["contract_date"].dt.year

# --- EXPIRE CAL LOGIC ---
today = datetime.today()
cutoff_year = today.year + 5  # Rolling 5-year forward horizon

# Cal year 'YY' expires at end of Dec(YY-1)
valid_years = []
for y in df["year"].unique():
    if today <= datetime(y - 1, 12, 31) and y <= cutoff_year:
        valid_years.append(y)

# Filter to valid years
df = df[df["year"].isin(valid_years)]

# --- CALENDAR AVERAGES ---
cal_avg = df.groupby("year")[[  
    "JKM ($/MMBtu)",
    "TTF ($/MMBtu)",
    "NBP ($/MMBtu)",
    "Henry Hub ($/MMBtu)",
    "JKM v TTF"
]].mean().round(4).reset_index()

# Format label: Cal 'YY
cal_avg["Spread Label"] = "Cal '" + cal_avg["year"].astype(str).str[-2:]
cal_avg = cal_avg.drop(columns="year")

# Reorder & rename
cal_avg = cal_avg[[  
    "Spread Label",
    "JKM ($/MMBtu)",
    "TTF ($/MMBtu)",
    "NBP ($/MMBtu)",
    "Henry Hub ($/MMBtu)",
    "JKM v TTF"
]].rename(columns={
    "JKM ($/MMBtu)": "JKM ($/MMBtu) Spread",
    "TTF ($/MMBtu)": "TTF ($/MMBtu) Spread",
    "NBP ($/MMBtu)": "NBP ($/MMBtu) Spread",
    "Henry Hub ($/MMBtu)": "Henry Hub ($/MMBtu) Spread",
    "JKM v TTF": "JKM v TTF Spread"
})

# --- KEEP partial data ---
cal_avg = cal_avg.dropna(
    subset=[
        "JKM ($/MMBtu) Spread",
        "TTF ($/MMBtu) Spread",
        "NBP ($/MMBtu) Spread",
        "Henry Hub ($/MMBtu) Spread",
        "JKM v TTF Spread"
    ],
    how="all"  # drop only rows with ALL NaNs
).reset_index(drop=True)

# --- CALENDAR SPREADS ---
cal_spreads = pd.DataFrame()
cal_spreads["Spread Label"] = (
    cal_avg["Spread Label"].shift() + " - " + cal_avg["Spread Label"]
)

# Compute spreads for each column individually
for col in [
    "JKM ($/MMBtu) Spread",
    "TTF ($/MMBtu) Spread",
    "NBP ($/MMBtu) Spread",
    "Henry Hub ($/MMBtu) Spread",
    "JKM v TTF Spread"
]:
    cal_spreads[col] = cal_avg[col].shift() - cal_avg[col]

# Drop first row (invalid)
cal_spreads = cal_spreads.dropna(subset=["Spread Label"]).reset_index(drop=True)

# --- COMBINE ---
calendar_df = pd.concat([cal_avg, cal_spreads], ignore_index=True)

## Bringing  all together for sheet 2
 # Define consistent column order
columns_order = [
    "Spread Label",
    "JKM ($/MMBtu) Spread",
    "TTF ($/MMBtu) Spread",
    "NBP ($/MMBtu) Spread",
    "Henry Hub ($/MMBtu) Spread",
    "JKM v TTF Spread"
]

# Ensure all DataFrames have the same columns
def align_columns_safe(df, name):
    try:
        return df.reindex(columns=columns_order)
    except Exception as e:
        print(f"[ERROR] Align failed for {name}: {e}")
        return pd.DataFrame(columns=columns_order)

df_time_spreads = align_columns_safe(df_time_spreads, "df_time_spreads")
quarterly_df = align_columns_safe(df_quarterly_strips, "quarterly_df")
df_season_output = align_columns_safe(df_seasonal_strips, "df_season_output")
calendar_df = align_columns_safe(calendar_df, "calendar_df")

# Create a single blank row with NaNs
blank_row = pd.DataFrame([[""] + [None] * (len(columns_order) - 1)], columns=columns_order)

# Concatenate with blank rows in between
df_all_spreads = pd.concat([
    df_time_spreads,
    blank_row,
    quarterly_df,
    blank_row,
    df_season_output,
    blank_row,
    calendar_df
], ignore_index=True) # this can go in sheet 2

# =========================== END of Sheet 2 construction ===========================

# Create filename
today_str = datetime.today().strftime("%Y-%m-%d")
filename = f"LNG_Pricing_Sheet_{today_str}.xlsx"

# --- Round ONLY sheet 2 ---
for col in df_all_spreads.columns:
    if "Spread" in col and df_all_spreads[col].dtype.kind in "fc":
        df_all_spreads[col] = df_all_spreads[col].round(3)

# --- Export to Excel ---
with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
    # Write both DataFrames
    df_merged_flat_price.to_excel(writer, sheet_name="Flat Prices", index=False)
    df_all_spreads.to_excel(writer, sheet_name="Spread Summary", index=False)

    # Access workbook and worksheets
    workbook = writer.book
    sheet1 = writer.sheets["Flat Prices"]
    sheet2 = writer.sheets["Spread Summary"]

    # Define formats
    format_2dp = workbook.add_format({"align": "center", "valign": "vcenter", "num_format": "0.00"})
    format_3dp = workbook.add_format({"align": "center", "valign": "vcenter", "num_format": "0.000"})
    format_4dp = workbook.add_format({"align": "center", "valign": "vcenter", "num_format": "0.0000"})
    format_default = workbook.add_format({"align": "center", "valign": "vcenter"})

    # --- Sheet 1 formatting ---
    for idx, col in enumerate(df_merged_flat_price.columns):
        max_len = max(len(str(col)), df_merged_flat_price[col].astype(str).map(len).max()) + 2

        if col == "Brent ($/bbl)" or col == "NBP (p/th)":
            fmt = format_2dp
        elif col == "Cable Fwd Rate":
            fmt = format_4dp
        elif df_merged_flat_price[col].dtype.kind in "fc":
            fmt = format_3dp
        else:
            fmt = format_default

        sheet1.set_column(idx, idx, max_len, fmt)

    # --- Sheet 2 formatting (all 3dp) ---
    for idx, col in enumerate(df_all_spreads.columns):
        max_len = max(len(str(col)), df_all_spreads[col].astype(str).map(len).max()) + 2
        sheet2.set_column(idx, idx, max_len, format_3dp)

    # Freeze top row on both
    sheet1.freeze_panes(1, 0)
    sheet2.freeze_panes(1, 0)

# Serves as a checker in the original ipynb
#print(f"Exported to: {filename}")

def send_email_with_attachment(
    filename,       # Excel file path to attach
    subject,        # Subject line
    body,           # Plain-text body
    to_email,       # List of TO recipients
    from_email,     # Sender address
    smtp_server,    # e.g. "smtp.gmail.com" or "smtp.office365.com"
    smtp_port,      # Port: Gmail=465 (SSL), O365=587 (STARTTLS) / 25 (TLS)
    login,          # SMTP login (usually same as from_email)
    password,       # Password or App password
    cc=None,        # Optional CC list
    bcc=None,       # Optional BCC list
    use_ssl=False,  # Use SSL (Gmail 465)
    use_starttls=False,  # Use STARTTLS (Office365 587)
    debug=False     # Print SMTP conversation (optional)
):
    # --- Build the email ---
    msg = EmailMessage()                  
    msg['Subject'] = subject              # Add subject
    msg['From'] = from_email              # Set sender
    msg['To'] = ', '.join(to_email)       # Join list of TO recipients

    if cc:
        msg['Cc'] = ', '.join(cc)         # Add CC recipients if provided
    
    msg.set_content(body)                 # Add plain-text body

    # --- Attach the Excel file ---
    with open(filename, 'rb') as f:
        file_data = f.read()
        file_name = os.path.basename(f.name)
    msg.add_attachment(
        file_data, 
        maintype='application', 
        subtype='octet-stream',
        filename=file_name
    )

    # --- Combine all recipients (ensures CC + BCC also get it) ---
    all_recipients = to_email + (cc if cc else []) + (bcc if bcc else [])

    # --- Send via SMTP ---
    if use_ssl:
        # Gmail: SSL on port 465
        with smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=60) as smtp:
            if debug: smtp.set_debuglevel(1)
            smtp.login(login, password)              # Authenticate
            smtp.send_message(msg, to_addrs=all_recipients)
    else:
        # Office365: STARTTLS on 587 or TLS on 25
        with smtplib.SMTP(smtp_server, smtp_port, timeout=60) as smtp:
            if debug: smtp.set_debuglevel(1)
            smtp.ehlo()
            if use_starttls:
                smtp.starttls()                      # Upgrade to TLS
                smtp.ehlo()
            smtp.login(login, password)              # Authenticate
            smtp.send_message(msg, to_addrs=all_recipients)    

# Email config
send_email_with_attachment(
    filename=filename,                                   # Path to LNG Excel file
    subject=f"LNG Pricing Sheet – {today_str}",          # Dynamic subject line
    body="Hi Miguel,\n\nPlease find attached the latest LNG pricing sheet.",

    to_email=["Miguel.Arroyo@irh.ae"],                   # Main recipient
    cc=["energy@irh.ae", "vedant.bundellu@irh.ae", "Zinat.Juma@irh.ae"],  # CC list
    bcc=["Jayesh.Verma@irh.ae"],                         # Optional BCC

    from_email="alert@irh.ae",                           # Always send from alert@irh
    smtp_server="smtp.office365.com",                    # Office365 SMTP
    smtp_port=587,                                       # Submission port
    login="alert@irh.ae",
    password=os.environ.get("EMAIL_PASSWORD"),           # Secure env var
    use_starttls=True                                    # STARTTLS required by O365
)