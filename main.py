import streamlit as st
import subprocess
import sys
try:
    st.set_page_config(page_title="Zak Fund", layout="wide")
except:
    pass
@st.cache_resource
def connectingToStockData():
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "yfinance"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "streamlit-cookies-manager"], check=True)

connectingToStockData()

import pytz
import time as tm
import random
from datetime import datetime, time
import pandas as pd
import yfinance as yf
import cloudscraper
import altair as alt
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
from supabase import create_client, Client
from streamlit_cookies_manager import EncryptedCookieManager
from streamlit_javascript import st_javascript


cookiesPass = st.secrets["COOKIES_PASSWORD"]
cookies = EncryptedCookieManager(
    prefix="zakfund/",
    password=cookiesPass,
)

if not cookies.ready():
    st.stop()

# -----------------------------------------------------------------------------
# Streamlit / Supabase Setup
# -----------------------------------------------------------------------------

class tradeAction:
    BUY = "BOUGHT"
    SELL = "SELL"
    CLOSE = "CLOSED"
    SPREAD = "SPREAD"
    BTO='BTO'
    STC='STC'
    STO='STO'
    BTC='BTC'

url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]
ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
baseURL = st.secrets["BASEAPI"]  # For fetching option chain data

supabase: Client = create_client(url, key)

# For storing multi-leg/spread picks
if "spread_legs" not in st.session_state:
    st.session_state["spread_legs"] = []


# -----------------------------------------------------------------------------
# Database CRUD Helpers
# -----------------------------------------------------------------------------
def get_team_by_password(pwd: str):
    """Return the team row if password matches, else None."""
    resp = supabase.table("teams").select("*").eq("team_password", pwd).execute()
    if resp.data:
        return resp.data[0]
    return None

def get_team_info(team_id: int):
    resp = supabase.table("teams").select("*").eq("id", team_id).execute()
    if resp.data:
        return resp.data[0]
    return None

def upsert_team(data_dict, team_id=None):
    if team_id is None:
        supabase.table("teams").insert(data_dict).execute()
    else:
        supabase.table("teams").update(data_dict).eq("id", team_id).execute()

# ---- SHARES ----
def load_shares(team_id: int) -> pd.DataFrame:
    resp = supabase.table("portfolio_shares").select("*").eq("team_id", team_id).execute()
    return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()

def upsert_shares(team_id: int, ticker: str, shares_held: float, avg_cost: float, current_price: float, row_id=None):
    unreal_pl = (current_price - avg_cost) * shares_held
    data = {
        "team_id": team_id,
        "ticker": ticker.upper(),
        "shares_held": shares_held,
        "avg_cost": avg_cost,
        "current_price": current_price,
        "unrealized_pl": unreal_pl
    }
    if row_id:
        supabase.table("portfolio_shares").update(data).eq("id", row_id).execute()
    else:
        # upsert on team_id,ticker
        supabase.table("portfolio_shares").upsert(data, on_conflict="team_id,ticker").execute()

def delete_shares_by_id(row_id: int):
    supabase.table("portfolio_shares").delete().eq("id", row_id).execute()

# ---- OPTIONS ----
def load_options(team_id: int) -> pd.DataFrame:
    resp = supabase.table("portfolio_options").select("*").eq("team_id", team_id).execute()
    return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()

def upsert_option(opt_id: int, team_id: int, symbol: str, call_put: str, expiration: str, strike: float,
                  contracts_held: float, avg_cost: float, current_price: float):
    unreal_pl = (current_price - avg_cost) * (contracts_held * 100)
    data_dict = {
        "team_id": team_id,
        "symbol": symbol.upper(),
        "call_put": call_put,
        "expiration": expiration,
        "strike": strike,
        "contracts_held": contracts_held,
        "avg_cost": avg_cost,
        "current_price": current_price,
        "unrealized_pl": unreal_pl
    }
    if opt_id is None:
        supabase.table("portfolio_options").insert(data_dict).execute()
    else:
        supabase.table("portfolio_options").update(data_dict).eq("id", opt_id).execute()

def delete_option_by_id(opt_id: int):
    supabase.table("portfolio_options").delete().eq("id", opt_id).execute()

# ---- PERFORMANCE ----
def load_performance(team_id: int) -> pd.DataFrame:
    resp = supabase.table("performance").select("*").eq("team_id", team_id).execute()
    return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()

def upsert_performance(team_id: int, date_str: str, shares_val: float, options_val: float, total_val: float, pnl: float):
    supabase.table("performance").upsert({
        "team_id": team_id,
        "date": date_str,
        "shares_value": shares_val,
        "options_value": options_val,
        "total_value": total_val,
        "pnl": pnl
    }, on_conflict="team_id,date").execute()

# ---- ACTIVITY LOG ----
def load_activity(team_id: int) -> pd.DataFrame:
    resp = (
        supabase.table("portfolio_activity")
        .select("*")
        .eq("team_id", team_id)
        .order("id", desc=True)
        .execute()
    )
    return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()


# -----------------------------------------------------------------------------
# Price Fetching
# -----------------------------------------------------------------------------
def fetch_share_price(ticker: str) -> float:
    try:
        data = yf.Ticker(ticker).history(period="1d")
        if len(data) > 0:
            return float(data["Close"].iloc[-1])
    except:
        st.error(f"❌ Failed fetching {ticker} Price")
    return 0.0

@st.cache_data(ttl=60*5)
def get_options_chain(symbol: str):
    """Returns the JSON from your custom API structure:
       { "options": {
            "YYYY-MM-DD": {
               "c": { "STRIKE": {"b":..., "a":..., "oi":..., "v":...}, ... },
               "p": { ... }
            }, ...
         }
       }
    """
    full_url = f"{baseURL}?stock={symbol.upper()}&reqId={random.randint(1, 1000000)}"
    scraper = cloudscraper.create_scraper()
    response = scraper.get(full_url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"❌ Failed to fetch options chain for {symbol}. Code: {response.status_code}")
        return None

def compute_option_fill_price(bid: float, ask: float, is_buy: bool) -> float:
    """
    If (ask - bid)/ask > 0.2 => use mid price,
    else if buy => 60% of spread,
    else if sell => 40% of spread.
    """
    spread = ask - bid
    if ask <= 0:
        return 0
    spread_ratio = spread / ask
    if spread_ratio > 0.2:
        return (bid + ask) / 2
    else:

        if is_buy:
            return bid + random.uniform(0.5, 0.7) * spread
        else:
            return bid + random.uniform(0.3, 0.5) * spread


def fetch_option_price(symbol: str, expiration: str, strike: float, call_put: str, is_buy: bool) -> float:
    option_type = call_put.upper()

    try:
        ticker = yf.Ticker(symbol)
        opt_chain = ticker.option_chain(expiration)
        options = opt_chain.calls if option_type == "CALL" else opt_chain.puts

        match = options[options['strike'] == strike]
        if not match.empty:
            bid = float(match['bid'].values[0])
            ask = float(match['ask'].values[0])
            return compute_option_fill_price(bid, ask, is_buy)

    except Exception as e:
        st.warning(f"⚠ Using FallBack API")

    # -------- Fallback to BASEAPI ----------
    try:
        chain_data = get_options_chain(symbol)  # your BASEAPI wrapper
        exp_chain = chain_data["options"].get(expiration, {})

        if option_type == "CALL":
            option_map = exp_chain.get("c", {})
        else:
            option_map = exp_chain.get("p", {})

        strike_str = f"{strike:.2f}"
        option_entry = option_map.get(strike_str)

        if not option_entry:
            raise ValueError(f"Strike {strike} not found in BASEAPI either.")

        # Some chains return list of entries (OI, bid, ask, etc.)
        bid = option_entry.get("b", 0)
        ask = option_entry.get("a", 0)

        return compute_option_fill_price(bid, ask, is_buy)

    except Exception as e:
        st.error(f"❌ Failed to fetch option price from BASEAPI: {e}")
        return 0.0

# -----------------------------------------------------------------------------
# Activity Logging with Pytz
# -----------------------------------------------------------------------------
def get_est_time() -> str:
    """
    Returns the current time in Eastern Standard Time (EST) as a formatted string.

    Returns:
        str: Current EST/NY time in the format "MM/DD/YYYY HH:MM AM/PM".
    """
    est = pytz.timezone("America/New_York")
    now_est = datetime.now(est)
    return now_est.strftime("%m/%d/%Y %I:%M %p")

def isMarketHours() -> bool:
    """
    Checks if the current time is within U.S. stock market hours (9:30 AM - 4:15 PM EST).

    Returns:
        bool: True if within market hours, False otherwise.
    """
    est = pytz.timezone("America/New_York")
    now = datetime.now(est).time()

    market_open = time(9, 30)
    market_close = time(16, 15)

    return market_open <= now <= market_close

def log_shares_activity(team_id: int, trdAction:str, ticker: str, shares_added: float, price: float, realized_pl=0):
    action = "BOUGHT" if shares_added > 0 else "SOLD"
    color = "#65FE08" if shares_added > 0 else "red"
    sign = "+" if shares_added > 0 else ""

    colorNet = "red" if shares_added > 0 else "#65FE08"
    signNet = "-" if shares_added > 0 else "+"
    cost = price * shares_added
    now_str = get_est_time()

    msg = (
        f"<b style='color:{color};'>{action} {sign}{shares_added} shares</b> "
        f"of <b style='color:#FFD700;'>{ticker}</b> "
        f"at <b style='color:#FFD700;'>\${price:,.2f}</b> "
        f"(Total: <b style='color:{colorNet};'>{signNet}\${abs(cost):,.2f}</b>) "
        f"(Realized P/L: <b style='color:{'green' if realized_pl >= 0 else 'red'};'>\${realized_pl:,.2f}</b>) "
        f"on {now_str}"
    )

    supabase.table("portfolio_activity").insert({
        "team_id": team_id,
        "message": msg,
        "symbol": ticker.upper(),
        "call_put": None,  # Not applicable for stocks
        "expiration": None,  # Not applicable for stocks
        "strike": None,  # Not applicable for stocks
        "quantity": shares_added,
        "trade_type": trdAction,
        "type": "Stock",
        "fill_price": price,
        "realized_pl": realized_pl,
    }).execute()


def log_options_activity(team_id: int, trdAction:str, symbol: str, call_put: str, expiration, strike: float, contracts_added: float, price: float, realized_pl=0):
    color = "#65FE08" if contracts_added > 0 else "red"
    sign = "+" if contracts_added > 0 else ""

    colorNet = "red" if contracts_added > 0 else "#65FE08"
    signNet = "-" if contracts_added > 0 else "+"

    total_cost = price * abs(contracts_added) * 100
    now_str = get_est_time()
    exp_str = expiration if isinstance(expiration, str) else expiration.strftime("%Y-%m-%d")

    msg = (
        f"<b style='color:{color};'>{trdAction} {sign}{contracts_added} contract(s)</b> of "
        f"<b style='color:#FFD700;'> {symbol} {strike:.2f} {call_put} {exp_str}</b> "
        f"at <b>\${price:,.2f}</b> "
        f"(Total: <b style='color:{colorNet};'>{signNet}\${abs(total_cost):,.2f}</b>) "
        f"(Realized P/L: <b style='color:{'green' if realized_pl >= 0 else 'red'};'>\${realized_pl:,.2f}</b>) "
        f"on {now_str}"
    )

    supabase.table("portfolio_activity").insert({
        "team_id": team_id,
        "message": msg,
        "symbol": symbol.upper(),
        "call_put": call_put.upper(),
        "expiration": expiration,
        "strike": strike,
        "quantity": contracts_added,
        "trade_type": trdAction,
        "type": "Option",
        "fill_price": price,
        "realized_pl": realized_pl,
    }).execute()
# -----------------------------------------------------------------------------
# Refresh All Positions
# -----------------------------------------------------------------------------
def refresh_portfolio_prices(team_id: int):
    # Refresh options
    opts_df = load_options(team_id)
    today = datetime.today().strftime("%Y-%m-%d")

    for idx, row in opts_df.iterrows():
        symbol = row["symbol"]
        exp = row["expiration"]
        strike = float(row["strike"])
        call_put = row["call_put"]
        contracts_held = float(row["contracts_held"])
        current_price = row["current_price"]  # Last recorded price

        if contracts_held == 0:
            continue

        try:
            current_px = fetch_option_price(symbol, exp, strike, call_put, is_buy=(contracts_held > 0))
        except Exception as e:
            current_px = current_price  # fallback

        # If option has expired
        if today > exp:
            stock_px = fetch_share_price(symbol)
            intrinsic_value = 0
            is_itm = False

            if call_put.upper() == "CALL":
                intrinsic_value = max(0, stock_px - strike)
                is_itm = intrinsic_value > 0
            elif call_put.upper() == "PUT":
                intrinsic_value = max(0, strike - stock_px)
                is_itm = intrinsic_value > 0

            action = tradeAction.STC if contracts_held > 0 else tradeAction.BTC

            # OTM: expires worthless → no transaction beyond logging
            if not is_itm:
                realized_pl = row["avg_cost"] * abs(contracts_held) * 100 * (-1 if contracts_held > 0 else 1)

                log_options_activity(
                    team_id=team_id,
                    trdAction=action + " (Expired - OTM)",
                    symbol=symbol,
                    call_put=call_put,
                    expiration=exp,
                    strike=strike,
                    contracts_added=-contracts_held,
                    price=0.00,
                    realized_pl=realized_pl
                )

            # ITM: simulate exercise as a stock transaction at strike, using adjust_share_position
            else:
                total_shares = abs(contracts_held) * 100

                # Determine shares_delta based on option type and position
                if call_put.upper() == "CALL":
                    # Long Call → Receive shares (simulate buy) if contracts_held > 0;
                    # Short Call → Deliver shares (simulate sell) if contracts_held < 0
                    shares_delta = total_shares if contracts_held > 0 else -total_shares
                elif call_put.upper() == "PUT":
                    # Long Put → Deliver shares (simulate sell) if contracts_held > 0;
                    # Short Put → Receive shares (simulate buy) if contracts_held < 0
                    shares_delta = -total_shares if contracts_held > 0 else total_shares
                else:
                    shares_delta = 0

                # Adjust the share position using our helper
                adjust_share_position(team_id, symbol, shares_delta, strike, stock_px)

                # Log the option activity for ITM expiration.
                # (This logs the option premium component; the stock P/L is handled separately.)
                log_options_activity(
                    team_id=team_id,
                    trdAction=action + " (Expired ITM - Exercised)",
                    symbol=symbol,
                    call_put=call_put,
                    expiration=exp,
                    strike=strike,
                    contracts_added=-contracts_held,
                    price=strike,
                    realized_pl = -row["avg_cost"] * contracts_held * 100
                )

            delete_option_by_id(row["id"])
            continue

        # If not expired, continue normal unrealized PnL update for options
        if contracts_held > 0:
            new_unreal = (current_px - row["avg_cost"]) * contracts_held * 100
        else:
            new_unreal = (row["avg_cost"] - current_px) * abs(contracts_held) * 100

        supabase.table("portfolio_options").update({
            "current_price": current_px,
            "unrealized_pl": new_unreal
        }).eq("id", row["id"]).execute()

    # Refresh shares
    shares_df = load_shares(team_id)
    for idx, row in shares_df.iterrows():
        new_px = fetch_share_price(row["ticker"])
        new_unreal = (new_px - row["avg_cost"]) * row["shares_held"]
        supabase.table("portfolio_shares").update({
            "current_price": new_px,
            "unrealized_pl": new_unreal
        }).eq("id", row["id"]).execute()

    # Recalc performance
    calculate_and_record_performance(team_id)


def adjust_share_position(team_id, symbol, shares_delta, strike_price, stock_px):
    existing = load_shares(team_id)
    row = existing[existing["ticker"] == symbol.upper()]
    old_shares = float(row["shares_held"].iloc[0]) if not row.empty else 0
    old_avg = float(row["avg_cost"].iloc[0]) if not row.empty else 0
    row_id = row["id"].iloc[0] if not row.empty else None

    new_total = old_shares + shares_delta

    # 🎯 Determine proper avg cost
    if shares_delta > 0:
        # Buying more = weighted average
        if new_total != 0:
            new_avg = ((old_shares * old_avg) + (shares_delta * strike_price)) / new_total
        else:
            new_avg = strike_price
        realized_pl = 0  # No profit yet on buys
    elif shares_delta < 0:
        if new_total >= 0:
            # Selling from existing holdings → realized profit
            realized_pl = (strike_price - old_avg) * abs(shares_delta)
            new_avg = old_avg
        else:
            # You flipped to short → new short basis
            realized_pl = (strike_price - old_avg) * old_shares  # Profit only from shares you actually owned
            new_avg = strike_price  # Reset cost basis for net short position
    else:
        new_avg = old_avg
        realized_pl = 0

    upsert_shares(team_id, symbol, shares_held=new_total, avg_cost=new_avg, current_price=stock_px, row_id=row_id)

    trade_type = tradeAction.BUY if shares_delta > 0 else tradeAction.SELL
    log_shares_activity(team_id, trade_type, symbol, shares_delta, strike_price, realized_pl=realized_pl)


def compute_portfolio_stats(team_id):
    """Calculate portfolio performance stats for a given team."""
    shares_df = load_shares(team_id)
    opts_df = load_options(team_id)

    # Share & Option Valuation
    sum_shares_val = (shares_df["shares_held"] * shares_df["current_price"]).sum() if not shares_df.empty else 0.0
    sum_opts_val = (opts_df["contracts_held"] * 100 * opts_df["current_price"]).sum() if not opts_df.empty else 0.0

    # Capital Allocation
    team = get_team_info(team_id)
    original_cap = float(team["initial_capital"])
    restricted_cap = float(team["restricted_capital"])

    activity_df = load_activity(team_id)
    realized_pl = activity_df["realized_pl"].sum() if not activity_df.empty else 0.0
    free_pool = original_cap + realized_pl - restricted_cap

    spent_shares = (shares_df["shares_held"] * shares_df["avg_cost"]).sum() if not shares_df.empty else 0.0
    spent_opts = (opts_df["contracts_held"] * 100 * opts_df["avg_cost"]).sum() if not opts_df.empty else 0.0
    used_invest = spent_shares + spent_opts
    free_cash = free_pool - used_invest

    # PnL Calculation
    unrealized_pl = (sum_opts_val + sum_shares_val) - used_invest
    total_pnl = realized_pl + unrealized_pl

    # Total Portfolio Value
    total_val = sum_shares_val + sum_opts_val + free_cash + restricted_cap

    return {
        "shares_df": shares_df,
        "opts_df": opts_df,
        "sum_shares_val": float(sum_shares_val),
        "sum_opts_val": float(sum_opts_val),
        "free_cash": float(free_cash),
        "total_val": float(total_val),
        "realized_pl": float(realized_pl),
        "unrealized_pl": float(unrealized_pl),
        "total_pnl": float(total_pnl)
    }

def calculate_and_record_performance(team_id):
    """Update the performance table based on computed portfolio values."""
    stats = compute_portfolio_stats(team_id)
    today_str = datetime.today().strftime("%Y-%m-%d")

    supabase.table("performance").upsert({
        "team_id": int(team_id),
        "date": today_str,
        "shares_value": stats["sum_shares_val"],
        "options_value": stats["sum_opts_val"],
        "total_value": stats["total_val"],
        "realized_pl": stats["realized_pl"],
        "unrealized_pl": stats["unrealized_pl"]
    }, on_conflict="team_id,date").execute()


# -----------------------------------------------------------------------------
# Main Streamlit
# -----------------------------------------------------------------------------

def main():
    if "user_tz" not in cookies or not cookies["user_tz"]:
        tz_js = st_javascript("Intl.DateTimeFormat().resolvedOptions().timeZone")
        if tz_js:
            st.session_state["user_tz"] = tz_js
            cookies["user_tz"] = tz_js
            cookies.save()
    else:
        st.session_state["user_tz"] = cookies["user_tz"]

    # Synchronize session state with cookies
    st.session_state["logged_in"] = cookies.get("logged_in", "false") == "true"
    st.session_state["is_admin"] = cookies.get("is_admin", "false") == "true"
    st.session_state["team_id"] = int(cookies.get("team_id", '-1'))

    # Authentication logic
    if not st.session_state["logged_in"]:
        show_login_screen()
    else:
        if st.sidebar.button("🚪 Logout"):
            handle_logout()
        if st.session_state["is_admin"]:
            show_admin_panel()
        else:
            show_team_portfolio()


def handle_logout():
    # Clear session state
    st.session_state.clear()

    # **Explicitly remove cookies by setting them to empty strings**
    cookies["logged_in"] = "false"
    cookies["is_admin"] = "false"
    cookies["team_id"] = '-1'
    cookies.save()

    st.success("Logged out.")
    tm.sleep(1)
    st.rerun()

def show_login_screen():
    st.title("💼 Zak Fund")
    pwd = st.text_input("🔐 Enter Team Password or Admin Password:", type="password")

    if st.button("Login"):
        if pwd == ADMIN_PASSWORD:
            st.session_state["logged_in"] = True
            st.session_state["is_admin"] = True

            # Store values as strings in cookies
            cookies["logged_in"] = "true"
            cookies["is_admin"] = "true"
            cookies.save()

            st.rerun()
        else:
            team_row = get_team_by_password(pwd)
            if team_row:
                st.session_state["logged_in"] = True
                st.session_state["is_admin"] = False
                st.session_state["team_id"] = team_row["id"]

                # Store values as strings in cookies
                cookies["logged_in"] = "true"
                cookies["is_admin"] = "false"
                cookies["team_id"] = str(team_row["id"])
                cookies.save()
                refresh_portfolio_prices(team_row["id"])

                st.rerun()
            else:
                st.error("Invalid password!")



def show_admin_panel():
    st.title("🛠️ Admin Panel")


    st.subheader("👥 Create / Edit Teams")
    teams_resp = supabase.table("teams").select("*").execute()
    teams_list = teams_resp.data if teams_resp.data else []
    team_labels = ["(New)"] + [f"{t['id']}: {t['team_name']}" for t in teams_list]
    selected_team = st.selectbox("Select Team to Edit", team_labels)

    if selected_team == "(New)":
        team_id = None
        def_name = ""
        def_pwd = ""
        def_init = 100000.0
        def_restrict = 20000.0
    else:
        tid = int(selected_team.split(":")[0])
        existing = next((x for x in teams_list if x["id"] == tid), None)
        if existing:
            team_id = existing["id"]
            def_name = existing["team_name"]
            def_pwd = existing["team_password"]
            def_init = float(existing["initial_capital"])
            def_restrict = float(existing["restricted_capital"])
        else:
            team_id = None
            def_name, def_pwd, def_init, def_restrict = "", "", 100000.0, 20000.0

    tname = st.text_input("Team Name", value=def_name)
    tpwd = st.text_input("Team Password", value=def_pwd)
    init_cap = st.number_input("Initial Capital", value=def_init, step=5000.0)
    restr_cap = st.number_input("Restricted Capital", value=def_restrict, step=1000.0)

    if st.button("🔽 Save Team"):
        data_dict = {
            "team_name": tname,
            "team_password": tpwd,
            "initial_capital": init_cap,
            "restricted_capital": restr_cap
        }
        upsert_team(data_dict, team_id)
        st.success("Team saved/updated!")
        st.rerun()

    st.subheader("📋 All Teams Overview")
    # Quick summary table
    rows = []
    for t in teams_list:
        tid = t["id"]
        # Refresh current data from DB
        refresh_portfolio_prices(tid)
        port_stat = compute_portfolio_stats(tid)

        rows.append({
                "Team": t["team_name"],
                "Initial Cap": round(t["initial_capital"], 2),
                "Restricted": round(t["restricted_capital"], 2),
                "Portfolio Value": round(port_stat["total_val"], 2),
                "PnL": round(port_stat["total_pnl"], 2),
            })
    df_admin = pd.DataFrame(rows)

    st.dataframe(df_admin, use_container_width=True)

def show_team_portfolio():
    team_id = st.session_state["team_id"]
    team = get_team_info(team_id)
    if not team:
        st.error("Team not found.")
        return

    st.title(f"🤼 Team: {team['team_name']}")

    # 🔥 Use the centralized function to get portfolio data
    portfolio_stats = compute_portfolio_stats(team_id)

    shares_df = portfolio_stats["shares_df"]
    opts_df = portfolio_stats["opts_df"]

    free_cash = portfolio_stats["free_cash"]

    with st.sidebar:
        if st.button("🔃 Refresh Prices"):
            refresh_portfolio_prices(team_id)
            st.rerun()
        st.header("📊 Portfolio Overview")
        st.text_input("💰 Starting Capital", value=f"${float(team['initial_capital']):,.2f}", disabled=True)
        st.text_input("📈 Total Account Value", value=f"${float(portfolio_stats['total_val']):,.2f}", disabled=True)
        st.text_input(
            f"💹 Shares Portion - {portfolio_stats['sum_shares_val'] / portfolio_stats['total_val'] * 100:.2f}%",
            value=f"${float(portfolio_stats['sum_shares_val']):,.2f}",
            disabled=True
        )
        st.text_input(
            f"📃 Options Portion - {portfolio_stats['sum_opts_val'] / portfolio_stats['total_val'] * 100:.2f}%",
            value=f"${float(portfolio_stats['sum_opts_val']):,.2f}",
            disabled=True
        )
        st.text_input(
            f"💵 Free Cash - {portfolio_stats['free_cash'] / portfolio_stats['total_val'] * 100:.2f}%",
            value=f"${float(free_cash):,.2f}",
            disabled=True
        )
        st.text_input("🤑 Total PnL", value=f"${float(portfolio_stats['total_pnl']):,.2f}", disabled=True)

    # TABS
    tab_shares, tab_opts, tab_log, tab_perf = st.tabs([
        "💹 Shares",
        "📊 Options",
        "📝 Activity Log",
        "📈 Performance"
    ])


    with tab_shares:
        st.subheader("💼 Your Shares")
        if shares_df.empty:
            st.info("No share positions.")
        else:
            df_disp = shares_df.copy()
            df_disp["Position Value"] = df_disp["shares_held"] * df_disp["current_price"]
            st.dataframe(df_disp, use_container_width=True)

        st.subheader("🛒 Trade Shares")
        col1, col2 = st.columns(2)
        ticker_in = col1.text_input("Ticker (e.g. AAPL, SPY)")
        px = fetch_share_price(ticker_in)
        qty_in = col2.number_input("Shares to Buy(+) or Sell(-)", step=1.0)

        st.write(f"Current Price: ${px:,.2f}")
        if st.button("📝 Execute Share Trade"):
            if qty_in == 0:
                st.warning("No shares specified.")
            else:

                if px <= 0:
                    st.error("Invalid price fetch.")
                    return
                # find existing
                if not shares_df.empty:
                    row = shares_df[shares_df["ticker"] == ticker_in.upper()]
                    old_shares = float(row["shares_held"].iloc[0]) if not row.empty else 0
                    old_avg = float(row["avg_cost"].iloc[0]) if not row.empty else 0
                    row_id = row["id"].iloc[0] if not row.empty else None

                    new_total = old_shares + qty_in
                    if new_total < 0:
                        st.error("Cannot have negative shares.")
                        return
                else:
                    old_shares = 0
                    old_avg = 0
                    row_id = None
                    new_total = qty_in
                if qty_in > 0:
                    # Buying => check free_cash
                    cost = px * qty_in
                    if cost > free_cash:
                        st.error("Not enough free cash.")
                        return
                    # Weighted average
                    if new_total != 0:
                        new_avg = (old_shares * old_avg + qty_in * px) / new_total
                    else:
                        new_avg = 0
                    upsert_shares(team_id, ticker_in, new_total, new_avg, px, row_id=row_id)
                    log_shares_activity(team_id, tradeAction.BUY, ticker_in, qty_in, px, realized_pl=0)
                    st.success(f"Bought {qty_in} shares of {ticker_in} at ${px:,.2f}.")
                else:
                    # Selling
                    sell_amount = abs(qty_in)
                    if sell_amount > old_shares:
                        st.error("Not enough shares to sell.")
                        return
                    realized = (px - old_avg) * sell_amount
                    leftover = old_shares - sell_amount
                    if leftover == 0:
                        if row_id:
                            delete_shares_by_id(row_id)
                            log_shares_activity(team_id,tradeAction.CLOSE, ticker_in, qty_in, px, realized_pl=realized)
                            st.success(f"Closed {qty_in} shares of {ticker_in} at \${px:,.2f}. Realized PnL:\${realized:,.2f}")
                    else:
                        upsert_shares(team_id, ticker_in, leftover, old_avg, px, row_id=row_id)
                        log_shares_activity(team_id,tradeAction.SELL, ticker_in, qty_in, px, realized_pl=realized)
                        st.success(f"Sold {qty_in} shares of {ticker_in} at \${px:,.2f}. Realized PnL: \${realized:,.2f}")

                refresh_portfolio_prices(team_id)
                st.rerun()


    clear_selection_js = JsCode("""
        function() {
            const api = this.api;
            api.deselectAll();
        }
        """)

    def clearAggrids():
        st.components.v1.html(f"""
            <script>
            const iframe = parent.document.querySelector('iframe[title="exist_opts"]');
            if (iframe) {{
                const grid = iframe.contentWindow;
                grid.{clear_selection_js};
            }}
            </script>
            """, height=0)

    with tab_opts:

        # -------------------------
        # Options Trading Section
        # -------------------------

        # Step 1: Select an Existing Position or Search for a New Trade
        st.markdown("### 🎯 Select Existing Position or Find a New Trade")
        existing_opts = load_options(st.session_state["team_id"])
        if "selected_existing" not in st.session_state:
            st.session_state.selected_existing = pd.DataFrame()
        if "selected_calls" not in st.session_state:
            st.session_state.selected_calls = pd.DataFrame()
        if "selected_puts" not in st.session_state:
            st.session_state.selected_puts = pd.DataFrame()

        # 🔹 User selects one or multiple existing positions for closing (supports spreads)
        if not existing_opts.empty:
            existing_opts.rename(columns={'symbol': 'Symbol', 'expiration': 'Expiration', 'strike': 'Strike', 'call_put': 'Type',
                                          'contracts_held': 'Contracts Held', 'avg_cost': 'Avg Cost', 'current_price': 'Current Price', 'unrealized_pl': 'Total PnL',
                                          'created_at': 'Bought On', 'updated_at': 'Last Updated'
                                        }, inplace=True)

            user_tz = pytz.timezone(st.session_state.get("user_tz", "America/New_York"))

            existing_opts["Bought On"] = pd.to_datetime(existing_opts["Bought On"], utc=True).dt.tz_convert(user_tz).dt.strftime("%b %d, %Y %I:%M %p")
            existing_opts["Last Updated"] = pd.to_datetime(existing_opts["Last Updated"], utc=True).dt.tz_convert(user_tz).dt.strftime("%b %d, %Y %I:%M %p")
            existing_opts["Avg Cost"] = existing_opts["Avg Cost"].round(2)
            existing_opts["Current Price"] = existing_opts["Current Price"].round(2)
            existing_opts['Total PnL'] = existing_opts['Total PnL'].round(2)

            existing_opts["PnL %"] = ((existing_opts["Current Price"] - existing_opts["Avg Cost"]) / existing_opts["Avg Cost"] * 100).round(0).astype(int).astype(str) + "%"
            st.markdown("#### 📂 Existing Positions (Select to Close)")
            gb_exist = GridOptionsBuilder.from_dataframe(existing_opts.drop(columns=["id", "team_id"]))

            pre_selected = st.session_state.selected_existing.to_dict('records') if not st.session_state.selected_existing.empty else []
            gb_exist.configure_selection(selection_mode="multiple", use_checkbox=True, pre_selected_rows=pre_selected)
            gb_exist.configure_grid_options(onFirstDataRendered=clear_selection_js)
            gb_exist.configure_default_column(
                resizable=True,
                wrapText=True,
                minWidth=80  # Adjust as needed
            )
            grid_options_exist = gb_exist.build()
            grid_response_exist = AgGrid(
                existing_opts,
                gridOptions=grid_options_exist,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                height=250,
                fit_columns_on_grid_load=False,
                key='exist_opts',
                allow_unsafe_jscode=True
            )
            clearAggrids()

            st.session_state.selected_existing = pd.DataFrame(grid_response_exist.get("selected_rows", []))
        else:
            st.info("No existing option positions.")


        # 🔹 If no existing position is selected, allow the user to enter a new symbol
        selected_symbol = st.text_input("### 🎫 Symbol for Chain (e.g. SPY)", key="chain_symbol").upper()

        if selected_symbol:
            chain_data = get_options_chain(selected_symbol)
            if chain_data and "options" in chain_data:
                exps = sorted(chain_data["options"].keys())
                selected_expiration = st.selectbox("Expiration", exps, key="chain_exp")

                if selected_expiration:
                    stock_price = fetch_share_price(selected_symbol)

                    # Load Calls & Puts into separate DataFrames
                    calls_list = []
                    puts_list = []

                    for option_type, type_label in [("c", "CALL"), ("p", "PUT")]:
                        for strike_str, sdata in chain_data["options"][selected_expiration].get(option_type, {}).items():
                            strike = float(strike_str)
                            option_data = {
                                "Type": type_label,
                                'Symbol': selected_symbol,
                                "Strike": strike,
                                "Bid": sdata.get("b", 0),
                                "Ask": sdata.get("a", 0),
                                "Expiration": selected_expiration,
                                "OI": sdata.get("oi", 0),
                                "Volume": sdata.get("v", 0),
                                "Moneyness": (
                                    "ITM" if (stock_price > strike and type_label == "CALL") or
                                            (stock_price < strike and type_label == "PUT")
                                    else "OTM" if (stock_price < strike and type_label == "CALL") or
                                                (stock_price > strike and type_label == "PUT")
                                    else "ATM"
                                )
                            }
                            if type_label == "CALL":
                                calls_list.append(option_data)
                            else:
                                puts_list.append(option_data)

                    df_calls = pd.DataFrame(calls_list)
                    df_puts = pd.DataFrame(puts_list)

                    tab_calls, tab_puts = st.columns(2)

                    def configure_aggrid_dynamic(df, grid_type):
                        gb = GridOptionsBuilder.from_dataframe(df)
                        gb.configure_default_column(
                            resizable=True,
                            wrapText=True,
                            minWidth=80
                        )
                        if grid_type == 'CALLS':
                            pre_selected_calls = st.session_state.selected_calls.to_dict('records') if not st.session_state.selected_calls.empty else []
                            gb.configure_selection(selection_mode="multiple", use_checkbox=True, pre_selected_rows=pre_selected_calls)
                        elif grid_type == 'PUTS':
                            pre_selected_puts = st.session_state.selected_puts.to_dict('records') if not st.session_state.selected_puts.empty else []
                            gb.configure_selection(selection_mode="multiple", use_checkbox=True, pre_selected_rows=pre_selected_puts)
                        else:
                            gb.configure_selection(selection_mode="multiple", use_checkbox=True)
                        return gb.build()

                    with tab_calls:
                        st.markdown("#### 📈 Calls")
                        grid_options_calls = configure_aggrid_dynamic(df_calls, 'CALLS')
                        grid_response_calls = AgGrid(
                            df_calls,
                            gridOptions=grid_options_calls,
                            update_mode=GridUpdateMode.SELECTION_CHANGED,
                            height=250,
                            fit_columns_on_grid_load=False,
                            key='calls'
                        )
                        st.session_state.selected_calls = pd.DataFrame(grid_response_calls.get("selected_rows", []))

                    with tab_puts:
                        st.markdown("#### 📉 Puts")

                        grid_options_puts = configure_aggrid_dynamic(df_puts, 'PUTS')
                        grid_response_puts = AgGrid(
                            df_puts,
                            gridOptions=grid_options_puts,
                            update_mode=GridUpdateMode.SELECTION_CHANGED,
                            height=250,
                            fit_columns_on_grid_load=False,
                            key='puts'
                        )
                        st.session_state.selected_puts = pd.DataFrame(grid_response_puts.get("selected_rows", []))
        # Step 2: Trade Details Input
        trade_details = []
        if not st.session_state.selected_existing.empty or not st.session_state.selected_calls.empty or not st.session_state.selected_puts.empty:
            st.markdown("### 🧮 Specify Trade Details")

            selected_legs = pd.concat([st.session_state.selected_existing, st.session_state.selected_calls, st.session_state.selected_puts], ignore_index=True)
            for idx, leg in selected_legs.iterrows():
                with st.container():
                    col1, col2, col3, col4 = st.columns(4)
                    bid = leg.get('Bid', None)
                    ask = leg.get('Ask', None)

                    if pd.notnull(bid) and pd.notnull(ask):
                        bid_ask_text = f"💰 **{float(bid):.2f}B - {float(ask):.2f}A**"
                    else:
                        avg = leg.get('Avg Cost', 0)
                        bid_ask_text = f"💸 **Current Avg: ${avg:,.2f}**"

                    contracts_held = leg.get("Contracts Held", 0)
                    contracts_display = int(contracts_held) if pd.notnull(contracts_held) else 0

                    col1.markdown(f"""
                    ### {idx+1}. **{leg['Symbol']} {leg['Strike']} {leg['Type']}**
                    📅 Exp: {leg['Expiration']} {'📦 ' + str(contracts_display) + ' Contracts' if contracts_display != 0 else ''}

                    {bid_ask_text}
                    """)

                    avgCost = leg.get('Avg Cost', 0)
                    default_action_index = 1 if pd.notnull(avgCost) and contracts_display > 0 else 0
                    action = col2.selectbox("Action", options=["🟢 Buy", "🔴 Sell"], key=f"action_{idx}", index=default_action_index)

                    qty = col3.number_input("Contracts", value=abs(contracts_display) if contracts_display != 0 else 1, min_value=1, step=1, key=f"qty_{idx}")
                    fill_price = col4.number_input("Estimated Fill Price",value=fetch_option_price(leg["Symbol"], leg["Expiration"], leg["Strike"], leg["Type"], is_buy=(action == "Buy")),  disabled=True, key=f"price_{idx}")

                    trade_details.append({
                        "Type": leg["Type"],
                        "Strike": leg["Strike"],
                        "Action": action,
                        "Qty": int(qty),
                        "Fill Price": fill_price,
                        "Symbol": leg["Symbol"],
                        "Expiration": leg["Expiration"]
                    })



            # if isMarketHours() and trade_details and  st.button("Execute Trade", on_click=reviewTrade):
            if trade_details:
                # Save in session state to persist after rerun
                closing_positions = []
                opening_positions = []
                if all(detail["Qty"] == 0 for detail in trade_details):
                    st.warning("⚠️ Please specify a nonzero quantity for at least one leg.")
                else:


                    opts_df = load_options(st.session_state["team_id"])

                    for detail in trade_details:
                        existing = pd.DataFrame() if opts_df.empty else opts_df[
                            (opts_df["symbol"] == detail['Symbol'].upper()) &
                            (opts_df["call_put"] == detail["Type"]) &
                            (opts_df["strike"] == detail["Strike"]) &
                            (opts_df["expiration"] == detail['Expiration'])
                        ]

                        old_contracts = 0.0
                        old_avg = 0.0
                        opt_id = None

                        if not existing.empty:
                            old_contracts = float(existing["contracts_held"].iloc[0])
                            old_avg = float(existing["avg_cost"].iloc[0])
                            opt_id = existing["id"].iloc[0]

                        trade_qty = detail["Qty"]
                        fill_price = detail["Fill Price"]
                        action = detail["Action"]

                        # ✅ Determine if this is a closing or opening trade
                        if action == "🟢 Buy":
                            if old_contracts < 0:
                                detail["Qty"] = -detail["Qty"]
                                closing_positions.append(detail)
                            else:
                                opening_positions.append(detail)

                        elif action == "🔴 Sell":
                            if old_contracts > 0:
                                closing_positions.append(detail)
                            else:
                                detail["Qty"] = -detail["Qty"]
                                opening_positions.append(detail)

                    amount_spent = 0
                    amount_received = 0

                    for d in opening_positions:
                        leg_total = d["Fill Price"] * d["Qty"] * 100

                        if leg_total > 0:
                            amount_spent += abs(leg_total) #buy to open (Spending money)
                        elif leg_total < 0:
                            amount_received += abs(leg_total) #sell to open (getting credit for it)

                    for d in closing_positions:
                        leg_total = d["Fill Price"] * d["Qty"] * 100

                        if leg_total > 0:
                            amount_received += abs(leg_total) #sell to close (getting money)
                        elif leg_total < 0:
                            amount_spent += abs(leg_total) #buying to close (spending money to close)

                    net_total = amount_spent - amount_received

                    st.markdown(f"""
                    ### 💸 Trade Cost Summary:
                    - 🔴 **Amount Spent**: ${amount_spent:,.2f}
                    - 🟢 **Amount Received**: ${amount_received:,.2f}
                    - 💰 **Total (Spent - Received)**: ${net_total:,.2f}
                    """)

                    # if opening_positions:
                    #     # st.write(total_closing_credit, total_opening_cost)
                    #     if total_closing_credit > total_opening_cost:
                    #         st.error("❌ You are trying to execute a **credit** trade. Only **debit** trades are allowed.")
                    #         st.stop()

                # if st.button('🚀 Execute Trade') and isMarketHours():
                if st.button('🚀 Execute Trade') and isMarketHours():
                    # ✅ Process Opening Trades
                    for detail in opening_positions:
                        existing = pd.DataFrame() if opts_df.empty else opts_df[
                            (opts_df["symbol"] == detail['Symbol'].upper()) &
                            (opts_df["call_put"] == detail["Type"]) &
                            (opts_df["strike"] == detail["Strike"]) &
                            (opts_df["expiration"] == detail['Expiration'])
                        ]

                        old_contracts = float(existing["contracts_held"].iloc[0]) if not existing.empty else 0.0
                        old_avg = float(existing["avg_cost"].iloc[0]) if not existing.empty else 0.0
                        opt_id = existing["id"].iloc[0] if not existing.empty else None

                        trade_qty = detail["Qty"]
                        fill_price = fetch_option_price(detail["Symbol"], detail["Expiration"], detail["Strike"], detail["Type"], is_buy= True if trade_qty > 0 else False)

                        new_total = old_contracts + trade_qty
                        new_avg = ((old_contracts * old_avg) + (trade_qty * fill_price)) / new_total if new_total != 0 else fill_price

                        upsert_option(opt_id, st.session_state["team_id"], detail['Symbol'], detail["Type"], detail['Expiration'], detail["Strike"], new_total, new_avg, fill_price)
                        log_options_activity(st.session_state["team_id"], tradeAction.BTO if trade_qty > 0 else tradeAction.STO, detail['Symbol'], detail["Type"],
                                            detail['Expiration'], detail["Strike"], trade_qty, fill_price, realized_pl=0)

                    # ✅ Process Closing Trades
                    for detail in closing_positions:
                        existing = opts_df[
                            (opts_df["symbol"] == detail['Symbol'].upper()) &
                            (opts_df["call_put"] == detail["Type"]) &
                            (opts_df["strike"] == detail["Strike"]) &
                            (opts_df["expiration"] == detail['Expiration'])
                        ]

                        if existing.empty:
                            st.error(f"Cannot close {detail['Qty']} contracts of {detail['Type']} {detail['Strike']} - No existing position.")
                            st.stop()

                        old_contracts = float(existing["contracts_held"].iloc[0])
                        old_avg = float(existing["avg_cost"].iloc[0])
                        opt_id = existing["id"].iloc[0]

                        trade_qty = detail["Qty"]
                        fill_price =  fetch_option_price(detail["Symbol"], detail["Expiration"], detail["Strike"], detail["Type"], is_buy= True if trade_qty > 0 else False)

                        if abs(trade_qty) > abs(old_contracts):
                            #this should also prevent the whole closing just one side and turn that into a new position
                            st.error(f"Cannot close more contracts than you own: {abs(old_contracts)} available.")
                            st.stop()

                        realized = (fill_price - old_avg) * trade_qty * 100 if old_contracts > 0 else (old_avg - fill_price) * trade_qty * 100
                        leftover = old_contracts - trade_qty

                        if leftover == 0:
                            delete_option_by_id(opt_id)
                            log_options_activity(st.session_state["team_id"], tradeAction.CLOSE, detail['Symbol'], detail["Type"],
                                                detail['Expiration'], detail["Strike"], -trade_qty, fill_price, realized_pl=realized)
                        else:
                            upsert_option(opt_id, st.session_state["team_id"], detail['Symbol'], detail["Type"], detail['Expiration'], detail["Strike"], leftover, old_avg, fill_price)
                            log_options_activity(st.session_state["team_id"], tradeAction.STC if trade_qty > 0 else tradeAction.BTC, detail['Symbol'], detail["Type"],
                                                detail['Expiration'], detail["Strike"], -trade_qty, fill_price, realized_pl=realized)

                    st.success("✅ Trade executed successfully.")
                    st.session_state.trade_stage = 0
                    st.session_state.opening_positions = []
                    st.session_state.closing_positions = []


                    clearAggrids()
                    calculate_and_record_performance(st.session_state["team_id"])

                    st.rerun()
                elif not isMarketHours():
                    st.error('⌚ OUTSIDE MARKET HOURS')

    with tab_log:
        st.subheader("👟 Activity Log")
        act_df = load_activity(team_id)
        if act_df.empty:
            st.info("🚫 No recent activity.")
        else:
            for _, row in act_df.iterrows():
                st.markdown(f"• {row['message']}", unsafe_allow_html=True)

    with tab_perf:
        st.subheader("📈 Performance Over Time")
        perf_df = load_performance(team_id)
        if perf_df.empty:
            st.info("⛔ No performance records yet.")
        else:
            perf_df = perf_df.sort_values("date")
            chart = (
                alt.Chart(perf_df)
                .mark_line()
                .encode(
                    x="date:T",
                    y=alt.Y("total_value:Q", title="Portfolio Value", scale=alt.Scale(zero=False))
                )
                .properties(height=300, width="container")
            )
            st.altair_chart(chart, use_container_width=True)

if __name__ == "__main__":
    main()
