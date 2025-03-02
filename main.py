import streamlit as st
import subprocess
import sys
try:
    st.set_page_config(page_title="Zak Fund", layout="wide")
except:
    pass
@st.cache_resource
def install_latest_yfinance():
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "yfinance"], check=True)

install_latest_yfinance()

import pytz
import time
import random
import datetime
import pandas as pd
import yfinance as yf
import cloudscraper
import altair as alt
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from supabase import create_client, Client

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

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False
if "team_id" not in st.session_state:
    st.session_state["team_id"] = None

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
        .limit(25)
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
    time.sleep(1)
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
    data = get_options_chain(symbol)
    if not data or "options" not in data:
        raise ValueError("Option chain data not found or invalid JSON structure.")

    all_exps = data["options"]
    if expiration not in all_exps:
        raise ValueError(f"No expiration {expiration} found for {symbol}.")

    cp_key = "c" if call_put.upper() == "CALL" else "p"
    cp_dict = all_exps[expiration].get(cp_key, {})
    if not cp_dict:
        raise ValueError(f"No {call_put} data for expiration {expiration} in chain.")

    strike_key = f"{strike:.2f}"
    if strike_key not in cp_dict:
        raise ValueError(f"Strike {strike} not found for {call_put} {expiration} {symbol}.")

    option_data = cp_dict[strike_key]
    bid = option_data.get("b", 0)
    ask = option_data.get("a", 0)
    fill_price = compute_option_fill_price(bid, ask, is_buy)
    return fill_price

# -----------------------------------------------------------------------------
# Activity Logging with Pytz
# -----------------------------------------------------------------------------
def get_est_time():
    """Return EST/NY time string."""
    tz = pytz.timezone("America/New_York")
    now_est = datetime.datetime.now(tz)
    return now_est.strftime("%m/%d/%Y %I:%M %p")


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
    """
    Fetch the latest share price for each ticker, and
    fetch the latest option price for each option position.
    Update the DB. Then recalc performance.
    """
    # Refresh shares
    shares_df = load_shares(team_id)
    for idx, row in shares_df.iterrows():
        new_px = fetch_share_price(row["ticker"])
        new_unreal = (new_px - row["avg_cost"]) * row["shares_held"]
        supabase.table("portfolio_shares").update({
            "current_price": new_px,
            "unrealized_pl": new_unreal
        }).eq("id", row["id"]).execute()

    # Refresh options
    opts_df = load_options(team_id)
    for idx, row in opts_df.iterrows():
        symbol = row["symbol"]
        exp = row["expiration"]
        strike = float(row["strike"])
        call_put = row["call_put"]
        contracts_held = float(row["contracts_held"])
        if contracts_held == 0:
            continue

        # We'll treat it as a "buy" to get the fill price near ask if net > 0
        # But strictly speaking, you'd do something more advanced for just "current price."
        # We'll do "mid" logic or so:
        # Let's just treat is_buy = True if contracts_held>0
        # This is a simplification to get a "mark price"
        is_buy = True
        try:
            current_px = fetch_option_price(symbol, exp, strike, call_put, is_buy)
        except:
            current_px = row["current_price"]  # fallback

        new_unreal = (current_px - row["avg_cost"]) * (contracts_held * 100)
        supabase.table("portfolio_options").update({
            "current_price": current_px,
            "unrealized_pl": new_unreal
        }).eq("id", row["id"]).execute()

    # Recalc performance
    calculate_and_record_performance(team_id)

def calculate_total_pnl(team_id: int) -> float:
    """Return the total realized PnL for the team."""
    act_df = load_activity(team_id)
    if act_df.empty:
        return 0.0
    return act_df["realized_pl"].sum()

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
    free_pool = original_cap - restricted_cap

    spent_shares = (shares_df["shares_held"] * shares_df["avg_cost"]).sum() if not shares_df.empty else 0.0
    spent_opts = (opts_df["contracts_held"] * 100 * opts_df["avg_cost"]).sum() if not opts_df.empty else 0.0
    used_invest = spent_shares + spent_opts
    free_cash = free_pool - used_invest

    # PnL Calculation
    realized_pl = load_activity(team_id)["realized_pl"].sum() if not load_activity(team_id).empty else 0.0
    unrealized_pl = sum_opts_val + sum_shares_val - used_invest
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
    today_str = datetime.date.today().strftime("%Y-%m-%d")

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
    if not st.session_state["logged_in"]:
        show_login_screen()
    else:
        if st.session_state["is_admin"]:
            show_admin_panel()
        else:
            show_team_portfolio()

def show_login_screen():
    st.title("Zak Fund")
    pwd = st.text_input("Enter Team Password or Admin Password:", type="password")
    if st.button("Login"):
        if pwd == ADMIN_PASSWORD:
            st.session_state["logged_in"] = True
            st.session_state["is_admin"] = True
            st.rerun()
        else:
            # check if team password
            team_row = get_team_by_password(pwd)
            if team_row:
                st.session_state["logged_in"] = True
                st.session_state["is_admin"] = False
                st.session_state["team_id"] = team_row["id"]
                # Auto-refresh prices once
                refresh_portfolio_prices(team_row["id"])
                st.rerun()
            else:
                st.error("Invalid password!")

def show_admin_panel():
    st.title("Admin Panel")
    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.session_state["is_admin"] = False
        st.session_state["team_id"] = None
        st.rerun()

    st.subheader("Create / Edit Teams")
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

    if st.button("Save Team"):
        data_dict = {
            "team_name": tname,
            "team_password": tpwd,
            "initial_capital": init_cap,
            "restricted_capital": restr_cap
        }
        upsert_team(data_dict, team_id)
        st.success("Team saved/updated!")
        st.rerun()

    st.subheader("All Teams Overview")
    # Quick summary table
    rows = []
    for t in teams_list:
        tid = t["id"]
        # Refresh current data from DB
        port_stat = compute_portfolio_stats(tid)
        shares_df = port_stat["shares_df"]
        opts_df = port_stat["opts_df"]
        sum_shares_val = port_stat["sum_shares_val"]
        sum_opts_val = port_stat["sum_opts_val"]

        free_pool = float(t["initial_capital"]) - float(t["restricted_capital"])
        free_cash = port_stat["free_cash"]
        total_val = port_stat["total_val"]

        rows.append({
            "Team": t["team_name"],
            "Initial Cap": t["initial_capital"],
            "Restricted": t["restricted_capital"],
            "Portfolio Value": total_val
        })
    df_admin = pd.DataFrame(rows)
    st.dataframe(df_admin, use_container_width=True)
def show_team_portfolio():
    team_id = st.session_state["team_id"]
    team = get_team_info(team_id)
    if not team:
        st.error("Team not found.")
        return

    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.session_state["team_id"] = None
        st.rerun()

    st.title(f"Team: {team['team_name']}")

    if st.button("Refresh Prices"):
        refresh_portfolio_prices(team_id)
        st.rerun()

    # 🔥 Use the centralized function to get portfolio data
    portfolio_stats = compute_portfolio_stats(team_id)

    shares_df = portfolio_stats["shares_df"]
    opts_df = portfolio_stats["opts_df"]

    free_cash = portfolio_stats["free_cash"]

    colA = st.columns(6)
    colA[0].number_input("Original Starting Capital ($)", value=float(team["initial_capital"]), step=1000.0, disabled=True)
    colA[1].number_input("Total Account ($)", value=portfolio_stats["total_val"], step=500.0, disabled=True)
    colA[2].number_input(
        f"Shares Portion ($) - {portfolio_stats['sum_shares_val'] / portfolio_stats['total_val'] * 100:.2f}%",
        value=portfolio_stats["sum_shares_val"], step=500.0, disabled=True
    )
    colA[3].number_input(
        f"Options Portion ($) - {portfolio_stats['sum_opts_val'] / portfolio_stats['total_val'] * 100:.2f}%",
        value=portfolio_stats["sum_opts_val"], step=500.0, disabled=True
    )
    colA[4].number_input(
        f"Free Cash ($) - {portfolio_stats['free_cash'] / portfolio_stats['total_val'] * 100:.2f}%",
        value=free_cash, step=500.0, disabled=True
    )
    colA[5].number_input("Total PnL ($)", value=portfolio_stats["total_pnl"], step=500.0, disabled=True)

    # TABS
    tab_shares, tab_opts, tab_log, tab_perf = st.tabs([
        "Shares", "Options", "Activity Log", "Performance"
    ])

    with tab_shares:
        st.subheader("Your Shares")
        if shares_df.empty:
            st.info("No share positions.")
        else:
            df_disp = shares_df.copy()
            df_disp["Position Value"] = df_disp["shares_held"] * df_disp["current_price"]
            st.dataframe(df_disp, use_container_width=True)

        st.subheader("Trade Shares")
        col1, col2 = st.columns(2)
        ticker_in = col1.text_input("Ticker (e.g. AAPL, SPY)")
        px = fetch_share_price(ticker_in)
        qty_in = col2.number_input("Shares to Buy(+) or Sell(-)", step=1.0)

        st.write(f"Current Price: ${px:,.2f}")
        if st.button("Execute Share Trade"):
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

    with tab_opts:

        # -------------------------
        # Options Trading Section
        # -------------------------

        # Step 1: Select an Existing Position or Search for a New Trade
        st.markdown("### Select Existing Position or Find a New Trade")

        existing_opts = load_options(st.session_state["team_id"])
        selected_existing = pd.DataFrame()
        selected_calls = pd.DataFrame()
        selected_puts = pd.DataFrame()
        # 🔹 User selects one or multiple existing positions for closing (supports spreads)
        if not existing_opts.empty:
            existing_opts.rename(columns={'symbol': 'Symbol', 'expiration': 'Expiration', 'strike': 'Strike', 'call_put': 'Type',
                                        }, inplace=True)
            st.markdown("#### Existing Positions (Select to Close)")
            gb_exist = GridOptionsBuilder.from_dataframe(existing_opts.drop(columns=["id", "team_id"]))
            gb_exist.configure_selection(selection_mode="multiple", use_checkbox=True)  # Multi-leg support
            grid_options_exist = gb_exist.build()
            grid_response_exist = AgGrid(
                existing_opts,
                gridOptions=grid_options_exist,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                height=250,
                fit_columns_on_grid_load=True
            )
            selected_existing = pd.DataFrame(grid_response_exist.get("selected_rows", []))
        else:
            st.info("No existing option positions.")
        # 🔹 If no existing position is selected, allow the user to enter a new symbol
        selected_symbol = st.text_input("Symbol for Chain (e.g. SPY)", key="chain_symbol").upper()

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

                    # Display Calls and Puts Side-by-Side
                    col_calls, col_puts = st.columns(2)

                    with col_calls:
                        st.markdown("#### Calls")
                        gb_calls = GridOptionsBuilder.from_dataframe(df_calls)
                        gb_calls.configure_default_column(
                            resizable=True, autoWidth=True, wrapText=True
                        )

                        gb_calls.configure_selection(selection_mode="multiple", use_checkbox=True)  # Multi-leg support
                        grid_options_calls = gb_calls.build()
                        grid_response_calls = AgGrid(
                            df_calls,
                            gridOptions=grid_options_calls,
                            update_mode=GridUpdateMode.SELECTION_CHANGED,
                            height=250,
                            fit_columns_on_grid_load=True
                        )
                        selected_calls = pd.DataFrame(grid_response_calls.get("selected_rows", []))

                    with col_puts:
                        st.markdown("#### Puts")
                        gb_puts = GridOptionsBuilder.from_dataframe(df_puts)
                        gb_puts.configure_default_column(
                            resizable=True, autoWidth=True, wrapText=True
                        )
                        gb_puts.configure_selection(selection_mode="multiple", use_checkbox=True)  # Multi-leg support
                        grid_options_puts = gb_puts.build()
                        grid_response_puts = AgGrid(
                            df_puts,
                            gridOptions=grid_options_puts,
                            update_mode=GridUpdateMode.SELECTION_CHANGED,
                            height=250,
                            fit_columns_on_grid_load=True
                        )
                        selected_puts = pd.DataFrame(grid_response_puts.get("selected_rows", []))

        # Step 2: Trade Details Input
        trade_details = []

        if not selected_existing.empty or not selected_calls.empty or not selected_puts.empty:
            st.markdown("### Specify Trade Details")

            selected_legs = pd.concat([selected_existing, selected_calls, selected_puts], ignore_index=True)
            for idx, leg in selected_legs.iterrows():
                with st.container():
                    col1, col2, col3, col4 = st.columns(4)
                    col1.subheader(f"""**{leg['Symbol']} {leg['Strike']} {leg['Type']}**\n  \n {f"**{leg['Bid']}B - {leg['Ask']}A**" if
                                    'Bid' in leg and 'Ask' in leg else leg['avg_cost']}""")

                    action = col2.selectbox("Action", options=["Buy", "Sell"], key=f"action_{idx}")
                    qty = col3.number_input("Contracts", value=abs(leg.get("contracts_held", 1)), min_value=1, step=1, key=f"qty_{idx}")
                    fill_price = col4.number_input("Estimated Fill Price",value=fetch_option_price(leg["Symbol"], leg["Expiration"], leg["Strike"], leg["Type"], is_buy=(action == "Buy")),  disabled=True, key=f"price_{idx}")

                    trade_details.append({
                        "Type": leg["Type"],
                        "Strike": leg["Strike"],
                        "Action": action,
                        "Qty": qty,
                        "Fill Price": fill_price,
                        "Symbol": leg["Symbol"],
                        "Expiration": leg["Expiration"]
                    })

            # Step 3: Execute Trade
            if st.button("Execute Trade") and trade_details:
                if all(detail["Qty"] == 0 for detail in trade_details):
                    st.warning("Please specify a nonzero quantity for at least one leg.")
                else:
                    closing_positions = []
                    opening_positions = []

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
                        if action == "Buy":
                            if old_contracts < 0:
                                closing_positions.append(detail)
                            else:
                                opening_positions.append(detail)

                        elif action == "Sell":
                            detail["Qty"] = -detail["Qty"]
                            if old_contracts > 0:
                                closing_positions.append(detail)
                            else:
                                opening_positions.append(detail)

                    # 🚨 Ensure New Trades Are Debit Only
                    total_opening_cost = sum(d["Fill Price"] * d["Qty"] * 100 for d in opening_positions)
                    total_closing_credit = sum(d["Fill Price"] * d["Qty"] * 100 for d in closing_positions)
                    st.write(total_closing_credit, total_opening_cost)
                    if total_closing_credit > total_opening_cost:
                        st.error("You are trying to execute a **credit** trade. Only **debit** trades are allowed.")
                        st.stop()

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
                        st.write(trade_qty)
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
                            log_options_activity(st.session_state["team_id"], tradeAction.SELL, detail['Symbol'], detail["Type"],
                                                detail['Expiration'], detail["Strike"], -trade_qty, fill_price, realized_pl=realized)

                    st.success("Trade executed successfully.")
                    calculate_and_record_performance(st.session_state["team_id"])
                    st.rerun()


    with tab_log:
        st.subheader("Activity Log")
        act_df = load_activity(team_id)
        if act_df.empty:
            st.info("No recent activity.")
        else:
            for _, row in act_df.iterrows():
                st.markdown(f"• {row['message']}", unsafe_allow_html=True)

    with tab_perf:
        st.subheader("Performance Over Time")
        perf_df = load_performance(team_id)
        if perf_df.empty:
            st.info("No performance records yet.")
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
