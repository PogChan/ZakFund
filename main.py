import streamlit as st
import subprocess
import sys

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
st.set_page_config(page_title="Zak Fund", layout="wide")

class tradeAction:
    BUY = "BUY"
    SELL = "SELL"
    CLOSE = "CLOSE"
    SPREAD = "SPREAD"

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

@st.cache_data(ttl=60*60)
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
    action = "BOUGHT" if contracts_added > 0 else "SOLD"
    color = "#65FE08" if contracts_added > 0 else "red"
    sign = "+" if contracts_added > 0 else ""

    colorNet = "red" if contracts_added > 0 else "#65FE08"
    signNet = "-" if contracts_added > 0 else "+"

    total_cost = price * abs(contracts_added) * 100
    now_str = get_est_time()
    exp_str = expiration if isinstance(expiration, str) else expiration.strftime("%Y-%m-%d")

    msg = (
        f"<b style='color:{color};'>{action} {sign}{contracts_added} contract(s)</b> of "
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
    st.title("Multi-Team Paper Trading Login")
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
                    else:
                        upsert_shares(team_id, ticker_in, leftover, old_avg, px, row_id=row_id)
                        log_shares_activity(team_id,tradeAction.SELL, ticker_in, qty_in, px, realized_pl=realized)

                refresh_portfolio_prices(team_id)
                st.rerun()

    with tab_opts:
        st.subheader("Your Options")
        if opts_df.empty:
            st.info("No options yet.")
        else:
            st.dataframe(opts_df, use_container_width=True)

        # -------------------------
        # Options Trading Section
        # -------------------------
        st.subheader("Options Trading")

        # Let the user choose the trade mode:
        trade_mode = st.radio("Select Trade Mode:", options=["Close Existing Position", "Open New Trade"])

        if trade_mode == "Close Existing Position":
            st.markdown("### Existing Option Positions")
            existing_opts = load_options(st.session_state["team_id"])
            if existing_opts.empty:
                st.info("No existing option positions to close out.")
            else:
                # Allow single row selection (only one position can be closed out at a time)
                gb_exist = GridOptionsBuilder.from_dataframe(existing_opts)
                gb_exist.configure_selection(selection_mode="single", use_checkbox=True)
                grid_options_exist = gb_exist.build()
                grid_response_exist = AgGrid(
                    existing_opts,
                    gridOptions=grid_options_exist,
                    update_mode=GridUpdateMode.SELECTION_CHANGED,
                    height=250,
                    fit_columns_on_grid_load=True
                )
                selected_existing = pd.DataFrame(grid_response_exist.get("selected_rows", []))
                if not selected_existing.empty:
                    sel = selected_existing.iloc[0]
                    st.success(f"Selected Position: {sel['symbol']} {sel['call_put']} {sel['strike']} exp {sel['expiration']} | Holding {sel['contracts_held']} contracts")
                    # In Close mode, only selling is allowed (i.e. reducing or closing the position)
                    sell_qty = st.number_input("Contracts to Sell", max_value=-1, step=-1, value = sel['contracts_held'] * -1)
                    sell_qty = abs(sell_qty)
                    if st.button("Close/Reduce Position"):
                        current_px = fetch_option_price(sel["symbol"], sel["expiration"], float(sel["strike"]), sel["call_put"], is_buy=False)
                        # For closing, ensure that the sell quantity does not exceed current holding
                        if sell_qty > sel["contracts_held"]:
                            st.error("Cannot sell more than currently held.")
                        else:
                            # Compute realized P/L
                            realized = (current_px - sel["avg_cost"]) * sell_qty * 100
                            leftover = sel["contracts_held"] - sell_qty
                            if leftover == 0:
                                log_options_activity(st.session_state["team_id"], tradeAction.CLOSE, sel["symbol"], sel["call_put"],
                                                sel["expiration"], float(sel["strike"]), -sell_qty, current_px, realized_pl=realized)
                                delete_option_by_id(sel["id"])
                            else:
                                upsert_option(sel["id"], st.session_state["team_id"], sel["symbol"], sel["call_put"],
                                            sel["expiration"], float(sel["strike"]), leftover, sel["avg_cost"], current_px)
                                log_options_activity(st.session_state["team_id"], tradeAction.SELL, sel["symbol"], sel["call_put"],
                                                sel["expiration"], float(sel["strike"]), -sell_qty, current_px, realized_pl=realized)
                            st.success(f"Closed {sell_qty} contracts. Realized P/L: ${realized:,.2f}")
                            calculate_and_record_performance(st.session_state["team_id"])
                            st.rerun()

        elif trade_mode == "Open New Trade":
            st.markdown("### Options Chain (Multi-Selection)")
            symbol_chain = st.text_input("Symbol for Chain (e.g. SPY)", key="chain_symbol")
            if symbol_chain:
                symbol_chain = symbol_chain.upper()
                chain_data = get_options_chain(symbol_chain)
                if chain_data and "options" in chain_data:
                    exps = sorted(chain_data["options"].keys())
                    chosen_exp = st.selectbox("Expiration", exps, key="chain_exp")
                    if chosen_exp:
                        stock_price = fetch_share_price(symbol_chain)
                        # Build separate DataFrames for calls and puts with moneyness info:
                        calls_list = []
                        puts_list = []
                        for strike_str, sdata in chain_data["options"][chosen_exp].get("c", {}).items():
                            strike = float(strike_str)
                            calls_list.append({
                                "Type": "CALL",
                                "Strike": strike,
                                "Bid": sdata.get("b", 0),
                                "Ask": sdata.get("a", 0),
                                "OI": sdata.get("oi", 0),
                                "Volume": sdata.get("v", 0),
                                "Moneyness": ( "ITM" if stock_price > strike else "OTM" if stock_price < strike else "ATM")
                            })
                        for strike_str, sdata in chain_data["options"][chosen_exp].get("p", {}).items():
                            strike = float(strike_str)
                            puts_list.append({
                                "Type": "PUT",
                                "Strike": strike,
                                "Bid": sdata.get("b", 0),
                                "Ask": sdata.get("a", 0),
                                "OI": sdata.get("oi", 0),
                                "Volume": sdata.get("v", 0),
                                "Moneyness": ( "OTM" if stock_price > strike else "ITM" if stock_price < strike else "ATM")
                            })
                        df_calls = pd.DataFrame(calls_list)
                        df_puts = pd.DataFrame(puts_list)

                        # Display side-by-side: Calls in one column, Puts in the other.
                        col_calls, col_puts = st.columns(2)
                        with col_calls:
                            st.markdown("#### Calls")
                            gb_calls = GridOptionsBuilder.from_dataframe(df_calls)
                            gb_calls.configure_selection(selection_mode="multiple", use_checkbox=True)
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
                            gb_puts.configure_selection(selection_mode="multiple", use_checkbox=True)
                            grid_options_puts = gb_puts.build()
                            grid_response_puts = AgGrid(
                                df_puts,
                                gridOptions=grid_options_puts,
                                update_mode=GridUpdateMode.SELECTION_CHANGED,
                                height=250,
                                fit_columns_on_grid_load=True
                            )
                            selected_puts = pd.DataFrame(grid_response_puts.get("selected_rows", []))

                        # Combine the selections (if any)
                        if not selected_calls.empty or not selected_puts.empty:
                            selected_options = pd.concat([selected_calls, selected_puts], ignore_index=True)


                            # Now, for each selected leg, let the user specify independent trade details:
                            st.markdown(f"#### Specify Trade Details for {chosen_exp}")
                            trade_details = []
                            for idx, leg in selected_options.iterrows():
                                # Group each leg’s inputs in its own container.
                                with st.container():
                                    col1, col2, col3, col4 = st.columns(4)
                                    col1.subheader(f"**{symbol_chain} {leg['Strike']} {leg['Type']}**\n  \n **{leg['Bid']}B - {leg['Ask']}A**")
                                    action = col2.selectbox("Action", options=["Buy", "Sell"], key=f"action_{idx}")
                                    qty = col3.number_input("Contracts", value = 0, min_value=0, step=1, key=f"qty_{idx}")

                                    # Optionally, allow a manual override for fill price; else use default:
                                    fill_price = col4.number_input("Fill Price", value=fetch_option_price(symbol_chain, chosen_exp, leg['Strike'], leg['Type'], True if qty > 0 else False), key=f"price_{idx}", disabled=True)

                                    trade_details.append({
                                        "Type": leg["Type"],
                                        "Strike": leg["Strike"],
                                        "Action": action,
                                        "Qty": qty,
                                        "Fill Price": fill_price
                                    })
                            # Execute the trade when the user clicks:
                            if st.button("Execute Trade"):
                                # Ensure at least one leg has a nonzero quantity:
                                if all(detail["Qty"] == 0 for detail in trade_details):
                                    st.warning("Please specify a nonzero quantity for at least one leg.")
                                elif all(detail["Action"] == "Sell" for detail in trade_details):
                                    st.error("At least one leg must be a buy.")
                                else:
                                    net_cost = 0.0
                                    # Compute net cost across legs.
                                    for detail in trade_details:
                                        leg_cost = detail["Fill Price"] * detail["Qty"] * 100
                                        # For buys, cost is added; for sells, cost is subtracted.
                                        if detail["Action"] == "Buy":
                                            net_cost += leg_cost
                                        else:
                                            net_cost -= leg_cost
                                    if net_cost <= 0:
                                        st.error("Only net debit trades are allowed.")
                                    else:

                                        # For each leg, process the trade.
                                        opts_df = load_options(st.session_state["team_id"])
                                        for detail in trade_details:
                                            # Retrieve existing position for this leg if any:
                                            existing = pd.DataFrame() if opts_df.empty else opts_df[
                                                (opts_df["symbol"] == symbol_chain.upper()) &
                                                (opts_df["call_put"] == detail["Type"]) &
                                                (opts_df["strike"] == detail["Strike"]) &
                                                (opts_df["expiration"] == chosen_exp)
                                            ]
                                            if existing.empty:
                                                old_contracts = 0.0
                                                old_avg = 0.0
                                                opt_id = None
                                            else:
                                                old_contracts = float(existing["contracts_held"].iloc[0])
                                                old_avg = float(existing["avg_cost"].iloc[0])
                                                opt_id = existing["id"].iloc[0]

                                            if detail["Action"] == "Buy":
                                                new_total = old_contracts + detail["Qty"]
                                                # Weighted average for new position:
                                                new_avg = (old_contracts * old_avg + detail["Qty"] * detail["Fill Price"]) / new_total if new_total != 0 else detail["Fill Price"]
                                                upsert_option(opt_id, st.session_state["team_id"], symbol_chain, detail["Type"], chosen_exp, detail["Strike"], new_total, new_avg, detail["Fill Price"])
                                                log_options_activity(st.session_state["team_id"], tradeAction.BUY ,symbol_chain, detail["Type"], chosen_exp, detail["Strike"], detail["Qty"], detail["Fill Price"], realized_pl=0)
                                            else:  # Sell
                                                # In new trades, selling should be part of a multi-leg spread;
                                                # if selling, check that you have enough contracts.
                                                sell_amt = detail["Qty"]
                                                if old_contracts > 0 and sell_amt > old_contracts:
                                                    st.error(f"Not enough contracts to sell for leg {detail}")
                                                    st.stop()
                                                leftover = old_contracts - sell_amt
                                                if old_contracts > 0:
                                                    realized = (detail["Fill Price"] - old_avg) * sell_amt * 100
                                                else:
                                                    realized = 0
                                                    old_avg = (old_contracts * old_avg + sell_amt * detail["Fill Price"]) / leftover

                                                if leftover == 0:
                                                    if opt_id is not None:
                                                        log_options_activity(st.session_state["team_id"], tradeAction.CLOSE, symbol_chain, detail["Type"],
                                                                             chosen_exp, detail["Strike"], -sell_amt, detail["Fill Price"], realized_pl=realized)
                                                        delete_option_by_id(opt_id)
                                                else:
                                                    upsert_option(opt_id, st.session_state["team_id"], symbol_chain, detail["Type"], chosen_exp, detail["Strike"], leftover, old_avg, detail["Fill Price"])
                                                    log_options_activity(st.session_state["team_id"], tradeAction.SELL, symbol_chain, detail["Type"], chosen_exp, detail["Strike"], -sell_amt, detail["Fill Price"], realized_pl=realized)
                                        st.success(f"Spread trade executed with net cost: ${net_cost:,.2f}")
                                        calculate_and_record_performance(st.session_state["team_id"])
                                        st.rerun()
                        else:
                            st.info("No legs selected. Please select at least one option from Calls or Puts.")
                else:
                    st.error("No options chain data available for the symbol.")



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
