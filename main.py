import streamlit as st
import subprocess
import sys

@st.cache_resource
def install_latest_yfinance():
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "yfinance"], check=True)

install_latest_yfinance()

import time
import random
import datetime
import pandas as pd
import yfinance as yf
import cloudscraper
import altair as alte

from supabase import create_client, Client

# -----------------------------------------------------------------------------
# Streamlit / Supabase Setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Zak Fund", layout="wide")
url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]
ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]

supabase: Client = create_client(url, key)

# For demonstration, store session states:
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False
if "team_id" not in st.session_state:
    st.session_state["team_id"] = None
if "selected_option" not in st.session_state:
    st.session_state["selected_option"] = None  # for the chain UI (call/put selection)

# -----------------------------------------------------------------------------
# Database Helpers
# -----------------------------------------------------------------------------
def get_team_by_password(pwd: str):
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

# -----------------------------------------------------------------------------
# Portfolio CRUD
# -----------------------------------------------------------------------------
def load_shares(team_id: int) -> pd.DataFrame:
    resp = supabase.table("portfolio_shares").select("*").eq("team_id", team_id).execute()
    return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()

def upsert_shares(team_id: int, ticker: str, shares_held: float, avg_cost: float, current_price: float):
    unreal_pl = (current_price - avg_cost) * shares_held
    supabase.table("portfolio_shares").upsert({
        "team_id": team_id,
        "ticker": ticker.upper(),
        "shares_held": shares_held,
        "avg_cost": avg_cost,
        "current_price": current_price,
        "unrealized_pl": unreal_pl
    }, on_conflict="team_id,ticker").execute()

def delete_shares(team_id: int, ticker: str):
    supabase.table("portfolio_shares").delete().match({"team_id": team_id, "ticker": ticker}).execute()

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

def delete_option(row_id: int):
    supabase.table("portfolio_options").delete().eq("id", row_id).execute()

def load_performance(team_id: int) -> pd.DataFrame:
    resp = supabase.table("performance").select("*").eq("team_id", team_id).execute()
    return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()

def upsert_performance(team_id: int, date_str: str, total_value: float):
    supabase.table("performance").upsert({
        "team_id": team_id,
        "date": date_str,
        "total_value": total_value
    }, on_conflict="team_id,date").execute()

def load_activity(team_id: int) -> pd.DataFrame:
    resp = (supabase.table("portfolio_activity")
            .select("*")
            .eq("team_id", team_id)
            .order("id", desc=True)
            .limit(25)
            .execute())
    return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()

def log_activity(team_id: int, message: str, realized_pl: float = 0):
    supabase.table("portfolio_activity").insert({
        "team_id": team_id,
        "message": message,
        "realized_pl": realized_pl
    }).execute()

# -----------------------------------------------------------------------------
# Price Fetching
# -----------------------------------------------------------------------------
def fetch_share_price(ticker: str) -> float:
    try:
        data = yf.Ticker(ticker).history(period="1d")
        if len(data) > 0:
            return float(data["Close"].iloc[-1])
    except:
        st.error(f"❌ Failed Fetching {ticker} Price")
    return 0.0

# Use your custom baseURL from secrets
baseURL = st.secrets["BASEAPI"]

@st.cache_data(ttl=60*60)
def get_options_chain(symbol: str):
    """
    Returns JSON with structure:
    {
      "options": {
         "2025-02-28": {
             "c": { "20.00": { "oi":..., "b":..., "a":..., "v":...}, "22.00": {...} },
             "p": { ... }
         },
         "2025-03-14": { ... }
      }
    }
    """
    time.sleep(1)  # simulate some network delay
    full_url = f"{baseURL}?stock={symbol.upper()}&reqId={random.randint(1, 1000000)}"
    scraper = cloudscraper.create_scraper()
    response = scraper.get(full_url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"❌ Failed to fetch options chain for {symbol}. Status code: {response.status_code}")
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
        # narrower spread => buy near ask, sell near bid
        if is_buy:
            return bid + 0.6 * spread
        else:
            return bid + 0.4 * spread

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
        raise ValueError(f"No {call_put} data found for {expiration} in chain.")

    strike_key = f"{strike:.2f}"
    if strike_key not in cp_dict:
        raise ValueError(f"Strike {strike} not found for {call_put} {expiration} {symbol}.")

    option_data = cp_dict[strike_key]
    bid = option_data.get("b", 0)
    ask = option_data.get("a", 0)
    fill_price = compute_option_fill_price(bid, ask, is_buy)
    return fill_price

# -----------------------------------------------------------------------------
# Activity Logging with HTML
# -----------------------------------------------------------------------------
def get_est_time():
    now_est = datetime.datetime.now()
    return now_est.strftime("%m/%d/%Y %I:%M%p")

def log_shares_activity(team_id: int, ticker: str, shares_added: float, price: float, realized_pl=0):
    action = "BOUGHT" if shares_added > 0 else "SOLD"
    color = "#65FE08" if shares_added > 0 else "red"
    sign = "+" if shares_added > 0 else ""
    cost = price * shares_added
    now_str = get_est_time()

    msg = (
        f"<b style='color:{color};'>{action} {sign}{shares_added} shares</b> "
        f"of <b style='color:#FFD700;'>{ticker}</b> "
        f"at <b>/${price:,.2f}</b> "
        f"(Total: <b style='color:{color};'>{sign}${abs(cost):,.2f}</b>) "
        f"on {now_str}"
    )
    log_activity(team_id, msg, realized_pl=realized_pl)

def log_options_activity(team_id: int, symbol: str, call_put: str, expiration, strike: float, contracts_added: float, price: float, realized_pl=0):
    action = "BOUGHT" if contracts_added > 0 else "SOLD"
    color = "#65FE08" if contracts_added > 0 else "red"
    sign = "+" if contracts_added > 0 else ""
    total_cost = price * abs(contracts_added) * 100
    now_str = get_est_time()
    exp_str = expiration if isinstance(expiration, str) else expiration.strftime("%Y-%m-%d")

    msg = (
        f"<b style='color:{color};'>{action} {sign}{abs(contracts_added)} contract(s)</b> of "
        f"<b style='color:#FFD700;'>{symbol} {strike:.2f} {call_put} {exp_str}</b> "
        f"at <b>/${price:,.2f}</b> "
        f"(Total: <b style='color:{color};'>{sign}${total_cost:,.2f}</b>) "
        f"on {now_str}"
    )
    log_activity(team_id, msg, realized_pl=realized_pl)

# -----------------------------------------------------------------------------
# Calculate Performance
# -----------------------------------------------------------------------------
def calculate_and_record_performance(team_id: int):
    team = get_team_info(team_id)
    if not team:
        return

    shares_df = load_shares(team_id)
    opts_df = load_options(team_id)

    # Basic approach: "buying_power" = team['initial_capital'] - restricted_capital - cost_of_positions
    # or you can do it in real-time. For simplicity, let's do:
    # available_cap = team['initial_capital'] - team['restricted_capital']
    # invested in shares:
    total_shares_val = 0.0
    if not shares_df.empty:
        total_shares_val = (shares_df["shares_held"] * shares_df["current_price"]).sum()

    # invested in options:
    total_opts_val = 0.0
    if not opts_df.empty:
        total_opts_val = (opts_df["contracts_held"] * 100 * opts_df["current_price"]).sum()

    # We'll define "portfolio_value" = restricted_capital + total_shares_val + total_opts_val + free_cash
    # but we need "free_cash" = initial_capital - restricted_capital - sum_of_invested
    # sum_of_invested_shares = shares_df["shares_held"] * shares_df["avg_cost"] ...
    sum_shares_spent = 0.0
    if not shares_df.empty:
        sum_shares_spent = (shares_df["shares_held"] * shares_df["avg_cost"]).sum()

    sum_opts_spent = 0.0
    if not opts_df.empty:
        sum_opts_spent = (opts_df["contracts_held"] * 100 * opts_df["avg_cost"]).sum()

    # The "cash" left from the portion that is not restricted:
    initial_cap = float(team["initial_capital"])
    restricted_cap = float(team["restricted_capital"])
    free_cap_pool = initial_cap - restricted_cap  # how much is available for investing
    spent_total = sum_shares_spent + sum_opts_spent
    free_cash = max(0, free_cap_pool - spent_total)

    # total_value = restricted + free_cash + value_of_shares + value_of_options
    total_value = restricted_cap + free_cash + total_shares_val + total_opts_val

    today_str = datetime.date.today().strftime("%Y-%m-%d")
    upsert_performance(team_id, today_str, total_value)

# -----------------------------------------------------------------------------
# Main App
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
            st.experimental_rerun()
        else:
            # check if team password
            team_row = get_team_by_password(pwd)
            if team_row:
                st.session_state["logged_in"] = True
                st.session_state["is_admin"] = False
                st.session_state["team_id"] = team_row["id"]
                st.experimental_rerun()
            else:
                st.error("Invalid password!")

def show_admin_panel():
    st.title("Admin Panel")
    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.session_state["is_admin"] = False
        st.session_state["team_id"] = None
        st.experimental_rerun()

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
        team_id = existing["id"]
        def_name = existing["team_name"]
        def_pwd = existing["team_password"]
        def_init = float(existing["initial_capital"])
        def_restrict = float(existing["restricted_capital"])

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
        st.experimental_rerun()

    st.subheader("Team Performance Overview")
    # We'll show # trades, sum of realized_pl, current total_value, etc.
    summary_data = []
    for t in teams_list:
        tid = t["id"]
        # load activity to see trades, realized PL
        act_df = (supabase.table("portfolio_activity")
                  .select("*")
                  .eq("team_id", tid)
                  .execute())
        act_df = pd.DataFrame(act_df.data) if act_df.data else pd.DataFrame()

        total_trades = 0
        wins = 0
        total_realized = 0.0
        if not act_df.empty:
            # count how many trades have realized_pl != 0
            trades_df = act_df[act_df["realized_pl"] != 0]
            total_trades = len(trades_df)
            total_realized = trades_df["realized_pl"].sum()
            wins = len(trades_df[trades_df["realized_pl"] > 0])

        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

        # current portfolio value (from performance or real-time)
        # simplest approach: re-calc real-time or check last performance
        shares_df = load_shares(tid)
        opts_df = load_options(tid)
        sum_shares_val = (shares_df["shares_held"] * shares_df["current_price"]).sum() if not shares_df.empty else 0
        sum_opts_val = (opts_df["contracts_held"] * 100 * opts_df["current_price"]).sum() if not opts_df.empty else 0
        # see function above for your "free_cash" logic if needed
        total_val = sum_shares_val + sum_opts_val + (t["initial_capital"] - t["restricted_capital"]) - \
                    (shares_df["shares_held"] * shares_df["avg_cost"]).sum() - \
                    ((opts_df["contracts_held"] * 100) * opts_df["avg_cost"]).sum() + \
                    t["restricted_capital"]

        summary_data.append({
            "Team": t["team_name"],
            "Trades": total_trades,
            "Win Rate (%)": f"{win_rate:.1f}",
            "Realized P/L": f"${total_realized:,.2f}",
            "Current Value": f"${total_val:,.2f}",
            "Restricted": f"${t['restricted_capital']:,.2f}"
        })

    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True)

def show_team_portfolio():
    team_id = st.session_state["team_id"]
    team_info = get_team_info(team_id)
    if not team_info:
        st.error("Team not found.")
        return

    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.session_state["team_id"] = None
        st.experimental_rerun()

    st.title(f"Team: {team_info['team_name']}")

    # Load data
    shares_df = load_shares(team_id)
    opts_df = load_options(team_id)

    # Calculate current total portfolio value:
    # see function or do quick approach
    sum_shares_val = (shares_df["shares_held"] * shares_df["current_price"]).sum() if not shares_df.empty else 0
    sum_opts_val = (opts_df["contracts_held"] * 100 * opts_df["current_price"]).sum() if not opts_df.empty else 0
    spent_shares = (shares_df["shares_held"] * shares_df["avg_cost"]).sum() if not shares_df.empty else 0
    spent_opts = (opts_df["contracts_held"] * 100 * opts_df["avg_cost"]).sum() if not opts_df.empty else 0

    free_cap_pool = team_info["initial_capital"] - team_info["restricted_capital"]
    free_cash = max(0, free_cap_pool - (spent_shares + spent_opts))
    total_val = sum_shares_val + sum_opts_val + free_cash + team_info["restricted_capital"]

    st.write(f"**Initial Capital**: ${team_info['initial_capital']:,.2f}")
    st.write(f"**Restricted Capital**: ${team_info['restricted_capital']:,.2f}")
    st.write(f"**Free Cash**: ${free_cash:,.2f}")
    st.write(f"**Portfolio Value**: ${total_val:,.2f}")

    tab_shares, tab_opts, tab_log, tab_perf = st.tabs(["Shares", "Options", "Activity Log", "Performance"])

    with tab_shares:
        st.subheader("Your Shares")
        if shares_df.empty:
            st.info("No shares yet.")
        else:
            df_disp = shares_df.copy()
            df_disp["Position Value"] = df_disp["shares_held"] * df_disp["current_price"]
            st.dataframe(df_disp, use_container_width=True)

        st.subheader("Trade Shares")
        col1, col2 = st.columns(2)
        ticker = col1.text_input("Ticker (e.g. AAPL, TSLA, SPY)")
        qty = col2.number_input("Shares to Buy (+) or Sell (-)", step=1.0)
        if st.button("Execute Share Trade"):
            if qty == 0:
                st.warning("No shares specified.")
            else:
                # fetch current price
                px = fetch_share_price(ticker)
                if px <= 0:
                    st.error("Could not fetch price or price is 0.")
                    return

                # load existing row
                row = shares_df[shares_df["ticker"] == ticker.upper()]
                old_shares = float(row["shares_held"].iloc[0]) if not row.empty else 0
                old_avg = float(row["avg_cost"].iloc[0]) if not row.empty else 0

                if qty > 0:
                    # Buying => check free_cash
                    cost = px * qty
                    if cost > free_cash:
                        st.error("Not enough free cash.")
                        return
                    # Weighted average cost for new total
                    new_shares = old_shares + qty
                    new_avg = 0
                    if new_shares != 0:
                        new_avg = (old_shares * old_avg + qty * px) / new_shares
                    upsert_shares(team_id, ticker, new_shares, new_avg, px)
                    # Realized P/L = 0 if buy
                    log_shares_activity(team_id, ticker, qty, px, realized_pl=0)
                else:
                    # Selling => can we sell that many?
                    sell_amount = abs(qty)
                    if sell_amount > old_shares:
                        st.error("You don't hold enough shares to sell.")
                        return
                    # Realized P/L on the portion sold
                    realized = (px - old_avg) * sell_amount
                    new_shares = old_shares - sell_amount
                    # If new_shares == 0 => remove row, else keep avg cost the same
                    if new_shares == 0:
                        delete_shares(team_id, ticker.upper())
                    else:
                        # avg cost stays the same for the leftover
                        upsert_shares(team_id, ticker, new_shares, old_avg, px)
                    log_shares_activity(team_id, ticker, qty, px, realized_pl=realized)

                calculate_and_record_performance(team_id)
                st.success("Trade executed.")
                st.experimental_rerun()

    with tab_opts:
        st.subheader("Your Options")
        if opts_df.empty:
            st.info("No options yet.")
        else:
            st.dataframe(opts_df, use_container_width=True)

        st.subheader("Options Chain UI (Select an Option)")
        symbol_in = st.text_input("Symbol for Chain (e.g. SPY)")
        if symbol_in:
            chain_data = get_options_chain(symbol_in)
            if chain_data and "options" in chain_data:
                exps = sorted(chain_data["options"].keys())
                chosen_exp = st.selectbox("Expiration", exps)
                if chosen_exp:
                    calls_dict = chain_data["options"][chosen_exp].get("c", {})
                    puts_dict = chain_data["options"][chosen_exp].get("p", {})

                    # Build a DF for calls
                    call_rows = []
                    for skey, sdata in calls_dict.items():
                        strike_f = float(skey)
                        row = {
                            "Type": "CALL",
                            "Strike": strike_f,
                            "Bid": sdata.get("b", 0),
                            "Ask": sdata.get("a", 0),
                            "Volume": sdata.get("v", 0),
                            "OI": sdata.get("oi", 0)
                        }
                        call_rows.append(row)
                    df_calls = pd.DataFrame(call_rows)

                    put_rows = []
                    for skey, sdata in puts_dict.items():
                        strike_f = float(skey)
                        row = {
                            "Type": "PUT",
                            "Strike": strike_f,
                            "Bid": sdata.get("b", 0),
                            "Ask": sdata.get("a", 0),
                            "Volume": sdata.get("v", 0),
                            "OI": sdata.get("oi", 0)
                        }
                        put_rows.append(row)
                    df_puts = pd.DataFrame(put_rows)

                    st.markdown("**Calls**")
                    if not df_calls.empty:
                        st.dataframe(df_calls, use_container_width=True)
                    else:
                        st.write("No Call data")

                    st.markdown("**Puts**")
                    if not df_puts.empty:
                        st.dataframe(df_puts, use_container_width=True)
                    else:
                        st.write("No Put data")

                    st.write("*(You could add 'select' buttons to autofill strike/call_put, etc.)*")

        st.subheader("Trade an Option (Manual Selection)")
        sym = st.text_input("Option Symbol (e.g. SPY) for trade")
        exp = st.text_input("Expiration YYYY-MM-DD")
        cp = st.selectbox("Call/Put", ["CALL", "PUT"])
        strike_in = st.number_input("Strike", step=1.0)
        contracts_in = st.number_input("Contracts to Buy (+) or Sell (-)", step=1.0)

        if st.button("Execute Option Trade"):
            if contracts_in == 0:
                st.warning("No contracts specified.")
            else:
                is_buy = (contracts_in > 0)
                fill_px = fetch_option_price(sym, exp, strike_in, cp, is_buy)
                if fill_px <= 0:
                    st.error("Option fill price invalid or zero.")
                    return

                # cost
                cost = fill_px * abs(contracts_in) * 100
                # check free_cash if buy
                if is_buy and cost > free_cash:
                    st.error("Not enough free cash.")
                    return

                # find existing position
                odf = opts_df[
                    (opts_df["symbol"] == sym.upper()) &
                    (opts_df["call_put"] == cp) &
                    (opts_df["strike"] == strike_in) &
                    (opts_df["expiration"] == exp)
                ]
                if odf.empty:
                    old_contracts = 0.0
                    old_avg = 0.0
                    opt_id = None
                else:
                    old_contracts = float(odf["contracts_held"].iloc[0])
                    old_avg = float(odf["avg_cost"].iloc[0])
                    opt_id = odf["id"].iloc[0]

                new_total = old_contracts + contracts_in
                if new_total < 0:
                    st.error("Cannot go negative contracts.")
                    return

                realized_pnl = 0.0
                if contracts_in > 0:
                    # Buying => no realized P/L
                    if new_total == 0:
                        # edge case: buy the exact negative?
                        delete_option(opt_id)
                    else:
                        new_avg = 0
                        if new_total != 0:
                            new_avg = (old_contracts * old_avg + contracts_in * fill_px) / new_total
                        upsert_option(opt_id, team_id, sym, cp, exp, strike_in, new_total, new_avg, fill_px)
                else:
                    # Selling
                    # Realized P/L if we reduce an existing long position
                    sell_amount = abs(contracts_in)
                    if sell_amount > old_contracts:
                        st.error("You don't hold enough contracts to sell.")
                        return
                    # realized = (sell_px - old_avg) * sold_contracts * 100
                    realized_pnl = (fill_px - old_avg) * sell_amount * 100
                    leftover = old_contracts - sell_amount
                    if leftover == 0:
                        delete_option(opt_id)
                    else:
                        # leftover avg cost stays same
                        upsert_option(opt_id, team_id, sym, cp, exp, strike_in, leftover, old_avg, fill_px)

                # log
                log_options_activity(team_id, sym, cp, exp, strike_in, contracts_in, fill_px, realized_pl=realized_pnl)
                calculate_and_record_performance(team_id)
                st.success("Option trade executed.")
                st.experimental_rerun()

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
