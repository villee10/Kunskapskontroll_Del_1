import streamlit as st
import requests
from google import genai
from google.genai import types
from datetime import datetime
import pytz

# --- 1. CONFIG & API ---
st.set_page_config(page_title="Big-Willieys Sport App", page_icon="🎾", layout="wide")

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
ODDS_API_KEY = st.secrets["ODDS_API_KEY"]
client = genai.Client(api_key=GEMINI_API_KEY)

# --- 2. DATASTRUKTUR ---
SPORTS_MAP = {
    "Fotboll": {
        "Allsvenskan 🇸🇪": "soccer_sweden_allsvenskan",
        "Superettan 🇸🇪": "soccer_sweden_superettan",
        "Premier League 🏴󠁧󠁢󠁥󠁮󠁧󠁿": "soccer_epl",
        "Champions League 🇪🇺": "soccer_uefa_champs_league",
        "Ligue 1 (PSG) 🇫🇷": "soccer_france_ligue1",
        "La Liga 🇪🇸": "soccer_spain_la_liga",
        "Serie A 🇮🇹": "soccer_italy_serie_a",
        "Bundesliga 🇩🇪": "soccer_germany_bundesliga"
    },
    "Tennis": {
        "WTA Charleston 🇺🇸": "tennis_wta_charleston_open",
        "ATP Houston 🇺🇸": "tennis_atp_houston",
        "ATP (Alla stora)": "tennis_atp_combined",
        "WTA (Alla stora)": "tennis_wta_combined",
        "ATP Challenger": "tennis_atp_challenger_combined",
        "ITF Herrar": "tennis_itf_men",
        "ITF Damer": "tennis_itf_women"
    },
    "Basket": {
        "NBA 🇺🇸": "basketball_nba",
        "Euroleague 🇪🇺": "basketball_euroleague",
        "NCAA 🇺🇸": "basketball_ncaa"
    },
    "Ishockey": {
        "NHL 🇺🇸": "icehockey_nhl",
        "SHL / Sv. Hockey": "icehockey_sweden_allsvenskan",
        "HockeyAllsvenskan": "icehockey_sweden_allsvenskan"
    },
    "MMA": {
        "UFC / MMA": "mma_mixed_martial_arts"
    }
}

# --- 3. STYLING ---
st.markdown("""
    <style>
    .odds-box { background-color: #1e1e1e; color: #ffdf1b; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; border: 1px solid #333; }
    .time-tag { color: #888; font-size: 0.85em; margin-bottom: -10px; font-weight: bold; }
    .league-nav { background-color: #007336; padding: 10px; border-radius: 10px; margin-bottom: 20px; color: white; text-align: center; font-weight: bold; }
    .stButton>button { width: 100%; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. FUNKTIONER ---
def fetch_odds(sport_key):
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?apiKey={ODDS_API_KEY}&regions=eu,us&markets=h2h&oddsFormat=decimal"
    try:
        res = requests.get(url)
        return res.json() if res.status_code == 200 else []
    except: return []

def format_time(iso_time):
    utc_dt = datetime.fromisoformat(iso_time.replace('Z', '+00:00'))
    sweden_tz = pytz.timezone('Europe/Stockholm')
    local_dt = utc_dt.astimezone(sweden_tz)
    if local_dt.date() == datetime.now().date():
        return f"Idag kl {local_dt.strftime('%H:%M')}"
    else:
        return local_dt.strftime('%d %b kl %H:%M')

def scan_active_leagues():
    url = f"https://api.the-odds-api.com/v4/sports/?apiKey={ODDS_API_KEY}"
    res = requests.get(url)
    return res.json() if res.status_code == 200 else []

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("Big-Willieys")
    valda_sporten = st.radio("VÄLJ SPORT", list(SPORTS_MAP.keys()))
    
    st.divider()
    with st.expander("🔍 API Scanner (Om matcher saknas)"):
        if st.button("Kör Live Scan"):
            active = scan_active_leagues()
            f_word = "tennis" if "Tennis" in valda_sporten else "soccer" if "Fotboll" in valda_sporten else "basketball" if "Basket" in valda_sporten else "mma" if "MMA" in valda_sporten else "icehockey"
            for s in active:
                if f_word in s['key'].lower():
                    st.code(f"{s['title']}: {s['key']}")

# --- 6. HUVUDYTA ---
st.title(f"{valda_sporten}")
ligor = SPORTS_MAP[valda_sporten]
valda_ligan_namn = st.selectbox("VÄLJ LIGA", list(ligor.keys()))
valda_ligan_nyckel = ligor[valda_ligan_namn]

if st.button("HÄMTA AKTUELLA ODDS"):
    with st.spinner("Hämtar senaste odds..."):
        st.session_state.matches = fetch_odds(valda_ligan_nyckel)

# --- 7. MATCHLISTA ---
if "matches" in st.session_state and st.session_state.matches:
    current_title = ""
    sorted_matches = sorted(st.session_state.matches, key=lambda x: x['commence_time'])
    
    for m in sorted_matches:
        if m['sport_title'] != current_title:
            current_title = m['sport_title']
            st.markdown(f"<div class='league-nav'>📍 {current_title}</div>", unsafe_allow_html=True)

        with st.container():
            st.markdown(f"<div class='time-tag'>{format_time(m['commence_time'])}</div>", unsafe_allow_html=True)
            c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 2])
            
            o1, ox, o2 = "-", "-", "-"
            try:
                if m['bookmakers']:
                    outcomes = m['bookmakers'][0]['markets'][0]['outcomes']
                    for out in outcomes:
                        if out['name'] == m['home_team']: o1 = out['price']
                        elif out['name'] == m['away_team']: o2 = out['price']
                        elif out['name'].lower() == 'draw': ox = out['price']
            except: pass

            c1.write(f"**{m['home_team']} - {m['away_team']}**")
            c2.markdown(f"<div class='odds-box'>{o1}</div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='odds-box'>{ox}</div>", unsafe_allow_html=True)
            c4.markdown(f"<div class='odds-box'>{o2}</div>", unsafe_allow_html=True)
            
            if c5.button("AI-ANALYS", key=f"btn_{m['id']}"):
                st.session_state.active_match = {"name": f"{m['home_team']} vs {m['away_team']}", "odds": f"1:{o1}, X:{ox}, 2:{o2}"}

# --- 8. AI ANALYS ---
if "active_match" in st.session_state:
    st.divider()
    am = st.session_state.active_match
    st.subheader(f" Expertanalys: {am['name']}")
    if prompt := st.chat_input("Fråga om skador, form eller vinstchans..."):
        with st.chat_message("assistant"):
            sys_instr = f"Du är expert-analytiker på Big-Willieys. Match: {am['name']}, Odds: {am['odds']}. Sök senaste info på nätet och ge en rekommendation på svenska."
            res = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=types.GenerateContentConfig(system_instruction=sys_instr, tools=[types.Tool(google_search={})])
            )
            st.markdown(res.text)