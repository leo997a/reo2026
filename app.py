import streamlit as st
import pandas as pd
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba, LinearSegmentedColormap
import seaborn as sns
import matplotlib.patches as patches
from mplsoccer import Pitch, VerticalPitch
from highlight_text import fig_text, ax_text
from PIL import Image
from urllib.request import urlopen
from unidecode import unidecode
from scipy.spatial import ConvexHull
import arabic_reshaper
from bidi.algorithm import get_display
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import warnings
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)

# تهيئة matplotlib لدعم العربية
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Amiri', 'Noto Sans Arabic', 'Arial', 'Tahoma']
plt.rcParams['axes.unicode_minus'] = False

# دالة لتحويل النص العربي
def reshape_arabic_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

# إضافة CSS لدعم RTL
st.markdown("""
    <style>
    body {
        direction: rtl;
        text-align: right;
    }
    .stSelectbox > div > div > div {
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)

# تعريف الألوان الافتراضية
default_hcol = '#d00000'
default_acol = '#003087'
default_bg_color = '#1e1e2f'
default_gradient_colors = ['#003087', '#d00000']
violet = '#800080'

# إعداد Session State
if 'json_data' not in st.session_state:
    st.session_state.json_data = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'teams_dict' not in st.session_state:
    st.session_state.teams_dict = None
if 'players_df' not in st.session_state:
    st.session_state.players_df = None
if 'analysis_triggered' not in st.session_state:
    st.session_state.analysis_triggered = False

# إضافة أدوات اختيار الألوان في الشريط الجانبي
st.sidebar.title('اختيار الألوان')
hcol = st.sidebar.color_picker('لون الفريق المضيف', default_hcol, key='hcol_picker')
acol = st.sidebar.color_picker('لون الفريق الضيف', default_acol, key='acol_picker')
bg_color = st.sidebar.color_picker('لون الخلفية', default_bg_color, key='bg_color_picker')
gradient_start = st.sidebar.color_picker('بداية التدرج', default_gradient_colors[0], key='gradient_start_picker')
gradient_end = st.sidebar.color_picker('نهاية التدرج', default_gradient_colors[1], key='gradient_end_picker')
gradient_colors = [gradient_start, gradient_end]
line_color = st.sidebar.color_picker('لون الخطوط', '#ffffff', key='line_color_picker')

# دالة استخراج البيانات من WhoScored
@st.cache_data
def extract_match_dict(match_url):
    driver = None
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")  # استخدام الوضع بدون واجهة
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument(
            "user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        )
        # إزالة معلمة version واستخدام أحدث إصدار
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        st.write("جارٍ تحميل الصفحة...")
        driver.get(match_url)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, 'script'))
        )
        time.sleep(5)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        element = soup.find(lambda tag: tag.name == 'script' and 'matchCentreData' in tag.text)
        if not element:
            st.error("لم يتم العثور على matchCentreData في الصفحة")
            return None
        matchdict = json.loads(element.text.split("matchCentreData: ")[1].split(',\n')[0])
        return matchdict
    except Exception as e:
        st.error(f"خطأ أثناء استخراج البيانات: {str(e)}")
        return None
    finally:
        if driver is not None:
            try:
                driver.quit()
            except:
                pass

# دالة معالجة البيانات
@st.cache_data
def extract_data_from_dict(data):
    try:
        events_dict = data["events"]
        teams_dict = {
            data['home']['teamId']: data['home']['name'],
            data['away']['teamId']: data['away']['name']
        }
        players_home_df = pd.DataFrame(data['home']['players'])
        players_home_df["teamId"] = data['home']['teamId']
        players_away_df = pd.DataFrame(data['away']['players'])
        players_away_df["teamId"] = data['away']['teamId']
        players_df = pd.concat([players_home_df, players_away_df])
        players_df['name'] = players_df['name'].astype(str)
        players_df['name'] = players_df['name'].apply(unidecode)
        return events_dict, players_df, teams_dict
    except Exception as e:
        st.error(f"خطأ أثناء معالجة البيانات: {str(e)}")
        return None, None, None

# دالة معالجة البيانات الأساسية
@st.cache_data
def get_event_data(json_data):
    events_dict, players_df, teams_dict = extract_data_from_dict(json_data)
    df = pd.DataFrame(events_dict)
    dfp = pd.DataFrame(players_df)
    
    df['type'] = df['type'].apply(lambda x: x.get('displayName') if isinstance(x, dict) else str(x))
    df['outcomeType'] = df['outcomeType'].apply(lambda x: x.get('displayName') if isinstance(x, dict) else str(x))
    df['period'] = df['period'].apply(lambda x: x.get('displayName') if isinstance(x, dict) else str(x))
    df['period'] = df['period'].replace({
        'FirstHalf': 1, 'SecondHalf': 2, 'FirstPeriodOfExtraTime': 3,
        'SecondPeriodOfExtraTime': 4, 'PenaltyShootout': 5, 'PostGame': 14, 'PreMatch': 16
    })
    
    def cumulative_match_mins(events_df):
        events_out = pd.DataFrame()
        match_events = events_df.copy()
        match_events['cumulative_mins'] = match_events['minute'] + (1/60) * match_events['second']
        for period in np.arange(1, match_events['period'].max() + 1, 1):
            if period > 1:
                t_delta = match_events[match_events['period'] == period - 1]['cumulative_mins'].max() - \
                          match_events[match_events['period'] == period]['cumulative_mins'].min()
            else:
                t_delta = 0
            match_events.loc[match_events['period'] == period, 'cumulative_mins'] += t_delta
        events_out = pd.concat([events_out, match_events])
        return events_out
    
    df = cumulative_match_mins(df)
    
    def insert_ball_carries(events_df, min_carry_length=3, max_carry_length=100, min_carry_duration=1, max_carry_duration=50):
        events_out = pd.DataFrame()
        min_carry_length = 3.0
        max_carry_length = 100.0
        min_carry_duration = 1.0
        max_carry_duration = 50.0
        match_events = events_df.reset_index()
        match_events.loc[match_events['type'] == 'BallRecovery', 'endX'] = match_events.loc[match_events['type'] == 'BallRecovery', 'endX'].fillna(match_events['x'])
        match_events.loc[match_events['type'] == 'BallRecovery', 'endY'] = match_events.loc[match_events['type'] == 'BallRecovery', 'endY'].fillna(match_events['y'])
        match_carries = pd.DataFrame()
        
        for idx, match_event in match_events.iterrows():
            if idx < len(match_events) - 1:
                prev_evt_team = match_event['teamId']
                next_evt_idx = idx + 1
                init_next_evt = match_events.loc[next_evt_idx]
                take_ons = 0
                incorrect_next_evt = True
                while incorrect_next_evt:
                    next_evt = match_events.loc[next_evt_idx]
                    if next_evt['type'] == 'TakeOn' and next_evt['outcomeType'] == 'Successful':
                        take_ons += 1
                        incorrect_next_evt = True
                    elif ((next_evt['type'] == 'TakeOn' and next_evt['outcomeType'] == 'Unsuccessful')
                          or (next_evt['teamId'] != prev_evt_team and next_evt['type'] == 'Challenge' and next_evt['outcomeType'] == 'Unsuccessful')
                          or (next_evt['type'] == 'Foul')
                          or (next_evt['type'] == 'Card')):
                        incorrect_next_evt = True
                    else:
                        incorrect_next_evt = False
                    next_evt_idx += 1
                same_team = prev_evt_team == next_evt['teamId']
                not_ball_touch = match_event['type'] != 'BallTouch'
                dx = 105*(match_event['endX'] - next_evt['x'])/100
                dy = 68*(match_event['endY'] - next_evt['y'])/100
                far_enough = dx ** 2 + dy ** 2 >= min_carry_length ** 2
                not_too_far = dx ** 2 + dy ** 2 <= max_carry_length ** 2
                dt = 60 * (next_evt['cumulative_mins'] - match_event['cumulative_mins'])
                min_time = dt >= min_carry_duration
                same_phase = dt < max_carry_duration
                same_period = match_event['period'] == next_evt['period']
                valid_carry = same_team & not_ball_touch & far_enough & not_too_far & min_time & same_phase & same_period
                if valid_carry:
                    carry = pd.DataFrame()
                    prev = match_event
                    nex = next_evt
                    carry.loc[0, 'eventId'] = prev['eventId'] + 0.5
                    carry['minute'] = np.floor(((init_next_evt['minute'] * 60 + init_next_evt['second']) + (
                        prev['minute'] * 60 + prev['second'])) / (2 * 60))
                    carry['second'] = (((init_next_evt['minute'] * 60 + init_next_evt['second']) +
                                        (prev['minute'] * 60 + prev['second'])) / 2) - (carry['minute'] * 60)
                    carry['teamId'] = nex['teamId']
                    carry['x'] = prev['endX']
                    carry['y'] = prev['endY']
                    carry['expandedMinute'] = np.floor(((init_next_evt['expandedMinute'] * 60 + init_next_evt['second']) +
                                                        (prev['expandedMinute'] * 60 + prev['second'])) / (2 * 60))
                    carry['period'] = nex['period']
                    carry['type'] = 'Carry'
                    carry['outcomeType'] = 'Successful'
                    carry['qualifiers'] = carry.apply(lambda x: {'type': {'value': 999, 'displayName': 'takeOns'}, 'value': str(take_ons)}, axis=1)
                    carry['satisfiedEventsTypes'] = carry.apply(lambda x: [], axis=1)
                    carry['isTouch'] = True
                    carry['playerId'] = nex['playerId']
                    carry['endX'] = nex['x']
                    carry['endY'] = nex['y']
                    carry['blockedX'] = np.nan
                    carry['blockedY'] = np.nan
                    carry['goalMouthZ'] = np.nan
                    carry['goalMouthY'] = np.nan
                    carry['isShot'] = np.nan
                    carry['relatedEventId'] = nex['eventId']
                    carry['relatedPlayerId'] = np.nan
                    carry['isGoal'] = np.nan
                    carry['cardType'] = np.nan
                    carry['isOwnGoal'] = np.nan
                    carry['cumulative_mins'] = (prev['cumulative_mins'] + init_next_evt['cumulative_mins']) / 2
                    match_carries = pd.concat([match_carries, carry], ignore_index=True, sort=False)
        match_events_and_carries = pd.concat([match_carries, match_events], ignore_index=True, sort=False)
        match_events_and_carries = match_events_and_carries.sort_values(['period', 'cumulative_mins']).reset_index(drop=True)
        events_out = pd.concat([events_out, match_events_and_carries])
        return events_out
    
    df = insert_ball_carries(df, min_carry_length=3, max_carry_length=100, min_carry_duration=1, max_carry_duration=50)
    df = df.reset_index(drop=True)
    df['index'] = range(1, len(df) + 1)
    df = df[['index'] + [col for col in df.columns if col != 'index']]
    
    df_base = df
    dfxT = df_base.copy()
    dfxT['qualifiers'] = dfxT['qualifiers'].astype(str)
    dfxT = dfxT[(~dfxT['qualifiers'].str.contains('Corner'))]
    dfxT = dfxT[(dfxT['type'].isin(['Pass', 'Carry'])) & (dfxT['outcomeType'] == 'Successful')]
    
    xT = pd.read_csv("xT_Grid.csv", header=None)
    xT = np.array(xT)
    xT_rows, xT_cols = xT.shape
    
    dfxT['x1_bin_xT'] = pd.cut(dfxT['x'], bins=xT_cols, labels=False)
    dfxT['y1_bin_xT'] = pd.cut(dfxT['y'], bins=xT_rows, labels=False)
    dfxT['x2_bin_xT'] = pd.cut(dfxT['endX'], bins=xT_cols, labels=False)
    dfxT['y2_bin_xT'] = pd.cut(dfxT['endY'], bins=xT_rows, labels=False)
    
    dfxT['start_zone_value_xT'] = dfxT[['x1_bin_xT', 'y1_bin_xT']].apply(lambda x: xT[x[1]][x[0]], axis=1)
    dfxT['end_zone_value_xT'] = dfxT[['x2_bin_xT', 'y2_bin_xT']].apply(lambda x: xT[x[1]][x[0]], axis=1)
    
    dfxT['xT'] = dfxT['end_zone_value_xT'] - dfxT['start_zone_value_xT']
    columns_to_drop = ['eventId', 'minute', 'second', 'teamId', 'x', 'y', 'expandedMinute', 'period', 'outcomeType', 'qualifiers', 'type', 'satisfiedEventsTypes', 'isTouch', 'playerId', 'endX', 'endY', 
                       'relatedEventId', 'relatedPlayerId', 'blockedX', 'blockedY', 'goalMouthZ', 'goalMouthY', 'isShot', 'cumulative_mins']
    dfxT.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
    df = df.merge(dfxT, on='index', how='left')
    df['teamName'] = df['teamId'].map(teams_dict)
    team_names = list(teams_dict.values())
    opposition_dict = {team_names[i]: team_names[1-i] for i in range(len(team_names))}
    df['oppositionTeamName'] = df['teamName'].map(opposition_dict)
    
    df['x'] = df['x'] * 1.05
    df['y'] = df['y'] * 0.68
    df['endX'] = df['endX'] * 1.05
    df['endY'] = df['endY'] * 0.68
    df['goalMouthY'] = df['goalMouthY'] * 0.68
    
    columns_to_drop = ['height', 'weight', 'age', 'isManOfTheMatch', 'field', 'stats', 'subbedInPlayerId', 'subbedOutPeriod', 'subbedOutExpandedMinute', 'subbedInPeriod', 'subbedInExpandedMinute', 'subbedOutPlayerId', 'teamId']
    dfp.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    df = df.merge(dfp, on='playerId', how='left')
    
    df['qualifiers'] = df['qualifiers'].astype(str)
    df['prog_pass'] = np.where((df['type'] == 'Pass'), 
                               np.sqrt((105 - df['x'])**2 + (34 - df['y'])**2) - np.sqrt((105 - df['endX'])**2 + (34 - df['endY'])**2), 0)
    df['prog_carry'] = np.where((df['type'] == 'Carry'), 
                                np.sqrt((105 - df['x'])**2 + (34 - df['y'])**2) - np.sqrt((105 - df['endX'])**2 + (34 - df['endY'])**2), 0)
    df['pass_or_carry_angle'] = np.degrees(np.arctan2(df['endY'] - df['y'], df['endX'] - df['x']))
    
    df['name'] = df['name'].astype(str)
    df['name'] = df['name'].apply(unidecode)
    
    def get_short_name(full_name):
        if pd.isna(full_name):
            return full_name
        parts = full_name.split()
        if len(parts) == 1:
            return full_name
        elif len(parts) == 2:
            return parts[0][0] + ". " + parts[1]
        else:
            return parts[0][0] + ". " + parts[1][0] + ". " + " ".join(parts[2:])
    
    df['shortName'] = df['name'].apply(get_short_name)
    columns_to_drop2 = ['id']
    df.drop(columns=columns_to_drop2, inplace=True, errors='ignore')
    
    def get_possession_chains(events_df, chain_check, suc_evts_in_chain):
        events_out = pd.DataFrame()
        match_events_df = events_df.reset_index()
        match_pos_events_df = match_events_df[~match_events_df['type'].isin(['OffsideGiven', 'CornerAwarded','Start', 'Card', 'SubstitutionOff',
                                                                              'SubstitutionOn', 'FormationChange','FormationSet', 'End'])].copy()
        match_pos_events_df['outcomeBinary'] = (match_pos_events_df['outcomeType']
                                                    .apply(lambda x: 1 if x == 'Successful' else 0))
        match_pos_events_df['teamBinary'] = (match_pos_events_df['teamName']
                             .apply(lambda x: 1 if x == min(match_pos_events_df['teamName']) else 0))
        match_pos_events_df['goalBinary'] = ((match_pos_events_df['type'] == 'Goal')
                             .astype(int).diff(periods=1).apply(lambda x: 1 if x < 0 else 0))
        pos_chain_df = pd.DataFrame()
        for n in np.arange(1, chain_check):
            pos_chain_df[f'evt_{n}_same_team'] = abs(match_pos_events_df['teamBinary'].diff(periods=-n))
            pos_chain_df[f'evt_{n}_same_team'] = pos_chain_df[f'evt_{n}_same_team'].apply(lambda x: 1 if x > 1 else x)
        pos_chain_df['enough_evt_same_team'] = pos_chain_df.sum(axis=1).apply(lambda x: 1 if x < chain_check - suc_evts_in_chain else 0)
        pos_chain_df['enough_evt_same_team'] = pos_chain_df['enough_evt_same_team'].diff(periods=1)
        pos_chain_df[pos_chain_df['enough_evt_same_team'] < 0] = 0
        match_pos_events_df['period'] = pd.to_numeric(match_pos_events_df['period'], errors='coerce')
        pos_chain_df['upcoming_ko'] = 0
        for ko in match_pos_events_df[(match_pos_events_df['goalBinary'] == 1) | (match_pos_events_df['period'].diff(periods=1))].index.values:
            ko_pos = match_pos_events_df.index.to_list().index(ko)
            pos_chain_df.iloc[ko_pos - suc_evts_in_chain:ko_pos, pos_chain_df.columns.get_loc('upcoming_ko')] = 1
        pos_chain_df['valid_pos_start'] = (pos_chain_df.fillna(0)['enough_evt_same_team'] - pos_chain_df.fillna(0)['upcoming_ko'])
        pos_chain_df['kick_off_period_change'] = match_pos_events_df['period'].diff(periods=1)
        pos_chain_df['kick_off_goal'] = ((match_pos_events_df['type'] == 'Goal')
                         .astype(int).diff(periods=1).apply(lambda x: 1 if x < 0 else 0))
        pos_chain_df.loc[pos_chain_df['kick_off_period_change'] == 1, 'valid_pos_start'] = 1
        pos_chain_df.loc[pos_chain_df['kick_off_goal'] == 1, 'valid_pos_start'] = 1
        pos_chain_df['teamName'] = match_pos_events_df['teamName']
        pos_chain_df.loc[pos_chain_df.head(1).index, 'valid_pos_start'] = 1
        pos_chain_df.loc[pos_chain_df.head(1).index, 'possession_id'] = 1
        pos_chain_df.loc[pos_chain_df.head(1).index, 'possession_team'] = pos_chain_df.loc[pos_chain_df.head(1).index, 'teamName']
        valid_pos_start_id = pos_chain_df[pos_chain_df['valid_pos_start'] > 0].index
        possession_id = 2
        for idx in np.arange(1, len(valid_pos_start_id)):
            current_team = pos_chain_df.loc[valid_pos_start_id[idx], 'teamName']
            previous_team = pos_chain_df.loc[valid_pos_start_id[idx - 1], 'teamName']
            if ((previous_team == current_team) & (pos_chain_df.loc[valid_pos_start_id[idx], 'kick_off_goal'] != 1) & 
                (pos_chain_df.loc[valid_pos_start_id[idx], 'kick_off_period_change'] != 1)):
                pos_chain_df.loc[valid_pos_start_id[idx], 'possession_id'] = np.nan
            else:
                pos_chain_df.loc[valid_pos_start_id[idx], 'possession_id'] = possession_id
                pos_chain_df.loc[valid_pos_start_id[idx], 'possession_team'] = pos_chain_df.loc[valid_pos_start_id[idx], 'teamName']
                possession_id += 1
        match_events_df = pd.merge(match_events_df, pos_chain_df[['possession_id', 'possession_team']], how='left', left_index=True, right_index=True)
        match_events_df[['possession_id', 'possession_team']] = (match_events_df[['possession_id', 'possession_team']].fillna(method='ffill'))
        match_events_df[['possession_id', 'possession_team']] = (match_events_df[['possession_id', 'possession_team']].fillna(method='bfill'))
        events_out = pd.concat([events_out, match_events_df])
        return events_out
    
    df = get_possession_chains(df, 5, 3)
    df['period'] = df['period'].replace({1: 'FirstHalf', 2: 'SecondHalf', 3: 'FirstPeriodOfExtraTime', 4: 'SecondPeriodOfExtraTime', 5: 'PenaltyShootout', 14: 'PostGame', 16: 'PreMatch'})
    df = df[df['period'] != 'PenaltyShootout']
    df = df.reset_index(drop=True)
    return df, teams_dict, players_df

# واجهة Streamlit
st.title("تحليل مباراة كرة القدم")
match_url = st.text_input(
    "أدخل رابط المباراة من WhoScored:",
    value="https://1xbet.whoscored.com/Matches/1809770/Live/Europe-Europa-League-2023-2024-West-Ham-Bayer-Leverkusen"
)
uploaded_file = st.file_uploader("أو قم بتحميل ملف JSON (اختياري):", type="json")

if st.button("تحليل المباراة"):
    with st.spinner("جارٍ استخراج بيانات المباراة..."):
        st.session_state.json_data = None
        st.session_state.df = None
        st.session_state.teams_dict = None
        st.session_state.players_df = None
        if uploaded_file:
            try:
                st.session_state.json_data = json.load(uploaded_file)
            except Exception as e:
                st.error(f"خطأ في تحميل ملف JSON: {str(e)}")
        else:
            st.session_state.json_data = extract_match_dict(match_url)
        
        if st.session_state.json_data:
            st.session_state.df, st.session_state.teams_dict, st.session_state.players_df = get_event_data(st.session_state.json_data)
            st.session_state.analysis_triggered = True
            if st.session_state.df is not None and st.session_state.teams_dict and st.session_state.players_df is not None:
                st.success("تم استخراج البيانات بنجاح!")
            else:
                st.error("فشل في معالجة البيانات.")
        else:
            st.error("فشل في جلب بيانات المباراة.")

# عرض التحليل فقط إذا تم استخراج البيانات
if st.session_state.analysis_triggered and st.session_state.df is not None and st.session_state.teams_dict and st.session_state.players_df is not None:
    hteamID = list(st.session_state.teams_dict.keys())[0]
    ateamID = list(st.session_state.teams_dict.keys())[1]
    hteamName = st.session_state.teams_dict[hteamID]
    ateamName = st.session_state.teams_dict[ateamID]
    
    homedf = st.session_state.df[(st.session_state.df['teamName'] == hteamName)]
    awaydf = st.session_state.df[(st.session_state.df['teamName'] == ateamName)]
    hxT = homedf['xT'].sum().round(2)
    axT = awaydf['xT'].sum().round(2)
    
    hgoal_count = len(homedf[(homedf['teamName'] == hteamName) & (homedf['type'] == 'Goal') & (~homedf['qualifiers'].str.contains('OwnGoal'))])
    agoal_count = len(awaydf[(awaydf['teamName'] == ateamName) & (awaydf['type'] == 'Goal') & (~awaydf['qualifiers'].str.contains('OwnGoal'))])
    hgoal_count += len(awaydf[(awaydf['teamName'] == ateamName) & (awaydf['type'] == 'Goal') & (awaydf['qualifiers'].str.contains('OwnGoal'))])
    agoal_count += len(homedf[(homedf['teamName'] == hteamName) & (homedf['type'] == 'Goal') & (homedf['qualifiers'].str.contains('OwnGoal'))])
    
    df_teamNameId = pd.read_csv("teams_name_and_id.csv")
    hftmb_tid = df_teamNameId[df_teamNameId['teamName'] == hteamName]['teamId'].iloc[0] if not df_teamNameId[df_teamNameId['teamName'] == hteamName].empty else 0
    aftmb_tid = df_teamNameId[df_teamNameId['teamName'] == ateamName]['teamId'].iloc[0] if not df_teamNameId[df_teamNameId['teamName'] == ateamName].empty else 0
    
    st.header(f'{hteamName} {hgoal_count} - {agoal_count} {ateamName}')
    
    # دالة شبكة التمريرات
    def pass_network(ax, team_name, col, phase_tag):
        if phase_tag == 'Full Time':
            df_pass = st.session_state.df.copy()
            df_pass = df_pass.reset_index(drop=True)
        elif phase_tag == 'First Half':
            df_pass = st.session_state.df[st.session_state.df['period'] == 'FirstHalf']
            df_pass = df_pass.reset_index(drop=True)
        elif phase_tag == 'Second Half':
            df_pass = st.session_state.df[st.session_state.df['period'] == 'SecondHalf']
            df_pass = df_pass.reset_index(drop=True)
        total_pass = df_pass[(df_pass['teamName'] == team_name) & (df_pass['type'] == 'Pass')]
        accrt_pass = df_pass[(df_pass['teamName'] == team_name) & (df_pass['type'] == 'Pass') & (df_pass['outcomeType'] == 'Successful')]
        accuracy = round((len(accrt_pass) / len(total_pass)) * 100, 2) if len(total_pass) != 0 else 0
        df_pass['pass_receiver'] = df_pass.loc[(df_pass['type'] == 'Pass') & (df_pass['outcomeType'] == 'Successful') & (df_pass['teamName'].shift(-1) == team_name), 'name'].shift(-1)
        df_pass['pass_receiver'] = df_pass['pass_receiver'].fillna('No')
        off_acts_df = df_pass[(df_pass['teamName'] == team_name) & (df_pass['type'].isin(['Pass', 'Goal', 'MissedShots', 'SavedShot', 'ShotOnPost', 'TakeOn', 'BallTouch', 'KeeperPickup']))]
        off_acts_df = off_acts_df[['name', 'x', 'y']].reset_index(drop=True)
        avg_locs_df = off_acts_df.groupby('name').agg(avg_x=('x', 'median'), avg_y=('y', 'median')).reset_index()
        team_pdf = st.session_state.players_df[['name', 'shirtNo', 'position', 'isFirstEleven']]
        avg_locs_df = avg_locs_df.merge(team_pdf, on='name', how='left')
        df_pass = df_pass[(df_pass['type'] == 'Pass') & (df_pass['outcomeType'] == 'Successful') & (df_pass['teamName'] == team_name) & (~df_pass['qualifiers'].str.contains('Corner|Freekick'))]
        df_pass = df_pass[['type', 'name', 'pass_receiver']].reset_index(drop=True)
        pass_count_df = df_pass.groupby(['name', 'pass_receiver']).size().reset_index(name='pass_count').sort_values(by='pass_count', ascending=False)
        pass_count_df = pass_count_df.reset_index(drop=True)
        pass_counts_df = pd.merge(pass_count_df, avg_locs_df, on='name', how='left')
        pass_counts_df.rename(columns={'avg_x': 'pass_avg_x', 'avg_y': 'pass_avg_y'}, inplace=True)
        pass_counts_df = pd.merge(pass_counts_df, avg_locs_df, left_on='pass_receiver', right_on='name', how='left', suffixes=('', '_receiver'))
        pass_counts_df.drop(columns=['name_receiver'], inplace=True)
        pass_counts_df.rename(columns={'avg_x': 'receiver_avg_x', 'avg_y': 'receiver_avg_y'}, inplace=True)
        pass_counts_df = pass_counts_df.sort_values(by='pass_count', ascending=False).reset_index(drop=True)
        pass_counts_df = pass_counts_df.dropna(subset=['shirtNo_receiver'])
        pass_btn = pass_counts_df[['name', 'shirtNo', 'pass_receiver', 'shirtNo_receiver', 'pass_count']]
        pass_btn['shirtNo_receiver'] = pass_btn['shirtNo_receiver'].astype(float).astype(int)
        MAX_LINE_WIDTH = 8
        MIN_LINE_WIDTH = 0.5
        MIN_TRANSPARENCY = 0.2
        MAX_TRANSPARENCY = 0.9
        pass_counts_df['line_width'] = (pass_counts_df['pass_count'] / pass_counts_df['pass_count'].max()) * (MAX_LINE_WIDTH - MIN_LINE_WIDTH) + MIN_LINE_WIDTH
        c_transparency = pass_counts_df['pass_count'] / pass_counts_df['pass_count'].max()
        c_transparency = (c_transparency * (MAX_TRANSPARENCY - MIN_TRANSPARENCY)) + MIN_TRANSPARENCY
        color = np.array(to_rgba(col))
        color = np.tile(color, (len(pass_counts_df), 1))
        color[:, 3] = c_transparency
        pitch = VerticalPitch(pitch_type='uefa', corner_arcs=True, linewidth=1.5, line_color=line_color)
        pitch.draw(ax=ax)
        gradient = LinearSegmentedColormap.from_list("pitch_gradient", gradient_colors, N=100)
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(x, y)
        Z = Y
        ax.imshow(Z, extent=[0, 68, 0, 105], cmap=gradient, alpha=0.8, aspect='auto', zorder=0)
        pitch.draw(ax=ax)
        for idx in range(len(pass_counts_df)):
            pitch.lines(
                pass_counts_df['pass_avg_x'].iloc[idx],
                pass_counts_df['pass_avg_y'].iloc[idx],
                pass_counts_df['receiver_avg_x'].iloc[idx],
                pass_counts_df['receiver_avg_y'].iloc[idx],
                lw=pass_counts_df['line_width'].iloc[idx],
                color=color[idx],
                zorder=1,
                ax=ax
            )
        for index, row in avg_locs_df.iterrows():
            if row['isFirstEleven'] == True:
                pitch.scatter(row['avg_x'], row['avg_y'], s=800, marker='o', color=col, edgecolor=line_color, linewidth=1.5, alpha=0.9, ax=ax)
            else:
                pitch.scatter(row['avg_x'], row['avg_y'], s=800, marker='s', color=col, edgecolor=line_color, linewidth=1.5, alpha=0.7, ax=ax)
        for index, row in avg_locs_df.iterrows():
            player_initials = row["shirtNo"]
            pitch.annotate(player_initials, xy=(row.avg_x, row.avg_y), c='white', ha='center', va='center', size=14, weight='bold', ax=ax)
        avgph = round(avg_locs_df['avg_x'].median(), 2)
        ax.axhline(y=avgph, color='white', linestyle='--', alpha=0.5, linewidth=1.5)
        center_backs_height = avg_locs_df[avg_locs_df['position'] == 'DC']
        def_line_h = round(center_backs_height['avg_x'].median(), 2) if not center_backs_height.empty else avgph
        Forwards_height = avg_locs_df[avg_locs_df['isFirstEleven'] == True].sort_values(by='avg_x', ascending=False).head(2)
        fwd_line_h = round(Forwards_height['avg_x'].mean(), 2) if not Forwards_height.empty else avgph
        ymid = [0, 0, 68, 68]
        xmid = [def_line_h, fwd_line_h, fwd_line_h, def_line_h]
        ax.fill(ymid, xmid, col, alpha=0.2)
        v_comp = round((1 - ((fwd_line_h - def_line_h) / 105)) * 100, 2)
        if phase_tag == 'Full Time':
            ax.text(34, 115, reshape_arabic_text('الوقت بالكامل: 0-90 دقيقة'), color='white', fontsize=14, ha='center', va='center', weight='bold')
            ax.text(34, 112, reshape_arabic_text(f'إجمالي التمريرات: {len(total_pass)} | الناجحة: {len(accrt_pass)} | الدقة: {accuracy}%'), color='white', fontsize=12, ha='center', va='center')
        elif phase_tag == 'First Half':
            ax.text(34, 115, reshape_arabic_text('الشوط الأول: 0-45 دقيقة'), color='white', fontsize=14, ha='center', va='center', weight='bold')
            ax.text(34, 112, reshape_arArabic_text(f'إجمالي التمريرات: {len(total_pass)} | الناجحة: {len(accrt_pass)} | الدقة: {accuracy}%'), color='white', fontsize=12, ha='center', va='center')
        elif phase_tag == 'Second Half':
            ax.text(34, 115, reshape_arabic_text('الشوط الثاني: 45-90 دقيقة'), color='white', fontsize=14, ha='center', va='center', weight='bold')
            ax.text(34, 112, reshape_arabic_text(f'إجمالي التمريرات: {len(total_pass)} | الناجحة: {len(accrt_pass)} | الدقة: {accuracy}%'), color='white', fontsize=12, ha='center', va='center')
        ax.text(34, -6, reshape_arabic_text(f"على الكرة\nالتماسك العمودي (المنطقة المظللة): {v_comp}%"), color='white', fontsize=12, ha='center', va='center', weight='bold')
        return pass_btn
    
    # علامات التبويب
    tab1, tab2, tab3, tab4 = st.tabs(['تحليل الفريق', 'تحليل اللاعبين', 'إحصائيات المباراة', 'أفضل اللاعبين'])
    
    with tab1:
        an_tp = st.selectbox('نوع التحليل:', [
            'شبكة التمريرات',
            'Defensive Actions Heatmap',
            'Progressive Passes',
            'Progressive Carries',
            'Shotmap',
            'إحصائيات الحراس',
            'Match Momentum',
            reshape_arabic_text('Zone14 & Half-Space Passes'),
            reshape_arabic_text('Final Third Entries'),
            reshape_arabic_text('Box Entries'),
            reshape_arabic_text('High-Turnovers'),
            reshape_arabic_text('Chances Creating Zones'),
            reshape_arabic_text('Crosses'),
            reshape_arabic_text('Team Domination Zones'),
            reshape_arabic_text('Pass Target Zones'),
            'Attacking Thirds'
        ], index=0, key='analysis_type')
        
        if an_tp == 'شبكة التمريرات':
            st.subheader('شبكة التمريرات')
            team_choice = st.selectbox('اختر الفريق:', [hteamName, ateamName], key='team_choice')
            phase_tag = st.selectbox('اختر الفترة:', ['Full Time', 'First Half', 'Second Half'], key='phase_tag')
            fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)
            col = hcol if team_choice == hteamName else acol
            pass_btn = pass_network(ax, team_choice, col, phase_tag)
            st.pyplot(fig)
            st.dataframe(pass_btn, hide_index=True)
    
    with tab2:
        st.write("تحليل اللاعبين قيد التطوير...")
    
    with tab3:
        st.write("إحصائيات المباراة قيد التطوير...")
    
    with tab4:
        top_type = st.selectbox('اختر النوع:', [
            'Top Ball Progressors',
            'Top Shot Sequences Involvements',
            'Top Defensive Involvements',
            'Top Ball Recoverers',
            'Top Threat Creating Players'
        ], index=None, key='top_players_selection')
        if top_type:
            def top_dfs():
                unique_players = st.session_state.df['name'].unique()
                progressor_counts = {'name': unique_players, 'Progressive Passes': [], 'Progressive Carries': [], 'team names': []}
                for name in unique_players:
                    dfp = st.session_state.df[(st.session_state.df['name'] == name) & (st.session_state.df['outcomeType'] == 'Successful')]
                    progressor_counts['Progressive Passes'].append(len(dfp[(dfp['prog_pass'] >= 9.144) & (dfp['x'] >= 35) & (~dfp['qualifiers'].str.contains('CornerTaken|Freekick'))]))
                    progressor_counts['Progressive Carries'].append(len(dfp[(dfp['prog_carry'] >= 9.144) & (dfp['endX'] >= 35)]))
                    progressor_counts['team names'].append(dfp['teamName'].max())
                progressor_df = pd.DataFrame(progressor_counts)
                progressor_df['total'] = progressor_df['Progressive Passes'] + progressor_df['Progressive Carries']
                progressor_df = progressor_df.sort_values(by=['total', 'Progressive Passes'], ascending=[False, False])
                progressor_df = progressor_df.head(10)
                progressor_df['shortName'] = progressor_df['name'].apply(get_short_name)
                
                xT_counts = {'name': unique_players, 'xT from Pass': [], 'xT from Carry': [], 'team names': []}
                for name in unique_players:
                    dfp = st.session_state.df[(st.session_state.df['name'] == name) & (st.session_state.df['outcomeType'] == 'Successful') & (st.session_state.df['xT'] > 0)]
                    xT_counts['xT from Pass'].append((dfp[(dfp['type'] == 'Pass') & (~dfp['qualifiers'].str.contains('CornerTaken|Freekick|ThrowIn'))])['xT'].sum().round(2))
                    xT_counts['xT from Carry'].append((dfp[(dfp['type'] == 'Carry')])['xT'].sum().round(2))
                    xT_counts['team names'].append(dfp['teamName'].max())
                xT_df = pd.DataFrame(xT_counts)
                xT_df['total'] = xT_df['xT from Pass'] + xT_df['xT from Carry']
                xT_df = xT_df.sort_values(by=['total', 'xT from Pass'], ascending=[False, False])
                xT_df = xT_df.head(10)
                xT_df['shortName'] = xT_df['name'].apply(get_short_name)
                
                df_no_carry = st.session_state.df[~st.session_state.df['type'].str.contains('Carry|TakeOn|Challenge')].reset_index(drop=True)
                shot_seq_counts = {'name': unique_players, 'Shots': [], 'Shot Assists': [], 'Buildup to Shot': [], 'team names': []}
                for name in unique_players:
                    dfp = df_no_carry[df_no_carry['name'] == name]
                    shot_seq_counts['Shots'].append(len(dfp[dfp['type'].isin(['MissedShots', 'SavedShot', 'ShotOnPost', 'Goal'])]))
                    shot_seq_counts['Shot Assists'].append(len(dfp[(dfp['qualifiers'].str.contains('KeyPass'))]))
                    shot_seq_counts['Buildup to Shot'].append(len(df_no_carry[(df_no_carry['type'] == 'Pass') & 
                                                                             (df_no_carry['outcomeType'] == 'Successful') & 
                                                                             (df_no_carry['name'] == name) & 
                                                                             (df_no_carry['qualifiers'].shift(-1).str.contains('KeyPass'))]))
                    shot_seq_counts['team names'].append(dfp['teamName'].max())
                sh_sq_df = pd.DataFrame(shot_seq_counts)
                sh_sq_df['total'] = sh_sq_df['Shots'] + sh_sq_df['Shot Assists'] + sh_sq_df['Buildup to Shot']
                sh_sq_df = sh_sq_df.sort_values(by=['total', 'Shots', 'Shot Assists'], ascending=[False, False, False])
                sh_sq_df = sh_sq_df.head(10)
                sh_sq_df['shortName'] = sh_sq_df['name'].apply(get_short_name)
                
                defensive_actions_counts = {'name': unique_players, 'Tackles': [], 'Interceptions': [], 'Clearance': [], 'team names': []}
                for name in unique_players:
                    dfp = st.session_state.df[(st.session_state.df['name'] == name) & (st.session_state.df['outcomeType'] == 'Successful')]
                    defensive_actions_counts['Tackles'].append(len(dfp[dfp['type'] == 'Tackle']))
                    defensive_actions_counts['Interceptions'].append(len(dfp[dfp['type'] == 'Interception']))
                    defensive_actions_counts['Clearance'].append(len(dfp[dfp['type'] == 'Clearance']))
                    defensive_actions_counts['team names'].append(dfp['teamName'].max())
                defender_df = pd.DataFrame(defensive_actions_counts)
                defender_df['total'] = defender_df['Tackles'] + defender_df['Interceptions'] + defender_df['Clearance']
                defender_df = defender_df.sort_values(by=['total', 'Tackles', 'Interceptions'], ascending=[False, False, False])
                defender_df = defender_df.head(10)
                defender_df['shortName'] = defender_df['name'].apply(get_short_name)
                
                recovery_counts = {'name': unique_players, 'Ball Recoveries': [], 'team names': []}
                for name in unique_players:
                    dfp = st.session_state.df[(st.session_state.df['name'] == name) & (st.session_state.df['outcomeType'] == 'Successful')]
                    recovery_counts['Ball Recoveries'].append(len(dfp[dfp['type'] == 'BallRecovery']))
                    recovery_counts['team names'].append(dfp['teamName'].max())
                recovery_df = pd.DataFrame(recovery_counts)
                recovery_df['total'] = recovery_df['Ball Recoveries']
                recovery_df = recovery_df.sort_values(by=['total'], ascending=False)
                recovery_df = recovery_df.head(10)
                recovery_df['shortName'] = recovery_df['name'].apply(get_short_name)
                
                return progressor_df, xT_df, sh_sq_df, defender_df, recovery_df
            
            progressor_df, xT_df, sh_sq_df, defender_df, recovery_df = top_dfs()
            
            def passer_bar(ax):
                top10_progressors = progressor_df['shortName'][::-1].tolist()
                progressor_pp = progressor_df['Progressive Passes'][::-1].tolist()
                progressor_pc = progressor_df['Progressive Carries'][::-1].tolist()
                ax.barh(top10_progressors, progressor_pp, label='Prog. Pass', zorder=3, color=hcol, left=0)
                ax.barh(top10_progressors, progressor_pc, label='Prog. Carry', zorder=3, color=acol, left=progressor_pp)
                for i, player in enumerate(top10_progressors):
                    for j, count in enumerate([progressor_pp[i], progressor_pc[i]]):
                        if count > 0:
                            x_position = sum([progressor_pp[i], progressor_pc[i]][:j]) + count / 2
                            ax.text(x_position, i, str(count), ha='center', va='center', color=bg_color, fontsize=10, fontweight='bold')
                    ax.text(progressor_df['total'].iloc[i] + 0.25, 9-i, str(progressor_df['total'].iloc[i]), ha='left', va='center', color=line_color, fontsize=10, fontweight='bold')
                ax.set_facecolor(bg_color)
                ax.tick_params(axis='x', colors=line_color, labelsize=8)
                ax.tick_params(axis='y', colors=line_color, labelsize=10)
                ax.grid(True, zorder=1, ls='dotted', lw=1, color='gray')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_visible(True)
                ax.legend(fontsize=8, loc='lower right')
                return
            
            def sh_sq_bar(ax):
                top10 = sh_sq_df.head(10).iloc[::-1]
                ax.barh(top10['shortName'], top10['Shots'], zorder=3, height=0.75, label='Shot', color=hcol)
                ax.barh(top10['shortName'], top10['Shot Assists'], zorder=3, height=0.75, label='Shot Assist', color='green', left=top10['Shots'])
                ax.barh(top10['shortName'], top10['Buildup to Shot'], zorder=3, height=0.75, label='Buildup to Shot', color=acol, left=top10[['Shots', 'Shot Assists']].sum(axis=1))
                for i, player in enumerate(top10['shortName']):
                    for j, count in enumerate(top10[['Shots', 'Shot Assists', 'Buildup to Shot']].iloc[i]):
                        if count > 0:
                            x_position = sum(top10.iloc[i, 1:1+j]) + count / 2
                            ax.text(x_position, i, str(count), ha='center', va='center', color=bg_color, fontsize=10, fontweight='bold')
                    ax.text(top10['total'].iloc[i] + 0.25, i, str(top10['total'].iloc[i]), ha='left', va='center', color=line_color, fontsize=10, fontweight='bold')
                ax.set_facecolor(bg_color)
                ax.tick_params(axis='x', colors=line_color, labelsize=8)
                ax.tick_params(axis='y', colors=line_color, labelsize=10)
                ax.grid(True, zorder=1, ls='dotted', lw=1, color='gray')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_visible(True)
                ax.legend(fontsize=8, loc='lower right')
                return
            
            def top_defender(ax):
                top10 = defender_df.head(10).iloc[::-1]
                ax.barh(top10['shortName'], top10['Tackles'], zorder=3, height=0.75, label='Tackle', color=hcol)
                ax.barh(top10['shortName'], top10['Interceptions'], zorder=3, height=0.75, label='Interception', color='green', left=top10['Tackles'])
                ax.barh(top10['shortName'], top10['Clearance'], zorder=3, height=0.75, label='Clearance', color=acol, left=top10[['Tackles', 'Interceptions']].sum(axis=1))
                for i, player in enumerate(top10['shortName']):
                    for j, count in enumerate(top10[['Tackles', 'Interceptions', 'Clearance']].iloc[i]):
                        if count > 0:
                            x_position = sum(top10.iloc[i, 1:1+j]) + count / 2
                            ax.text(x_position, i, str(count), ha='center', va='center', color=bg_color, fontsize=10, fontweight='bold')
                    ax.text(top10['total'].iloc[i] + 0.25, i, str(top10['total'].iloc[i]), ha='left', va='center', color=line_color, fontsize=10, fontweight='bold')
                ax.set_facecolor(bg_color)
                ax.tick_params(axis='x', colors=line_color, labelsize=8)
                ax.tick_params(axis='y', colors=line_color, labelsize=10)
                ax.grid(True, zorder=1, ls='dotted', lw=1, color='gray')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_visible(True)
                ax.legend(fontsize=8, loc='lower right')
                return
            
            def xT_bar(ax):
                top10_progressors = xT_df['shortName'][::-1].tolist()
                progressor_pp = xT_df['xT from Pass'][::-1].tolist()
                progressor_pc = xT_df['xT from Carry'][::-1].tolist()
                total_rounded = xT_df['total'].round(2)[::-1].tolist()
                ax.barh(top10_progressors, progressor_pp, label='xT from Pass', zorder=3, color=hcol, left=0)
                ax.barh(top10_progressors, progressor_pc, label='xT from Carry', zorder=3, color=acol, left=progressor_pp)
                for i, player in enumerate(top10_progressors):
                    for j, count in enumerate([progressor_pp[i], progressor_pc[i]]):
                        if count > 0:
                            x_position = sum([progressor_pp[i], progressor_pc[i]][:j]) + count / 2
                            ax.text(x_position, i, f"{count:.2f}", ha='center', va='center', color=bg_color, fontsize=10, fontweight='bold')
                    ax.text(xT_df['total'].iloc[i] + 0.01, 9-i, f"{total_rounded[i]:.2f}", ha='left', va='center', color=line_color, fontsize=10, fontweight='bold')
                ax.set_facecolor(bg_color)
                ax.tick_params(axis='x', colors=line_color, labelsize=8)
                ax.tick_params(axis='y', colors=line_color, labelsize=10)
                ax.grid(True, zorder=1, ls='dotted', lw=1, color='gray')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_visible(True)
                ax.legend(fontsize=8, loc='lower right')
                return
            
            def ball_recoverers_bar(ax):
                top10 = recovery_df.head(10).iloc[::-1]
                ax.barh(top10['shortName'], top10['Ball Recoveries'], zorder=3, height=0.75, label='Ball Recoveries', color=hcol)
                for i, player in enumerate(top10['shortName']):
                    count = top10['Ball Recoveries'].iloc[i]
                    if count > 0:
                        ax.text(count / 2, i, str(count), ha='center', va='center', color=bg_color, fontsize=10, fontweight='bold')
                    ax.text(count + 0.25, i, str(count), ha='left', va='center', color=line_color, fontsize=10, fontweight='bold')
                ax.set_facecolor(bg_color)
                ax.tick_params(axis='x', colors=line_color, labelsize=8)
                ax.tick_params(axis='y', colors=line_color, labelsize=10)
                ax.grid(True, zorder=1, ls='dotted', lw=1, color='gray')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_visible(True)
                ax.legend(fontsize=8, loc='lower right')
                return
            
            if top_type == 'Top Ball Progressors':
                fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)
                passer_bar(ax)
                fig.text(0.5, 1.02, 'Top Ball Progressors', fontsize=12, fontweight='bold', ha='center', va='center')
                fig.text(0.5, 0.97, f'in the match {hteamName} {hgoal_count} - {agoal_count} {ateamName}', color='#1a1a1a', fontsize=10, ha='center', va='center')
                st.pyplot(fig)
            elif top_type == 'Top Shot Sequences Involvements':
                fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)
                sh_sq_bar(ax)
                fig.text(0.5, 1.02, 'Top Shot Sequence Involvements', fontsize=12, fontweight='bold', ha='center', va='center')
                fig.text(0.5, 0.97, f'in the match {hteamName} {hgoal_count} - {agoal_count} {ateamName}', color='#1a1a1a', fontsize=10, ha='center', va='center')
                st.pyplot(fig)
            elif top_type == 'Top Defensive Involvements':
                fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)
                top_defender(ax)
                fig.text(0.5, 1.02, 'Top Defensive Involvements', fontsize=12, fontweight='bold', ha='center', va='center')
                fig.text(0.5, 0.97, f'in the match {hteamName} {hgoal_count} - {agoal_count} {ateamName}', color='#1a1a1a', fontsize=10, ha='center', va='center')
                st.pyplot(fig)
            elif top_type == 'Top Threat Creating Players':
                fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)
                xT_bar(ax)
                fig.text(0.5, 1.02, 'Top Threat Creating Players', fontsize=12, fontweight='bold', ha='center', va='center')
                fig.text(0.5, 0.97, f'in the match {hteamName} {hgoal_count} - {agoal_count} {ateamName}', color='#1a1a1a', fontsize=10, ha='center', va='center')
                st.pyplot(fig)
            elif top_type == 'Top Ball Recoverers':
                fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)
                ball_recoverers_bar(ax)
                fig.text(0.5, 1.02, 'Top Ball Recoverers', fontsize=12, fontweight='bold', ha='center', va='center')
                fig.text(0.5, 0.97, f'in the match {hteamName} {hgoal_count} - {agoal_count} {ateamName}', color='#1a1a1a', fontsize=10, ha='center', va='center')
                st.pyplot(fig)
else:
    st.write("أدخل رابط المباراة أو قم بتحميل ملف JSON واضغط 'تحليل المباراة'.")
