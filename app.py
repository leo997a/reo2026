import streamlit as st
import pandas as pd
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba, LinearSegmentedColormap
import seaborn as sns
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
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
plt.rcParams['font.sans-serif'] = ['Amiri',
                                   'Noto Sans Arabic', 'Arial', 'Tahoma']
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
hcol = st.sidebar.color_picker(
    'لون الفريق المضيف',
    default_hcol,
    key='hcol_picker')
acol = st.sidebar.color_picker(
    'لون الفريق الضيف',
    default_acol,
    key='acol_picker')
bg_color = st.sidebar.color_picker(
    'لون الخلفية',
    default_bg_color,
    key='bg_color_picker')
gradient_start = st.sidebar.color_picker(
    'بداية التدرج',
    default_gradient_colors[0],
    key='gradient_start_picker')
gradient_end = st.sidebar.color_picker(
    'نهاية التدرج',
    default_gradient_colors[1],
    key='gradient_end_picker')
gradient_colors = [gradient_start, gradient_end]
line_color = st.sidebar.color_picker(
    'لون الخطوط', '#ffffff', key='line_color_picker')

# دالة استخراج البيانات من WhoScored


@st.cache_data
def extract_match_dict(match_url):
    driver = None
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument(
            "user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        )
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        st.write("جارٍ تحميل الصفحة...")
        driver.get(match_url)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, 'script'))
        )
        time.sleep(5)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        element = soup.find(lambda tag: tag.name ==
                            'script' and 'matchCentreData' in tag.text)
        if not element:
            st.error("لم يتم العثور على matchCentreData في الصفحة")
            return None
        matchdict = json.loads(element.text.split(
            "matchCentreData: ")[1].split(',\n')[0])
        return matchdict
    except Exception as e:
        st.error(f"خطأ أثناء استخراج البيانات: {str(e)}")
        return None
    finally:
        if driver is not None:
            try:
                driver.quit()
            except BaseException:
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

    df['type'] = df['type'].apply(
        lambda x: x.get('displayName') if isinstance(
            x, dict) else str(x))
    df['outcomeType'] = df['outcomeType'].apply(
        lambda x: x.get('displayName') if isinstance(
            x, dict) else str(x))
    df['period'] = df['period'].apply(
        lambda x: x.get('displayName') if isinstance(
            x, dict) else str(x))
    df['period'] = df['period'].replace({
        'FirstHalf': 1, 'SecondHalf': 2, 'FirstPeriodOfExtraTime': 3,
        'SecondPeriodOfExtraTime': 4, 'PenaltyShootout': 5, 'PostGame': 14, 'PreMatch': 16
    })

    def cumulative_match_mins(events_df):
        events_out = pd.DataFrame()
        match_events = events_df.copy()
        match_events['cumulative_mins'] = match_events['minute'] + \
            (1 / 60) * match_events['second']
        for period in np.arange(1, match_events['period'].max() + 1, 1):
            if period > 1:
                t_delta = match_events[match_events['period'] == period - 1]['cumulative_mins'].max(
                ) - match_events[match_events['period'] == period]['cumulative_mins'].min()
            else:
                t_delta = 0
            match_events.loc[match_events['period'] ==
                             period, 'cumulative_mins'] += t_delta
        events_out = pd.concat([events_out, match_events])
        return events_out

    df = cumulative_match_mins(df)

    def insert_ball_carries(
            events_df,
            min_carry_length=3,
            max_carry_length=100,
            min_carry_duration=1,
            max_carry_duration=50):
        events_out = pd.DataFrame()
        min_carry_length = 3.0
        max_carry_length = 100.0
        min_carry_duration = 1.0
        max_carry_duration = 50.0
        match_events = events_df.reset_index()
        match_events.loc[match_events['type'] == 'BallRecovery',
                         'endX'] = match_events.loc[match_events['type'] == 'BallRecovery',
                                                    'endX'].fillna(match_events['x'])
        match_events.loc[match_events['type'] == 'BallRecovery',
                         'endY'] = match_events.loc[match_events['type'] == 'BallRecovery',
                                                    'endY'].fillna(match_events['y'])
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
                dx = 105 * (match_event['endX'] - next_evt['x']) / 100
                dy = 68 * (match_event['endY'] - next_evt['y']) / 100
                far_enough = dx ** 2 + dy ** 2 >= min_carry_length ** 2
                not_too_far = dx ** 2 + dy ** 2 <= max_carry_length ** 2
                dt = 60 * (next_evt['cumulative_mins'] -
                           match_event['cumulative_mins'])
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
                    carry['second'] = (((init_next_evt['minute'] * 60 + init_next_evt['second']) + (
                        prev['minute'] * 60 + prev['second'])) / 2) - (carry['minute'] * 60)
                    carry['teamId'] = nex['teamId']
                    carry['x'] = prev['endX']
                    carry['y'] = prev['endY']
                    carry['expandedMinute'] = np.floor(
                        ((init_next_evt['expandedMinute'] * 60 + init_next_evt['second']) + (
                            prev['expandedMinute'] * 60 + prev['second'])) / (
                            2 * 60))
                    carry['period'] = nex['period']
                    carry['type'] = 'Carry'
                    carry['outcomeType'] = 'Successful'
                    carry['qualifiers'] = carry.apply(
                        lambda x: {
                            'type': {
                                'value': 999,
                                'displayName': 'takeOns'},
                            'value': str(take_ons)},
                        axis=1)
                    carry['satisfiedEventsTypes'] = carry.apply(
                        lambda x: [], axis=1)
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
                    carry['cumulative_mins'] = (
                        prev['cumulative_mins'] + init_next_evt['cumulative_mins']) / 2
                    match_carries = pd.concat(
                        [match_carries, carry], ignore_index=True, sort=False)
        match_events_and_carries = pd.concat(
            [match_carries, match_events], ignore_index=True, sort=False)
        match_events_and_carries = match_events_and_carries.sort_values(
            ['period', 'cumulative_mins']).reset_index(drop=True)
        events_out = pd.concat([events_out, match_events_and_carries])
        return events_out

    df = insert_ball_carries(
        df,
        min_carry_length=3,
        max_carry_length=100,
        min_carry_duration=1,
        max_carry_duration=50)
    df = df.reset_index(drop=True)
    df['index'] = range(1, len(df) + 1)
    df = df[['index'] + [col for col in df.columns if col != 'index']]

    df_base = df
    dfxT = df_base.copy()
    dfxT['qualifiers'] = dfxT['qualifiers'].astype(str)
    dfxT = dfxT[(~dfxT['qualifiers'].str.contains('Corner'))]
    dfxT = dfxT[(dfxT['type'].isin(['Pass', 'Carry'])) &
                (dfxT['outcomeType'] == 'Successful')]

    # التحقق من وجود ملف xT_Grid.csv
    if not os.path.exists("xT_Grid.csv"):
        st.error("ملف xT_Grid.csv غير موجود. يرجى توفير الملف لمتابعة التحليل.")
        return None, None, None

    xT = pd.read_csv("xT_Grid.csv", header=None)
    xT = np.array(xT)
    xT_rows, xT_cols = xT.shape

    dfxT['x1_bin_xT'] = pd.cut(dfxT['x'], bins=xT_cols, labels=False)
    dfxT['y1_bin_xT'] = pd.cut(dfxT['y'], bins=xT_rows, labels=False)
    dfxT['x2_bin_xT'] = pd.cut(dfxT['endX'], bins=xT_cols, labels=False)
    dfxT['y2_bin_xT'] = pd.cut(dfxT['endY'], bins=xT_rows, labels=False)

    dfxT['start_zone_value_xT'] = dfxT[['x1_bin_xT', 'y1_bin_xT']].apply(
        lambda x: xT[x[1]][x[0]], axis=1)
    dfxT['end_zone_value_xT'] = dfxT[['x2_bin_xT', 'y2_bin_xT']].apply(
        lambda x: xT[x[1]][x[0]], axis=1)

    dfxT['xT'] = dfxT['end_zone_value_xT'] - dfxT['start_zone_value_xT']
    columns_to_drop = [
        'eventId',
        'minute',
        'second',
        'teamId',
        'x',
        'y',
        'expandedMinute',
        'period',
        'outcomeType',
        'qualifiers',
        'type',
        'satisfiedEventsTypes',
        'isTouch',
        'playerId',
        'endX',
        'endY',
        'relatedEventId',
        'relatedPlayerId',
        'blockedX',
        'blockedY',
        'goalMouthZ',
        'goalMouthY',
        'isShot',
        'cumulative_mins']
    dfxT.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    df = df.merge(dfxT, on='index', how='left')
    df['teamName'] = df['teamId'].map(teams_dict)
    team_names = list(teams_dict.values())
    opposition_dict = {team_names[i]: team_names[1 - i]
                       for i in range(len(team_names))}
    df['oppositionTeamName'] = df['teamName'].map(opposition_dict)

    df['x'] = df['x'] * 1.05
    df['y'] = df['y'] * 0.68
    df['endX'] = df['endX'] * 1.05
    df['endY'] = df['endY'] * 0.68
    df['goalMouthY'] = df['goalMouthY'] * 0.68

    columns_to_drop = [
        'height',
        'weight',
        'age',
        'isManOfTheMatch',
        'field',
        'stats',
        'subbedInPlayerId',
        'subbedOutPeriod',
        'subbedOutExpandedMinute',
        'subbedInPeriod',
        'subbedInExpandedMinute',
        'subbedOutPlayerId',
        'teamId']
    dfp.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    df = df.merge(dfp, on='playerId', how='left')

    df['qualifiers'] = df['qualifiers'].astype(str)
    df['prog_pass'] = np.where((df['type'] == 'Pass'), np.sqrt(
        (105 - df['x'])**2 + (34 - df['y'])**2) - np.sqrt((105 - df['endX'])**2 + (34 - df['endY'])**2), 0)
    df['prog_carry'] = np.where((df['type'] == 'Carry'), np.sqrt(
        (105 - df['x'])**2 + (34 - df['y'])**2) - np.sqrt((105 - df['endX'])**2 + (34 - df['endY'])**2), 0)
    df['pass_or_carry_angle'] = np.degrees(
        np.arctan2(df['endY'] - df['y'], df['endX'] - df['x']))

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
            return parts[0][0] + ". " + parts[1][0] + \
                ". " + " ".join(parts[2:])

    df['shortName'] = df['name'].apply(get_short_name)
    columns_to_drop2 = ['id']
    df.drop(columns=columns_to_drop2, inplace=True, errors='ignore')

    def get_possession_chains(events_df, chain_check, suc_evts_in_chain):
        events_out = pd.DataFrame()
        match_events_df = events_df.reset_index()
        match_pos_events_df = match_events_df[~match_events_df['type'].isin(['OffsideGiven', 'CornerAwarded', 'Start', 'Card', 'SubstitutionOff',
                                                                             'SubstitutionOn', 'FormationChange', 'FormationSet', 'End'])].copy()
        match_pos_events_df['outcomeBinary'] = (
            match_pos_events_df['outcomeType'] .apply(
                lambda x: 1 if x == 'Successful' else 0))
        match_pos_events_df['teamBinary'] = (
            match_pos_events_df['teamName'] .apply(
                lambda x: 1 if x == min(
                    match_pos_events_df['teamName']) else 0))
        match_pos_events_df['goalBinary'] = (
            (match_pos_events_df['type'] == 'Goal') .astype(int).diff(
                periods=1).apply(
                lambda x: 1 if x < 0 else 0))
        pos_chain_df = pd.DataFrame()
        for n in np.arange(1, chain_check):
            pos_chain_df[f'evt_{n}_same_team'] = abs(
                match_pos_events_df['teamBinary'].diff(periods=-n))
            pos_chain_df[f'evt_{n}_same_team'] = pos_chain_df[f'evt_{n}_same_team'].apply(
                lambda x: 1 if x > 1 else x)
        pos_chain_df['enough_evt_same_team'] = pos_chain_df.sum(axis=1).apply(
            lambda x: 1 if x < chain_check - suc_evts_in_chain else 0)
        pos_chain_df['enough_evt_same_team'] = pos_chain_df['enough_evt_same_team'].diff(
            periods=1)
        pos_chain_df[pos_chain_df['enough_evt_same_team'] < 0] = 0
        match_pos_events_df['period'] = pd.to_numeric(
            match_pos_events_df['period'], errors='coerce')
        pos_chain_df['upcoming_ko'] = 0
        for ko in match_pos_events_df[(match_pos_events_df['goalBinary'] == 1) | (
                match_pos_events_df['period'].diff(periods=1))].index.values:
            ko_pos = match_pos_events_df.index.to_list().index(ko)
            pos_chain_df.iloc[ko_pos - suc_evts_in_chain:ko_pos,
                              pos_chain_df.columns.get_loc('upcoming_ko')] = 1
        pos_chain_df['valid_pos_start'] = (
            pos_chain_df.fillna(0)['enough_evt_same_team'] -
            pos_chain_df.fillna(0)['upcoming_ko'])
        pos_chain_df['kick_off_period_change'] = match_pos_events_df['period'].diff(
            periods=1)
        pos_chain_df['kick_off_goal'] = (
            (match_pos_events_df['type'] == 'Goal') .astype(int).diff(
                periods=1).apply(
                lambda x: 1 if x < 0 else 0))
        pos_chain_df.loc[pos_chain_df['kick_off_period_change']
                         == 1, 'valid_pos_start'] = 1
        pos_chain_df.loc[pos_chain_df['kick_off_goal']
                         == 1, 'valid_pos_start'] = 1
        pos_chain_df['teamName'] = match_pos_events_df['teamName']
        pos_chain_df.loc[pos_chain_df.head(1).index, 'valid_pos_start'] = 1
        pos_chain_df.loc[pos_chain_df.head(1).index, 'possession_id'] = 1
        pos_chain_df.loc[pos_chain_df.head(
            1).index, 'possession_team'] = pos_chain_df.loc[pos_chain_df.head(1).index, 'teamName']
        valid_pos_start_id = pos_chain_df[pos_chain_df['valid_pos_start'] > 0].index
        possession_id = 2
        for idx in np.arange(1, len(valid_pos_start_id)):
            current_team = pos_chain_df.loc[valid_pos_start_id[idx], 'teamName']
            previous_team = pos_chain_df.loc[valid_pos_start_id[idx - 1], 'teamName']
            if ((previous_team == current_team) & (pos_chain_df.loc[valid_pos_start_id[idx], 'kick_off_goal'] != 1) &
                    (pos_chain_df.loc[valid_pos_start_id[idx], 'kick_off_period_change'] != 1)):
                pos_chain_df.loc[valid_pos_start_id[idx],
                                 'possession_id'] = np.nan
            else:
                pos_chain_df.loc[valid_pos_start_id[idx],
                                 'possession_id'] = possession_id
                pos_chain_df.loc[valid_pos_start_id[idx],
                                 'possession_team'] = pos_chain_df.loc[valid_pos_start_id[idx],
                                                                       'teamName']
                possession_id += 1
        match_events_df = pd.merge(match_events_df,
                                   pos_chain_df[['possession_id',
                                                 'possession_team']],
                                   how='left',
                                   left_index=True,
                                   right_index=True)
        match_events_df[['possession_id', 'possession_team']] = (
            match_events_df[['possession_id', 'possession_team']].fillna(method='ffill'))
        match_events_df[['possession_id', 'possession_team']] = (
            match_events_df[['possession_id', 'possession_team']].fillna(method='bfill'))
        events_out = pd.concat([events_out, match_events_df])
        return events_out

    df = get_possession_chains(df, 5, 3)
    df['period'] = df['period'].replace({1: 'FirstHalf',
                                         2: 'SecondHalf',
                                         3: 'FirstPeriodOfExtraTime',
                                         4: 'SecondPeriodOfExtraTime',
                                         5: 'PenaltyShootout',
                                         14: 'PostGame',
                                         16: 'PreMatch'})
    df = df[df['period'] != 'PenaltyShootout']
    df = df.reset_index(drop=True)
    return df, teams_dict, players_df

# دالة شبكة التمريرات


def pass_network(ax, team_name, col, phase_tag, hteamName, ateamName, hgoal_count, agoal_count, hteamID, ateamID):
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
    
    # التعامل مع قيم NaN في isFirstEleven
    avg_locs_df['isFirstEleven'] = avg_locs_df['isFirstEleven'].fillna(False)
    avg_locs_df['shirtNo'] = avg_locs_df['shirtNo'].fillna(0)  # تعبئة shirtNo بـ 0 إذا كانت مفقودة
    avg_locs_df['position'] = avg_locs_df['position'].fillna('Unknown')  # تعبئة position بـ 'Unknown' إذا كانت مفقودة
    
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
        if row['isFirstEleven']:
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
    
    # التعامل مع تصفية Forwards_height
    Forwards_height = avg_locs_df[avg_locs_df['isFirstEleven'] == True].sort_values(by='avg_x', ascending=False).head(2)
    fwd_line_h = round(Forwards_height['avg_x'].mean(), 2) if not Forwards_height.empty else avgph
    
    ymid = [0, 0, 68, 68]
    xmid = [def_line_h, fwd_line_h, fwd_line_h, def_line_h]
    ax.fill(ymid, xmid, col, alpha=0.2)
    v_comp = round((1 - ((fwd_line_h - def_line_h) / 105)) * 100, 2)
    
    # إضافة النتيجة
    score_text = reshape_arabic_text(f"{hteamName} {hgoal_count} - {agoal_count} {ateamName}")
    ax.text(34, 120, score_text, color='white', fontsize=16, ha='center', va='center', weight='bold')
    
    # إضافة شعاري الفريقين من FotMob
    try:
        # استخدام معرفات FotMob
        hteamID_fotmob = fotmob_team_ids.get(hteamName, hteamID)
        ateamID_fotmob = fotmob_team_ids.get(ateamName, ateamID)
        
        home_logo_url = f"https://images.fotmob.com/image_resources/logo/teamlogo/{hteamID_fotmob}.png"
        away_logo_url = f"https://images.fotmob.com/image_resources/logo/teamlogo/{ateamID_fotmob}.png"
        
        home_logo = Image.open(urlopen(home_logo_url))
        away_logo = Image.open(urlopen(away_logo_url))
        
        # تغيير حجم الصور
        home_logo = home_logo.resize((50, 50), Image.Resampling.LANCZOS)
        away_logo = away_logo.resize((50, 50), Image.Resampling.LANCZOS)
        
        # إضافة شعار الفريق المضيف
        home_logo_ax = ax.inset_axes([0.05, 0.95, 0.07, 0.07], transform=ax.transAxes)
        home_logo_ax.imshow(home_logo)
        home_logo_ax.axis('off')
        
        # إضافة شعار الفريق الضيف
        away_logo_ax = ax.inset_axes([0.88, 0.95, 0.07, 0.07], transform=ax.transAxes)
        away_logo_ax.imshow(away_logo)
        away_logo_ax.axis('off')
    except Exception as e:
        st.warning(f"فشل في تحميل شعارات الفريقين من FotMob: {str(e)}")
        ax.text(5, 115, reshape_arabic_text(hteamName), color='white', fontsize=12, ha='left', va='center')
        ax.text(63, 115, reshape_arabic_text(ateamName), color='white', fontsize=12, ha='right', va='center')
    
    if phase_tag == 'Full Time':
        ax.text(34, 115, reshape_arabic_text('الوقت بالكامل: 0-90 دقيقة'), color='white', fontsize=14, ha='center', va='center', weight='bold')
        ax.text(34, 112, reshape_arabic_text(f'إجمالي التمريرات: {len(total_pass)} | الناجحة: {len(accrt_pass)} | الدقة: {accuracy}%'), color='white', fontsize=12, ha='center', va='center')
    elif phase_tag == 'First Half':
        ax.text(34, 115, reshape_arabic_text('الشوط الأول: 0-45 دقيقة'), color='white', fontsize=14, ha='center', va='center', weight='bold')
        ax.text(34, 112, reshape_arabic_text(f'إجمالي التمريرات: {len(total_pass)} | الناجحة: {len(accrt_pass)} | الدقة: {accuracy}%'), color='white', fontsize=12, ha='center', va='center')
    elif phase_tag == 'Second Half':
        ax.text(34, 115, reshape_arabic_text('الشوط الثاني: 45-90 دقيقة'), color='white', fontsize=14, ha='center', va='center', weight='bold')
        ax.text(34, 112, reshape_arabic_text(f'إجمالي التمريرات: {len(total_pass)} | الناجحة: {len(accrt_pass)} | الدقة: {accuracy}%'), color='white', fontsize=12, ha='center', va='center')
    
    ax.text(34, -6, reshape_arabic_text(f"على الكرة\nالتماسك العمودي (المنطقة المظللة): {v_comp}%"), color='white', fontsize=12, ha='center', va='center', weight='bold')
    
    return pass_btn

# دالة مناطق السيطرة


def team_domination_zones(
        ax,
        phase_tag,
        hteamName,
        ateamName,
        hcol,
        acol,
        bg_color,
        line_color,
        gradient_colors):
    """رسم مناطق السيطرة لكل فريق بناءً على اللمسات في كل منطقة بتصميم عصري."""
    df = st.session_state.df.copy()
    if phase_tag == 'First Half':
        df = df[df['period'] == 'FirstHalf']
    elif phase_tag == 'Second Half':
        df = df[df['period'] == 'SecondHalf']

    # تصفية الأحداث التي تتضمن لمسات مفتوحة (Open-Play Touches)
    df_touches = df[df['isTouch']].copy()

    # التحقق من وجود بيانات لمسات
    if df_touches.empty:
        ax.text(
            52.5,
            34,
            reshape_arabic_text('لا توجد بيانات لمسات متاحة'),
            color='white',
            fontsize=14,
            ha='center',
            va='center',
            weight='bold')
        return

    # تقسيم الملعب إلى شبكة (مثل 5x4)
    x_bins = np.linspace(0, 105, 7)  # 6 أعمدة
    y_bins = np.linspace(0, 68, 6)   # 5 صفوف
    df_touches['x_bin'] = pd.cut(
        df_touches['x'],
        bins=x_bins,
        labels=False,
        include_lowest=True)
    df_touches['y_bin'] = pd.cut(
        df_touches['y'],
        bins=y_bins,
        labels=False,
        include_lowest=True)

    # حساب عدد اللمسات لكل فريق في كل منطقة
    touches_by_team = df_touches.groupby(['teamName', 'x_bin', 'y_bin']).size(
    ).unstack(fill_value=0).stack().reset_index(name='touch_count')

    # إنشاء قاموس لتخزين اللمسات لكل فريق
    hteam_touches = touches_by_team[touches_by_team['teamName'] == hteamName].pivot(
        index='y_bin', columns='x_bin', values='touch_count').fillna(0)
    ateam_touches = touches_by_team[touches_by_team['teamName'] == ateamName].pivot(
        index='y_bin', columns='x_bin', values='touch_count').fillna(0)

    # محاذاة المصفوفات للتأكد من أن لديهم نفس الأبعاد
    hteam_touches = hteam_touches.reindex(
        index=range(5),
        columns=range(6),
        fill_value=0)  # 5 صفوف، 6 أعمدة
    ateam_touches = ateam_touches.reindex(
        index=range(5), columns=range(6), fill_value=0)

    # حساب النسبة المئوية للسيطرة في كل منطقة
    total_touches = hteam_touches + ateam_touches
    hteam_percentage = hteam_touches / total_touches.replace(0, np.nan) * 100
    ateam_percentage = ateam_touches / total_touches.replace(0, np.nan) * 100

    # تحديد مناطق السيطرة
    domination = np.zeros((5, 6), dtype=object)
    for i in range(5):
        for j in range(6):
            h_percent = hteam_percentage.iloc[i, j]
            a_percent = ateam_percentage.iloc[i, j]
            if pd.isna(h_percent) or pd.isna(a_percent):
                domination[i, j] = 'contested'
            elif h_percent > 55:
                domination[i, j] = 'home'
            elif a_percent > 55:
                domination[i, j] = 'away'
            else:
                domination[i, j] = 'contested'

    # ترك فراغ كما طلبت
    # (الفراغ موجود هنا)

    # إعداد تصميم عصري
    # تدرج لوني داكن وعصري (من الأسود إلى الأزرق الداكن)
    gradient = LinearSegmentedColormap.from_list(
        "modern_gradient", ['#0D1B2A', '#1B263B'], N=100)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = Y
    ax.imshow(
        Z,
        extent=[
            0,
            105,
            0,
            68],
        cmap=gradient,
        alpha=0.9,
        aspect='auto',
        zorder=0)

    # رسم المناطق مع تأثيرات ظلال
    bin_width = 105 / 6  # عرض المربع
    bin_height = 68 / 5  # ارتفاع المربع
    for i in range(5):
        for j in range(6):
            x_start = j * bin_width
            y_start = i * bin_height
            if domination[i, j] == 'home':
                color = hcol
                alpha = 0.7
                percentage = hteam_percentage.iloc[i, j]
            elif domination[i, j] == 'away':
                color = acol
                alpha = 0.7
                percentage = ateam_percentage.iloc[i, j]
            else:
                color = '#555555'  # لون رمادي داكن للمناطق المتنازع عليها
                alpha = 0.3
                percentage = None

            # رسم المستطيل مع ظل
            rect = patches.Rectangle(
                (x_start,
                 y_start),
                bin_width,
                bin_height,
                linewidth=1.5,
                edgecolor=line_color,
                facecolor=color,
                alpha=alpha,
                zorder=1)
            rect.set_path_effects([path_effects.withStroke(
                linewidth=3, foreground='black', alpha=0.5)])  # إضافة ظل
            ax.add_patch(rect)

            # إضافة النسبة المئوية في منتصف المستطيل
            if percentage is not None and not pd.isna(percentage):
                text = ax.text(x_start + bin_width / 2,
                               y_start + bin_height / 2,
                               f'{percentage:.1f}%',
                               color='white',
                               fontsize=8,
                               ha='center',
                               va='center',
                               weight='bold',
                               zorder=3)
                text.set_path_effects(
                    [path_effects.withStroke(linewidth=2, foreground='black')])

    # رسم الملعب في الأعلى
    pitch = Pitch(
        pitch_type='uefa',
        line_color=line_color,
        linewidth=2,
        corner_arcs=True)
    pitch.draw(ax=ax)

    # ضبط zorder لخطوط الملعب يدويًا
    for artist in ax.get_children():
        if isinstance(
                artist,
                plt.Line2D):  # التحقق من أن العنصر هو خط (مثل خطوط الملعب)
            artist.set_zorder(2)

    # إضافة أسهم الهجوم بتصميم عصري (أعلى وأسفل)
    # سهم الفريق المضيف (أعلى)
    arrow1 = ax.arrow(
        5,
        72,
        20,
        0,
        head_width=2,
        head_length=2,
        fc=hcol,
        ec='white',
        linewidth=1.5,
        zorder=4)
    arrow1.set_path_effects([path_effects.withStroke(
        linewidth=3, foreground='black', alpha=0.5)])  # ظل للسهم
    text1 = ax.text(
        15,
        76,
        reshape_arabic_text(
            f'اتجاه هجوم {hteamName}'),
        color=hcol,
        fontsize=10,
        ha='center',
        va='center',
        zorder=4)
    text1.set_path_effects(
        [path_effects.withStroke(linewidth=2, foreground='black')])

    # سهم الفريق الضيف (أسفل)
    arrow2 = ax.arrow(100, -4, -20, 0, head_width=2, head_length=2,
                      fc=acol, ec='white', linewidth=1.5, zorder=4)
    arrow2.set_path_effects([path_effects.withStroke(
        linewidth=3, foreground='black', alpha=0.5)])
    text2 = ax.text(
        90,
        -8,
        reshape_arabic_text(
            f'اتجاه هجوم {ateamName}'),
        color=acol,
        fontsize=10,
        ha='center',
        va='center',
        zorder=4)
    text2.set_path_effects(
        [path_effects.withStroke(linewidth=2, foreground='black')])

    # إضافة النصوص مع تحسين التصميم
    period_text = 'الشوط الأول: 0-45 دقيقة' if phase_tag == 'First Half' else 'الشوط الثاني: 45-90 دقيقة' if phase_tag == 'Second Half' else 'الوقت بالكامل: 0-90 دقيقة'
    period = ax.text(
        52.5,
        82,
        reshape_arabic_text(period_text),
        color='white',
        fontsize=14,
        ha='center',
        va='center',
        weight='bold',
        zorder=4)
    period.set_path_effects(
        [path_effects.withStroke(linewidth=2, foreground='black')])

    teams = ax.text(
        52.5,
        78,
        reshape_arabic_text(
            f'{hteamName} | المتنازع عليها | {ateamName}'),
        color='white',
        fontsize=12,
        ha='center',
        va='center',
        zorder=4)
    teams.set_path_effects(
        [path_effects.withStroke(linewidth=2, foreground='black')])

    # ملاحظات توضيحية
    note1 = ax.text(
        52.5,
        -12,
        reshape_arabic_text('* المنطقة المسيطر عليها: الفريق لديه أكثر من 55% من اللمسات'),
        color='white',
        fontsize=8,
        ha='center',
        va='center',
        zorder=4)
    note1.set_path_effects(
        [path_effects.withStroke(linewidth=2, foreground='black')])
    note2 = ax.text(
        52.5,
        -16,
        reshape_arabic_text('* المتنازع عليها: الفريق لديه 45-55% من اللمسات'),
        color='white',
        fontsize=8,
        ha='center',
        va='center',
        zorder=4)
    note2.set_path_effects(
        [path_effects.withStroke(linewidth=2, foreground='black')])


def attack_zones_analysis(fig, ax, hteamName, ateamName, hcol, acol):
    # استخراج الأحداث الهجومية في الثلث الأخير (x >= 70)
    attack_events = st.session_state.df[
        (st.session_state.df['type'].isin(['Pass', 'Goal', 'MissedShots', 'SavedShot', 'ShotOnPost', 'TakeOn', 'Carry'])) &
        (st.session_state.df['outcomeType'] == 'Successful') &
        (st.session_state.df['x'] >= 70)
    ].copy()
    
    # تصنيف الأحداث حسب المنطقة بناءً على إحداثي y
    def classify_zone(y):
        if y < 22.67:
            return 'اليسار'
        elif y <= 45.33:
            return 'العمق'
        else:
            return 'اليمين'
    
    attack_events['zone'] = attack_events['y'].apply(classify_zone)
    # تجميع الأحداث حسب الفريق والمنطقة
    attack_summary = attack_events.groupby(['teamName', 'zone']).size().unstack(fill_value=0).reset_index()
    attack_summary = attack_summary.reindex(columns=['teamName', 'اليسار', 'العمق', 'اليمين'], fill_value=0)
    
    # حساب النسب المئوية
    attack_summary['مجموع'] = attack_summary[['اليسار', 'العمق', 'اليمين']].sum(axis=1)
    attack_summary['اليسار_%'] = (attack_summary['اليسار'] / attack_summary['مجموع'] * 100).round(1)
    attack_summary['العمق_%'] = (attack_summary['العمق'] / attack_summary['مجموع'] * 100).round(1)
    attack_summary['اليمين_%'] = (attack_summary['اليمين'] / attack_summary['مجموع'] * 100).round(1)
    
    # إعداد الملعب
    pitch = VerticalPitch(pitch_type='uefa', corner_arcs=True, linewidth=1.5, line_color='#ffffff', half=True)
    
    # إعداد التدرج اللوني
    gradient = LinearSegmentedColormap.from_list("pitch_gradient", ['#003087', '#d00000'], N=100)
    
    # عرض الفريقين معًا أو بشكل منفصل
    axes = [ax1, ax2] if display_together else [ax1]
    teams = [(hteamName, hcol, hteamID), (ateamName, acol, ateamID)] if display_together else [(hteamName, hcol, hteamID)]
    
    for ax, (team_name, team_color, team_id) in zip(axes, teams):
        pitch.draw(ax=ax)
        
        # خلفية التدرج
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(x, y)
        Z = Y
        ax.imshow(Z, extent=[0, 68, 52.5, 105], cmap=gradient, alpha=0.8, aspect='auto', zorder=0)
        pitch.draw(ax=ax)
        
        # استخراج بيانات الفريق
        team_events = attack_events[attack_events['teamName'] == team_name]
        team_summary = attack_summary[attack_summary['teamName'] == team_name]
        
        if not team_events.empty:
            # حساب عدد الأحداث في كل منطقة
            left_count = team_summary['اليسار'].iloc[0] if not team_summary.empty else 0
            center_count = team_summary['العمق'].iloc[0] if not team_summary.empty else 0
            right_count = team_summary['اليمين'].iloc[0] if not team_summary.empty else 0
            
            # إحداثيات مركز كل منطقة للأسهم
            zones = {
                'اليسار': {'x': 85, 'y': 11.335, 'count': left_count, 'dx': 0, 'dy': 10},
                'العمق': {'x': 85, 'y': 34, 'count': center_count, 'dx': 0, 'dy': 10},
                'اليمين': {'x': 85, 'y': 56.665, 'count': right_count, 'dx': 0, 'dy': 10}
            }
            
            # رسم الأسهم
            max_count = max(left_count, center_count, right_count, 1)  # تجنب القسمة على صفر
            for zone, info in zones.items():
                if info['count'] > 0:
                    arrow_width = (info['count'] / max_count) * 0.01  # تحجيم السهم بناءً على العدد
                    ax.arrow(
                        info['y'], info['x'], info['dy'], info['dx'],
                        width=arrow_width, head_width=arrow_width*2, head_length=2,
                        fc=team_color, ec='white', alpha=0.9, zorder=2
                    )
            
            # إضافة النسب المئوية كنصوص
            left_pct = team_summary['اليسار_%'].iloc[0] if not team_summary.empty else 0
            center_pct = team_summary['العمق_%'].iloc[0] if not team_summary.empty else 0
            right_pct = team_summary['اليمين_%'].iloc[0] if not team_summary.empty else 0
            
            ax.text(11.335, 95, f'{left_pct}%', color='white', fontsize=12, ha='center', va='center', weight='bold', bbox=dict(facecolor=team_color, alpha=0.5, edgecolor='none'))
            ax.text(34, 95, f'{center_pct}%', color='white', fontsize=12, ha='center', va='center', weight='bold', bbox=dict(facecolor=team_color, alpha=0.5, edgecolor='none'))
            ax.text(56.665, 95, f'{right_pct}%', color='white', fontsize=12, ha='center', va='center', weight='bold', bbox=dict(facecolor=team_color, alpha=0.5, edgecolor='none'))
        
        # إضافة خطوط تقسيم المناطق
        ax.axvline(x=22.67, color='white', linestyle='--', alpha=0.5)
        ax.axvline(x=45.33, color='white', linestyle='--', alpha=0.5)
        
        # إضافة تسميات المناطق
        ax.text(11.335, 100, reshape_arabic_text('اليسار'), color='white', fontsize=10, ha='center', va='center')
        ax.text(34, 100, reshape_arabic_text('العمق'), color='white', fontsize=10, ha='center', va='center')
        ax.text(56.665, 100, reshape_arabic_text('اليمين'), color='white', fontsize=10, ha='center', va='center')
        
        # إضافة عنوان
        ax.text(34, 110, reshape_arabic_text(f'مناطق الهجوم: {team_name}'), 
                color='white', fontsize=14, ha='center', va='center', weight='bold')
        
        # إضافة شعار الفريق من FotMob
        try:
            teamID_fotmob = fotmob_team_ids.get(team_name, team_id)
            logo_url = f"https://images.fotmob.com/image_resources/logo/teamlogo/{teamID_fotmob}.png"
            logo = Image.open(urlopen(logo_url)).resize((50, 50), Image.Resampling.LANCZOS)
            logo_ax = ax.inset_axes([0.85, 0.85, 0.1, 0.1], transform=ax.transAxes)
            logo_ax.imshow(logo)
            logo_ax.axis('off')
        except Exception as e:
            st.warning(f"فشل في تحميل شعار {team_name} من FotMob: {str(e)}")
            ax.text(63, 105, reshape_arabic_text(team_name), color='white', fontsize=12, ha='right', va='center')
    
    # إنشاء رسم بياني شريطي للمقارنة
    fig_bar, ax_bar = plt.subplots(figsize=(8, 6), facecolor=bg_color)
    ax_bar.set_facecolor(bg_color)
    
    zones = ['اليسار', 'العمق', 'اليمين']
    hteam_counts = attack_summary[attack_summary['teamName'] == hteamName][zones].iloc[0] if hteamName in attack_summary['teamName'].values else [0, 0, 0]
    ateam_counts = attack_summary[attack_summary['teamName'] == ateamName][zones].iloc[0] if ateamName in attack_summary['teamName'].values else [0, 0, 0]
    
    bar_width = 0.35
    x = np.arange(len(zones))
    
    ax_bar.bar(x - bar_width/2, hteam_counts, bar_width, label=reshape_arabic_text(hteamName), color=hcol)
    ax_bar.bar(x + bar_width/2, ateam_counts, bar_width, label=reshape_arabic_text(ateamName), color=acol)
    
    ax_bar.set_xlabel(reshape_arabic_text('المنطقة'), color='white', fontsize=12)
    ax_bar.set_ylabel(reshape_arabic_text('عدد الأحداث الهجومية'), color='white', fontsize=12)
    ax_bar.set_title(reshape_arabic_text('توزيع الأحداث الهجومية حسب المنطقة'), color='white', fontsize=14)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([reshape_arabic_text(z) for z in zones], color='white')
    ax_bar.tick_params(axis='y', colors='white')
    ax_bar.legend()
    
    # إرجاع الجدول والرسومات
    attack_summary = attack_summary.rename(columns={
        'teamName': 'الفريق',
        'اليسار': 'اليسار',
        'العمق': 'العمق',
        'اليمين': 'اليمين',
        'اليسار_%': 'اليسار (%)',
        'العمق_%': 'العمق (%)',
        'اليمين_%': 'اليمين (%)'
    })
    
    return attack_summary, fig, fig_bar

# واجهة Streamlit
st.title("تحليل مباراة كرة القدم")
match_url = st.text_input(
    "أدخل رابط المباراة من WhoScored:",
    value="https://1xbet.whoscored.com/Matches/1809770/Live/Europe-Europa-League-2023-2024-West-Ham-Bayer-Leverkusen"
)
uploaded_file = st.file_uploader(
    "أو قم بتحميل ملف JSON (اختياري):", type="json")

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
            st.session_state.df, st.session_state.teams_dict, st.session_state.players_df = get_event_data(
                st.session_state.json_data)
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

    homedf = st.session_state.df[(
        st.session_state.df['teamName'] == hteamName)]
    awaydf = st.session_state.df[(
        st.session_state.df['teamName'] == ateamName)]
    hxT = homedf['xT'].sum().round(2)
    axT = awaydf['xT'].sum().round(2)

    hgoal_count = len(homedf[(homedf['teamName'] == hteamName) & (
        homedf['type'] == 'Goal') & (~homedf['qualifiers'].str.contains('OwnGoal'))])
    agoal_count = len(awaydf[(awaydf['teamName'] == ateamName) & (
        awaydf['type'] == 'Goal') & (~awaydf['qualifiers'].str.contains('OwnGoal'))])
    hgoal_count += len(awaydf[(awaydf['teamName'] == ateamName) & (
        awaydf['type'] == 'Goal') & (awaydf['qualifiers'].str.contains('OwnGoal'))])
    agoal_count += len(homedf[(homedf['teamName'] == hteamName) & (
        homedf['type'] == 'Goal') & (homedf['qualifiers'].str.contains('OwnGoal'))])

    # التحقق من وجود ملف teams_name_and_id.csv
    if not os.path.exists("teams_name_and_id.csv"):
        st.error(
            "ملف teams_name_and_id.csv غير موجود. يرجى توفير الملف لمتابعة التحليل.")
        st.stop()

    df_teamNameId = pd.read_csv("teams_name_and_id.csv")
    hftmb_tid = df_teamNameId[df_teamNameId['teamName'] ==
                              hteamName]['teamId'].iloc[0] if not df_teamNameId[df_teamNameId['teamName'] == hteamName].empty else 0
    aftmb_tid = df_teamNameId[df_teamNameId['teamName'] ==
                              ateamName]['teamId'].iloc[0] if not df_teamNameId[df_teamNameId['teamName'] == ateamName].empty else 0

    st.header(f'{hteamName} {hgoal_count} - {agoal_count} {ateamName}')

    # علامات التبويب
    tab1, tab2, tab3, tab4 = st.tabs(
        ['تحليل الفريق', 'تحليل اللاعبين', 'إحصائيات المباراة', 'أفضل اللاعبين'])

with tab1:
    an_tp = st.selectbox('نوع التحليل:', [
        'شبكة التمريرات',
        'مناطق الهجوم',  # خيار جديد

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
        
        # حساب عدد الأهداف
        homedf = st.session_state.df[(st.session_state.df['teamName'] == hteamName)]
        awaydf = st.session_state.df[(st.session_state.df['teamName'] == ateamName)]
        hgoal_count = len(homedf[(homedf['teamName'] == hteamName) & (homedf['type'] == 'Goal') & (~homedf['qualifiers'].str.contains('OwnGoal'))])
        agoal_count = len(awaydf[(awaydf['teamName'] == ateamName) & (awaydf['type'] == 'Goal') & (~awaydf['qualifiers'].str.contains('OwnGoal'))])
        hgoal_count += len(awaydf[(awaydf['teamName'] == ateamName) & (awaydf['type'] == 'Goal') & (awaydf['qualifiers'].str.contains('OwnGoal'))])
        agoal_count += len(homedf[(homedf['teamName'] == hteamName) & (homedf['type'] == 'Goal') & (homedf['qualifiers'].str.contains('OwnGoal'))])
        
        # تحديد معرفات الفرق
        hteamID = list(st.session_state.teams_dict.keys())[0]
        ateamID = list(st.session_state.teams_dict.keys())[1]
        
        # تحديد اللون بناءً على اختيار الفريق
        col = hcol if team_choice == hteamName else acol
        
        # إنشاء الرسم
        fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)
        
        # استدعاء pass_network مع جميع الوسائط المطلوبة
        pass_btn = pass_network(
            ax,
            team_choice,
            col,
            phase_tag,
            hteamName,
            ateamName,
            hgoal_count,
            agoal_count,
            hteamID,
            ateamID
        )
        st.pyplot(fig)
        st.dataframe(pass_btn, hide_index=True)
        
    elif an_tp == 'مناطق الهجوم':
        st.subheader('تحليل مناطق الهجوم')
        fig, ax = plt.subplots(figsize=(10, 10), facecolor=bg_color)
        attack_summary, fig_heatmap, fig_bar = attack_zones_analysis(fig, ax, hteamName, ateamName, hcol, acol)
        
        # عرض خريطة حرارية
        st.pyplot(fig_heatmap)
        
        # عرض الرسم البياني الشريطي
        st.pyplot(fig_bar)
        
        # عرض الجدول الإحصائي
        st.subheader('إحصائيات مناطق الهجوم')
        attack_summary = attack_summary.rename(columns={
            'teamName': 'الفريق',
            'اليسار': 'اليسار',
            'العمق': 'العمق',
            'اليمين': 'اليمين'
        })
        st.dataframe(attack_summary, hide_index=True)    

    elif an_tp == reshape_arabic_text('Team Domination Zones'):
        st.subheader(reshape_arabic_text('مناطق سيطرة الفريق'))
        phase_tag = st.selectbox(
            'اختر الفترة:', [
                'Full Time', 'First Half', 'Second Half'], key='phase_tag_domination')
        fig, ax = plt.subplots(figsize=(12, 8), facecolor=bg_color)
        team_domination_zones(
            ax,
            phase_tag,
            hteamName,
            ateamName,
            hcol,
            acol,
            bg_color,
            line_color,
            gradient_colors)
        # إضافة عنوان أعلى الرسم
        fig.text(
            0.5,
            0.98,
            reshape_arabic_text(
                f'{hteamName} {hgoal_count} - {agoal_count} {ateamName}'),
            fontsize=16,
            fontweight='bold',
            ha='center',
            va='center',
            color='white')
        fig.text(0.5, 0.94, reshape_arabic_text('مناطق السيطرة'),
                 fontsize=14, ha='center', va='center', color='white')
        st.pyplot(fig)
