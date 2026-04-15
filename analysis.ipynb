#Import necessary libraries
from logging import shutdown
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
import statsmodels.api as sm

#Upload CSV file
uploaded = files.upload()
df = pd.read_csv("whl_2025 - whl_2025.csv")

# -------------------------
# EMPTY NET ADJUSTMENT
# Discount xG generated against an empty net to 25% of face value.
# When a team pulls their goalie (empty_net offensive line), any xG
# the opposing team generates is cheap — scoring into an empty net
# tells us nothing about their offensive quality. We keep the trailing
# team's xG at full value since they're generating against a real goalie.
# -------------------------
df["home_xg"] = np.where(df["away_off_line"] == "empty_net",
                          df["home_xg"] * 0.25,
                          df["home_xg"])

df["away_xg"] = np.where(df["home_off_line"] == "empty_net",
                          df["away_xg"] * 0.25,
                          df["away_xg"])

# -------------------------
# GAME LEVEL RESULTS
# Collapse the raw segment-level data into one row per game.
# home_goals/away_goals are summed across all segments.
# went_ot uses max() — if any segment has went_ot = 1, the game went to OT.
# -------------------------
game_results = (
    df.groupby("game_id")
      .agg({
          "home_team": "first",
          "away_team": "first",
          "home_goals": "sum",
          "away_goals": "sum",
          "went_ot": "max"
      })
      .reset_index()
)

game_results["goal_diff"] = (
    game_results["home_goals"] - game_results["away_goals"]
)

game_results["home_win"] = (game_results["goal_diff"] > 0).astype(int)
game_results["away_win"] = (game_results["goal_diff"] < 0).astype(int)

# -------------------------
# OT DURATION
# For OT games only, sum all segment TOI, divide by 2 to correct for
# double-counting, then subtract 3600 to isolate OT time in seconds.
# Clip at 1 second minimum to prevent zero/negative duration edge cases.
# -------------------------

ot_game_ids = game_results.loc[game_results["went_ot"] == 1, "game_id"]

ot_duration = (
    df[(df["game_id"].isin(ot_game_ids)) & (df["went_ot"] == 1)]
    .groupby("game_id")["toi"]
    .sum() / 2
).clip(lower=1)

game_results["ot_duration"] = game_results["game_id"].map(ot_duration)

# -------------------------
# OT WEIGHTING
# Regulation wins are worth 1.0.
# OT win weight decays linearly from 0.90 (1 sec OT) to 0.55 (2400 sec OT).
# OT loss weight is the complement (1 - win weight).
# Weights are computed per row and assigned explicitly to avoid
# index misalignment bugs from boolean mask + Series operations.
# -------------------------
MAX_OT_SECONDS = 2400

game_results["home_weighted_result"] = 0.0
game_results["away_weighted_result"] = 0.0

# Regulation wins
reg_mask = game_results["went_ot"] == 0
game_results.loc[reg_mask & (game_results["home_win"] == 1), "home_weighted_result"] = 1.0
game_results.loc[reg_mask & (game_results["away_win"] == 1), "away_weighted_result"] = 1.0

# Compute OT weights row by row to avoid Series index alignment issues
for idx, row in game_results[game_results["went_ot"] == 1].iterrows():
    ot_duration_clipped = min(row["ot_duration"], MAX_OT_SECONDS)
    ot_win_w  = 0.90 * (1 - (ot_duration_clipped / MAX_OT_SECONDS) * 0.35)
    ot_loss_w = 1 - ot_win_w

    if row["home_win"] == 1:
        game_results.at[idx, "home_weighted_result"] = ot_win_w
        game_results.at[idx, "away_weighted_result"] = ot_loss_w
    else:
        game_results.at[idx, "away_weighted_result"] = ot_win_w
        game_results.at[idx, "home_weighted_result"] = ot_loss_w

# -------------------------
# TEAM WEIGHTED WINS
# Sum each team's weighted results across all 82 games.
# Reshape to team-game format by splitting home and away rows,
# stack them, then sum per team.
# No recency weighting — games are not sorted chronologically.
# -------------------------
home_games = game_results[["game_id", "home_team", "home_weighted_result"]].rename(
    columns={"home_team": "team", "home_weighted_result": "weighted_result"}
)
away_games = game_results[["game_id", "away_team", "away_weighted_result"]].rename(
    columns={"away_team": "team", "away_weighted_result": "weighted_result"}
)

team_games = pd.concat([home_games, away_games])
team_weighted_wins = team_games.groupby("team")["weighted_result"].sum()

# -------------------------
# BASIC STANDINGS
# Raw win counts and goal totals for each team.
# -------------------------
home_wins     = game_results.groupby("home_team")["home_win"].sum()
away_wins     = game_results.groupby("away_team")["away_win"].sum()
team_wins     = home_wins.add(away_wins, fill_value=0)

home_GF       = game_results.groupby("home_team")["home_goals"].sum()
away_GF       = game_results.groupby("away_team")["away_goals"].sum()
goals_for     = home_GF.add(away_GF, fill_value=0)

home_GA       = game_results.groupby("home_team")["away_goals"].sum()
away_GA       = game_results.groupby("away_team")["home_goals"].sum()
goals_against = home_GA.add(away_GA, fill_value=0)

standings = pd.DataFrame({
    "Wins":          team_wins,
    "Goals_For":     goals_for,
    "Goals_Against": goals_against,
    "Weighted_Wins": team_weighted_wins
})

standings["Games_Played"] = 82
standings["Goal_Diff"]    = standings["Goals_For"] - standings["Goals_Against"]
standings["Win_Pct"]      = standings["Weighted_Wins"] / 82

# -------------------------
# TEAM PROCESS METRICS
# All counting stats converted to per-60-minute rates.
# TOI is summed from both home and away sides then divided by 2
# to correct for double-counting — each segment's TOI is shared
# by both teams so a naive sum counts every second twice.
# -------------------------
team_stats = pd.DataFrame(index=standings.index)

home_xg = df.groupby("home_team")["home_xg"].sum()
away_xg = df.groupby("away_team")["away_xg"].sum()

team_stats["xG_For"]     = home_xg.add(away_xg, fill_value=0)
team_stats["xG_Against"] = (
    df.groupby("home_team")["away_xg"].sum()
    .add(df.groupby("away_team")["home_xg"].sum(), fill_value=0)
)
team_stats["xG_Diff"] = team_stats["xG_For"] - team_stats["xG_Against"]

team_stats["Shots_For"] = (
    df.groupby("home_team")["home_shots"].sum()
    .add(df.groupby("away_team")["away_shots"].sum(), fill_value=0)
)
team_stats["Shots_Against"] = (
    df.groupby("home_team")["away_shots"].sum()
    .add(df.groupby("away_team")["home_shots"].sum(), fill_value=0)
)
team_stats["Shot_Diff"] = team_stats["Shots_For"] - team_stats["Shots_Against"]

# TOI: sum both sides then divide by 2 to correct double-counting
team_stats["TOI"] = (
    df.groupby("home_team")["toi"].sum()
    .add(df.groupby("away_team")["toi"].sum(), fill_value=0)
) / 2

team_stats["Goal_Diff"]     = standings["Goal_Diff"]
team_stats["Goals_Against"] = standings["Goals_Against"]

# Per-60 rates
team_stats["xG_Diff_per60"]    = team_stats["xG_Diff"]    / team_stats["TOI"] * 3600
team_stats["xG_Against_per60"] = team_stats["xG_Against"] / team_stats["TOI"] * 3600
team_stats["Goal_Diff_per60"]  = team_stats["Goal_Diff"]  / team_stats["TOI"] * 3600
team_stats["Shot_Diff_per60"]  = team_stats["Shot_Diff"]  / team_stats["TOI"] * 3600

# -------------------------
# GOALTENDING
# Save percentage derived from shots and goals since saves aren't stored.
# League average uses a weighted mean so high-volume teams contribute
# proportionally. Relative save percentage = team minus league average.
# -------------------------
team_stats["Save_Pct"] = (
    team_stats["Shots_Against"] - team_stats["Goals_Against"]
) / team_stats["Shots_Against"]

league_sv = (
    (team_stats["Shots_Against"] - team_stats["Goals_Against"]).sum()
    / team_stats["Shots_Against"].sum()
)

team_stats["Rel_Save_Pct"] = team_stats["Save_Pct"] - league_sv

# -------------------------
# SPECIAL TEAMS METRICS
# PP xG per 60 measures how much offensive threat a team generates
# when they have the man advantage. PK xG against per 60 measures
# how well they suppress the opponent when killing a penalty.
# We identify PP segments by filtering where the team's offensive
# line is labeled PP_up. PK segments are where the team's offensive
# line is labeled PP_kill_dwn.
# -------------------------

# --- POWER PLAY ---
# Home team on PP: home_off_line == PP_up
# Away team on PP: away_off_line == PP_up
home_pp = df[df["home_off_line"] == "PP_up"][["home_team", "home_xg", "toi"]].rename(
    columns={"home_team": "team", "home_xg": "xg", "toi": "pp_toi"}
)
away_pp = df[df["away_off_line"] == "PP_up"][["away_team", "away_xg", "toi"]].rename(
    columns={"away_team": "team", "away_xg": "xg", "toi": "pp_toi"}
)

pp_combined = pd.concat([home_pp, away_pp])
pp_stats = pp_combined.groupby("team").agg({"xg": "sum", "pp_toi": "sum"})
pp_stats["pp_xg_per60"] = (pp_stats["xg"] / pp_stats["pp_toi"]) * 3600

# --- PENALTY KILL ---
# When home team is on PP, away team is killing — away_off_line == PP_kill_dwn
# xG allowed on PK = the PP team's xG generated against them
# So PK xG against for the home team = away xG when away is on PP
# And PK xG against for the away team = home xG when home is on PP
home_pk = df[df["away_off_line"] == "PP_up"][["home_team", "away_xg", "toi"]].rename(
    columns={"home_team": "team", "away_xg": "xg_against", "toi": "pk_toi"}
)
away_pk = df[df["home_off_line"] == "PP_up"][["away_team", "home_xg", "toi"]].rename(
    columns={"away_team": "team", "home_xg": "xg_against", "toi": "pk_toi"}
)

pk_combined = pd.concat([home_pk, away_pk])
pk_stats = pk_combined.groupby("team").agg({"xg_against": "sum", "pk_toi": "sum"})
pk_stats["pk_xg_against_per60"] = (pk_stats["xg_against"] / pk_stats["pk_toi"]) * 3600

# --- MERGE INTO TEAM STATS ---
team_stats["pp_xg_per60"]       = pp_stats["pp_xg_per60"]
team_stats["pk_xg_against_per60"] = pk_stats["pk_xg_against_per60"]

# -------------------------
# UPDATED Z-SCORES
# PP z-score: higher is better (more xG generated on PP)
# PK z-score: lower xG against is better, so we negate it
# so that a positive z-score always means better performance
# -------------------------

def zscore(series):
    return (series - series.mean()) / series.std()

team_stats["Win_z"]              = zscore(standings["Win_Pct"])
team_stats["xG_z"]               = zscore(team_stats["xG_Diff_per60"])
team_stats["Goal_z"]             = zscore(team_stats["Goal_Diff_per60"])
team_stats["Shot_z"]             = zscore(team_stats["Shot_Diff_per60"])
team_stats["Goalie_z"]           = zscore(team_stats["Rel_Save_Pct"])
team_stats["xG_Against_per60_z"] = zscore(team_stats["xG_Against_per60"])
team_stats["pp_z"]               = zscore(team_stats["pp_xg_per60"])
team_stats["pk_z"]               = -zscore(team_stats["pk_xg_against_per60"])

# -------------------------
# SPECIAL TEAMS SCORE
# Combines PP and PK into a single metric.
# Weighted equally since both sides of special teams matter.
# -------------------------
team_stats["Special_Teams_Score"] = (
    0.50 * team_stats["pp_z"]
    + 0.50 * team_stats["pk_z"]
)

# -------------------------
# UPDATED POWER RATING
# Results       15%
# Process       40% (down from 50% to make room for special teams)
# Goalie        30% (down from 35%)
# Special Teams 15% (new)
# -------------------------
team_stats["Results_Score"] = team_stats["Win_z"]

team_stats["Process_Score"] = (
    0.70 * team_stats["xG_z"]
    + 0.30 * team_stats["Shot_z"]
)

team_stats["Goalie_Score"] = (
    0.70 * team_stats["Goalie_z"]
    - 0.30 * team_stats["xG_Against_per60_z"]
)

team_stats["Power_Rating"] = (
    0.15 * team_stats["Results_Score"]
    + 0.40 * team_stats["Process_Score"]
    + 0.30 * team_stats["Goalie_Score"]
    + 0.15 * team_stats["Special_Teams_Score"]
)

power_rankings = (
    team_stats["Power_Rating"]
    .sort_values(ascending=False)
    .reset_index()
)
power_rankings.columns = ["Team", "Power_Rating"]
power_rankings["Rank"] = power_rankings.index + 1

print(power_rankings)

# -------------------------
# UPDATED PROCESS RATING (for logistic regression — no wins)
# -------------------------
team_stats["Process_Rating"] = (
    0.47 * team_stats["Process_Score"]
    + 0.35 * team_stats["Goalie_Score"]
    + 0.18 * team_stats["Special_Teams_Score"]
)

process_dict = dict(zip(team_stats.index, team_stats["Process_Rating"]))

# -------------------------
# LOGISTIC REGRESSION CALIBRATION
# Predicts home win probability from process rating differential.
# Intercept captures home ice advantage. Scale coefficient converts
# rating differences into probabilities.
# -------------------------
game_results["home_process"] = game_results["home_team"].map(process_dict)
game_results["away_process"] = game_results["away_team"].map(process_dict)
game_results["process_diff"] = (
    game_results["home_process"] - game_results["away_process"]
)

y = game_results["home_win"]
X = sm.add_constant(game_results["process_diff"])

logit_model = sm.Logit(y, X).fit()
print(logit_model.summary())

intercept = logit_model.params["const"]
print(f"Intercept:{intercept}")
scale_est = logit_model.params["process_diff"]

# -------------------------
# WIN PROBABILITY FUNCTION
# Takes two team abbreviations, looks up their process ratings,
# and returns the win probability for team_a (treated as home team).
# Raises a helpful error if an abbreviation is not recognized.
# -------------------------
def win_probability(team_a, team_b):
    if team_a not in process_dict:
        raise ValueError(f"Team '{team_a}' not found. Check abbreviation.")
    if team_b not in process_dict:
        raise ValueError(f"Team '{team_b}' not found. Check abbreviation.")
    process_a = process_dict[team_a]
    process_b = process_dict[team_b]
    return 1 / (1 + np.exp(-(intercept + scale_est * (process_a - process_b))))

# -------------------------
# USER PREDICTION
# -------------------------
team_a = input("Enter Team A abbreviation (home): ")
team_b = input("Enter Team B abbreviation (away): ")

prob_a = win_probability(team_a, team_b)
print(f"{team_a} win probability vs {team_b}: {prob_a:.2%}")

# -------------------------
# DEFENSIVE BASELINES
# We want to know how much xG each defensive pairing (1st, 2nd, 3rd)
# allows per 60 minutes across the league. This becomes a multiplier
# that adjusts offensive xG upward if a line generated chances against
# a tough defensive pairing, and downward against a weak one.
#
# Each row represents a matchup segment. The home team's xG was generated
# against the away team's defensive pairing, and vice versa.
# We divide TOI by 2 here because the same segment TOI appears from both
# the home and away perspective when we concat — unlike the offensive line
# analysis where each team's lines are tracked independently, here we're
# pooling all defensive pairings together so each segment would be
# counted twice without the correction.
# -------------------------
def_home = df[['away_def_pairing', 'home_xg', 'toi']].copy()
def_home = def_home.rename(columns={'away_def_pairing': 'def_rank', 'home_xg': 'xg_allowed'})
def_home['toi'] = def_home['toi'] / 2

def_away = df[['home_def_pairing', 'away_xg', 'toi']].copy()
def_away = def_away.rename(columns={'home_def_pairing': 'def_rank', 'away_xg': 'xg_allowed'})
def_away['toi'] = def_away['toi'] / 2

def_combined = pd.concat([def_home, def_away])

# xG allowed per 60 for each defensive pairing rank
def_stats = def_combined.groupby('def_rank').agg({'xg_allowed': 'sum', 'toi': 'sum'})
def_stats['xg_per60_allowed'] = (def_stats['xg_allowed'] / def_stats['toi']) * 3600

# League average xG allowed across all pairings
league_avg_xg_allowed = (def_combined['xg_allowed'].sum() / def_combined['toi'].sum()) * 3600

# Multiplier = league average / pairing specific rate
# A tough pairing (low xG allowed) produces a multiplier > 1,
# crediting the offense more for generating against them.
# A weak pairing (high xG allowed) produces a multiplier < 1.
def_multiplier = (league_avg_xg_allowed / def_stats['xg_per60_allowed']).to_dict()

print("Defensive multipliers by pairing:")
print(def_multiplier)

# -------------------------
# APPLY DEFENSIVE MULTIPLIERS TO OFFENSIVE XG
# Instead of row-by-row apply(), we use vectorized map() which is
# faster and cleaner. We map each row's opposing defensive pairing
# to its multiplier, then multiply against the team's xG.
# fillna(1.0) means any unrecognized pairing label gets no adjustment.
# -------------------------
df['home_xg_adj'] = df['home_xg'] * df['away_def_pairing'].map(def_multiplier).fillna(1.0)
df['away_xg_adj'] = df['away_xg'] * df['home_def_pairing'].map(def_multiplier).fillna(1.0)

# -------------------------
# REORGANIZE FOR LINE-LEVEL ANALYSIS
# Reshape into a single team-line format so we can compare each
# team's first and second offensive lines directly.
# Each row tracks one team's offensive line, the TOI they were on ice,
# and their defensive-adjusted xG for that segment.
# -------------------------
home_df = pd.DataFrame({
    "team":   df["home_team"],
    "line":   df["home_off_line"],
    "toi":    df["toi"],
    "xg_adj": df["home_xg_adj"]
})

away_df = pd.DataFrame({
    "team":   df["away_team"],
    "line":   df["away_off_line"],
    "toi":    df["toi"],
    "xg_adj": df["away_xg_adj"]
})

team_df = pd.concat([home_df, away_df], ignore_index=True)

# -------------------------
# DISPARITY RATIO
# Filter to only first and second offensive lines — we're measuring
# how much better a team's top line is relative to their second line.
# Higher ratio = more top-heavy. Lower ratio (closer to 1.0) = more balanced.
#
# TOI filter (300 minutes minimum) applied before ratio calculation
# to exclude line combinations with too small a sample to be reliable.
#
# Per-60 rate uses seconds consistently with the rest of the codebase:
# xg_adj / toi * 3600
# -------------------------
team_df = team_df[team_df["line"].isin(["first_off", "second_off"])]

line_stats = (
    team_df.groupby(["team", "line"])
           .agg({"xg_adj": "sum", "toi": "sum"})
           .reset_index()
)

line_stats["xg_adj_per60"] = (line_stats["xg_adj"] / line_stats["toi"]) * 3600

# Apply TOI filter — 300 minutes minimum
MIN_TOI_SECONDS = 300 * 60
line_stats = line_stats[line_stats["toi"] >= MIN_TOI_SECONDS]

# Warn if any teams get dropped due to the TOI filter
teams_in_data   = set(team_df["team"].unique())
teams_surviving = set(line_stats["team"].unique())
dropped = teams_in_data - teams_surviving
if dropped:
    print(f"Warning: these teams were excluded from disparity rankings due to TOI filter: {dropped}")

# Calculate disparity ratio for each team
# Returns first_off xg_adj_per60 / second_off xg_adj_per60
# Returns None if a team is missing either line (filtered out above)
def calculate_disparity(group):
    stats = group.set_index('line')['xg_adj_per60']
    if 'first_off' in stats and 'second_off' in stats:
        return stats['first_off'] / stats['second_off']
    return None

team_disparity = (
    line_stats.groupby("team")
              .apply(calculate_disparity)
              .reset_index(name="disparity_ratio")
)

team_disparity = (
    team_disparity.dropna()
                  .sort_values(by="disparity_ratio", ascending=False)
                  .reset_index(drop=True)
)

team_disparity["rank"] = team_disparity.index + 1

print(team_disparity)

# --- 1. MERGE POWER RATINGS WITH DISPARITY DATA ---
final_analysis = pd.merge(
    team_disparity,
    power_rankings[['Team', 'Power_Rating']],
    left_on="team", right_on="Team"
).drop(columns=['Team'])

# --- 2. FORMAT TEAM LABELS ---
# Handle special cases: all-caps abbreviations, underscore replacement, title case
ALL_CAPS = {'usa', 'uae', 'uk'}

def format_team_name(name):
    if name in ALL_CAPS:
        return name.upper()
    return name.replace('_', ' ').title()

final_analysis['team_label'] = final_analysis['team'].apply(format_team_name)

# --- 3. SETUP FIGURE ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(15, 10))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

# --- 4. QUADRANT DIVIDERS ---
x_mean = final_analysis['disparity_ratio'].mean()
y_mean = final_analysis['Power_Rating'].mean()

ax.axvline(x_mean, color='#AAAAAA', linestyle='--', linewidth=1.0, alpha=0.8, zorder=1)
ax.axhline(y_mean, color='#AAAAAA', linestyle='--', linewidth=1.0, alpha=0.8, zorder=1)

# --- 5. QUADRANT SHADING ---
x_min_data = final_analysis['disparity_ratio'].min()
x_max_data = final_analysis['disparity_ratio'].max()
y_min_data = final_analysis['Power_Rating'].min()
y_max_data = final_analysis['Power_Rating'].max()

ax.axvspan(x_min_data - 0.1, x_mean, ymin=0.5, ymax=1.0,
           alpha=0.04, color='#2ecc71', zorder=0)
ax.axvspan(x_mean, x_max_data + 0.1, ymin=0.5, ymax=1.0,
           alpha=0.04, color='#e67e22', zorder=0)
ax.axvspan(x_min_data - 0.1, x_mean, ymin=0.0, ymax=0.5,
           alpha=0.04, color='#3498db', zorder=0)
ax.axvspan(x_mean, x_max_data + 0.1, ymin=0.0, ymax=0.5,
           alpha=0.04, color='#e74c3c', zorder=0)

# --- 6. COLOR-CODE POINTS BY QUADRANT ---
def assign_quadrant(row):
    if row['disparity_ratio'] <= x_mean and row['Power_Rating'] >= y_mean:
        return 'Elite & Deep'
    elif row['disparity_ratio'] > x_mean and row['Power_Rating'] >= y_mean:
        return 'Top-Heavy Contenders'
    elif row['disparity_ratio'] <= x_mean and row['Power_Rating'] < y_mean:
        return 'Balanced but Struggling'
    else:
        return 'Top-Heavy & Struggling'

final_analysis['quadrant'] = final_analysis.apply(assign_quadrant, axis=1)

quadrant_colors = {
    'Elite & Deep':            '#27ae60',
    'Top-Heavy Contenders':    '#d35400',
    'Balanced but Struggling': '#2980b9',
    'Top-Heavy & Struggling':  '#c0392b',
}

for quadrant, group in final_analysis.groupby('quadrant'):
    ax.scatter(
        group['disparity_ratio'],
        group['Power_Rating'],
        color=quadrant_colors[quadrant],
        s=180,
        alpha=0.90,
        edgecolors='white',
        linewidth=1.2,
        label=quadrant,
        zorder=4
    )

# --- 7. REGRESSION LINE ---
sns.regplot(
    data=final_analysis,
    x='disparity_ratio',
    y='Power_Rating',
    scatter=False,
    ax=ax,
    line_kws={'color': '#2c3e50', 'linestyle': '-', 'linewidth': 2.0},
    ci=95
)

# --- 8. LABEL EVERY TEAM ---
for _, row in final_analysis.iterrows():
    ax.annotate(
        row['team_label'],
        xy=(row['disparity_ratio'], row['Power_Rating']),
        xytext=(7, 5),
        textcoords='offset points',
        fontsize=8.5,
        fontweight='bold',
        color='#2c3e50',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.5, edgecolor='none')
    )

# --- 9. QUADRANT LABELS ---
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()
x_pad = (x_max - x_min) * 0.02
y_pad = (y_max - y_min) * 0.03

quadrant_labels = [
    (x_min + x_pad, y_max - y_pad, 'Elite & Deep',            '#27ae60', 'left',  'top'),
    (x_max - x_pad, y_max - y_pad, 'Top-Heavy Contenders',    '#d35400', 'right', 'top'),
    (x_min + x_pad, y_min + y_pad, 'Balanced but Struggling', '#2980b9', 'left',  'bottom'),
    (x_max - x_pad, y_min + y_pad, 'Top-Heavy & Struggling',  '#c0392b', 'right', 'bottom'),
]

for x, y, label, color, ha, va in quadrant_labels:
    ax.text(x, y, label,
            fontsize=12, fontweight='bold',
            color=color, ha=ha, va=va,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      alpha=0.75, edgecolor=color, linewidth=1.2))

# --- 10. TITLES AND AXES ---
ax.set_title(
    "WHL Offensive Line Disparity vs. Team Success",
    fontsize=20, fontweight='bold', pad=24, color='#2c3e50'
)
ax.set_xlabel(
    "Line Quality Disparity Ratio (1st Line xG / 2nd Line xG)\n← Balanced / Second Line Dominant                    Top Heavy →",
    fontsize=12, labelpad=12, color='#2c3e50'
)
ax.set_ylabel(
    "Standardized Team Power Rating",
    fontsize=12, labelpad=12, color='#2c3e50'
)

ax.tick_params(colors='#555555', labelsize=10)
for spine in ax.spines.values():
    spine.set_edgecolor('#DDDDDD')

# --- 11. CAPTION ---
plt.figtext(
    0.5, -0.03,
    "Finding: Offensive line balance alone is not a strong predictor of team success in the WHL.",
    ha="center", fontsize=11, color='#555555',
    bbox={"facecolor": "#FFF3CD", "alpha": 0.8, "pad": 6, "edgecolor": "#DDDDDD"}
)

# --- 12. SAVE ---
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('DataScienceTeam.png', dpi=300, bbox_inches='tight')
plt.show()

from sklearn.metrics import (
    log_loss, brier_score_loss, roc_auc_score, roc_curve
)
# -------------------------
# GENERATE PREDICTED PROBABILITIES
# -------------------------
game_results["predicted_prob"] = game_results["process_diff"].apply(
    lambda x: 1 / (1 + np.exp(-(intercept + scale_est * x)))
)

# -------------------------
# CORE METRICS
# -------------------------
actual    = game_results["home_win"]
predicted = game_results["predicted_prob"]

ll    = log_loss(actual, predicted)
bs    = brier_score_loss(actual, predicted)
auc   = roc_auc_score(actual, predicted)

print(f"Log-Loss:     {ll:.4f}  (lower is better, 0.693 = random guessing)")
print(f"Brier Score:  {bs:.4f}  (lower is better, 0.25 = random guessing)")
print(f"AUC-ROC:      {auc:.4f}  (higher is better, 0.5 = random guessing)")

# -------------------------
# CALIBRATION PLOT
# Buckets predicted probabilities into bins and compares
# average predicted probability vs actual win rate per bin.
# A perfectly calibrated model follows the diagonal line.
# -------------------------
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(
    actual, predicted, n_bins=10, strategy='uniform'
)

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(prob_pred, prob_true,
        marker='o', linewidth=2, color='#2c3e50', label='Model')
ax.plot([0, 1], [0, 1],
        linestyle='--', color='gray', label='Perfect Calibration')

ax.set_xlabel('Mean Predicted Probability', fontsize=12)
ax.set_ylabel('Observed Win Rate', fontsize=12)
ax.set_title('Logistic Regression Calibration Plot', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig('calibration_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------
# ROC CURVE
# Plots true positive rate vs false positive rate at all thresholds.
# Area under the curve (AUC) summarizes overall discriminative ability.
# -------------------------
fpr, tpr, _ = roc_curve(actual, predicted)

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(fpr, tpr,
        linewidth=2, color='#2c3e50', label=f'Model (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1],
        linestyle='--', color='gray', label='Random Guessing')

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------
# CALIBRATION BY PROBABILITY BUCKET
# Prints a table showing predicted vs actual win rates
# across five probability ranges so you can spot
# where the model over or underestimates.
# -------------------------
game_results["prob_bucket"] = pd.cut(
    game_results["predicted_prob"],
    bins=[0, 0.4, 0.45, 0.55, 0.60, 1.0],
    labels=["<40%", "40-45%", "45-55%", "55-60%", ">60%"]
)

calibration_table = (
    game_results.groupby("prob_bucket", observed=True)
    .agg(
        Games=("home_win", "count"),
        Predicted_Win_Pct=("predicted_prob", "mean"),
        Actual_Win_Pct=("home_win", "mean")
    )
    .round(3)
)

print(calibration_table)

# -------------------------
# VISUALIZATION SLIDE 2: PDO distribution chart
# -------------------------

# -------------------------
# PDO DISTRIBUTION CHART
# PDO = shooting% + save%
# Baseline of 1.000 = league average, no luck component
# Teams above 1.000 are likely overperforming, below are underperforming
# -------------------------

team_stats["Shooting_Pct"] = standings["Goals_For"] / team_stats["Shots_For"]
team_stats["PDO"] = team_stats["Shooting_Pct"] + team_stats["Save_Pct"]

pdo_df = team_stats[["PDO"]].copy().reset_index()
pdo_df.columns = ["team", "PDO"]
pdo_df["team_label"] = pdo_df["team"].apply(format_team_name)
pdo_df = pdo_df.sort_values("PDO", ascending=True).reset_index(drop=True)

# Threshold for labeling — teams more than 1 std dev from mean
pdo_mean = pdo_df["PDO"].mean()
pdo_std  = pdo_df["PDO"].std()
LABEL_THRESHOLD = 1.0 * pdo_std

# -------------------------
# PLOT
# -------------------------
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(11, 9))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

# Color code by deviation from 1.000
colors = ['#c0392b' if x < 1.000 else '#27ae60' for x in pdo_df["PDO"]]

# Draw horizontal lines from 1.000 to each point
for i, (_, row) in enumerate(pdo_df.iterrows()):
    ax.plot([1.000, row["PDO"]], [i, i],
            color='#DDDDDD', linewidth=1.0, zorder=1)

# Draw dots
ax.scatter(
    pdo_df["PDO"],
    range(len(pdo_df)),
    color=colors,
    s=120,
    zorder=3,
    edgecolors='white',
    linewidth=0.8
)

# Baseline at 1.000
ax.axvline(1.000, color='#2c3e50', linewidth=1.5, linestyle='--', zorder=2)
ax.text(1.000, len(pdo_df) - 0.3, "PDO = 1.000\n(League Average)",
        ha='center', va='top', fontsize=9, color='#2c3e50', fontweight='bold')

# Label only significant outliers
for i, row in pdo_df.iterrows():
    if abs(row["PDO"] - 1.000) >= LABEL_THRESHOLD:
        ha = 'left' if row["PDO"] > 1.000 else 'right'
        offset = 0.0008 if row["PDO"] > 1.000 else -0.0008
        ax.text(
            row["PDO"] + offset, i,
            row["team_label"],
            va='center', ha=ha,
            fontsize=9, fontweight='bold',
            color='#2c3e50'
        )

# Shading
x_min, x_max = ax.get_xlim()
ax.axvspan(x_min, 1.000, alpha=0.03, color='#c0392b', zorder=0)
ax.axvspan(1.000, x_max, alpha=0.03, color='#27ae60', zorder=0)

# Annotations for lucky / unlucky zones
ax.text(x_min + 0.001, len(pdo_df) * 0.95,
        "← Underperforming",
        fontsize=9, color='#c0392b', fontweight='bold', va='top')
ax.text(x_max - 0.001, len(pdo_df) * 0.95,
        "Overperforming",
        fontsize=9, color='#27ae60', fontweight='bold', va='top', ha='right')

# Remove y-axis ticks since teams are unlabeled except outliers
ax.set_yticks([])

# Titles and axes
ax.set_title(
    "PDO Distribution Across WHL Teams",
    fontsize=18, fontweight='bold', pad=20, color='#2c3e50'
)
ax.set_xlabel(
    "PDO (Shooting % + Save %)",
    fontsize=12, labelpad=10, color='#2c3e50', fontweight='bold'
)

ax.tick_params(axis='x', labelsize=10, colors='#555555')
for spine in ax.spines.values():
    spine.set_edgecolor('#DDDDDD')

plt.figtext(
    0.5, -0.02,
    "PDO regresses to 1.000 over time — teams significantly above or below are likely benefiting from or suffering from variance.",
    ha="center", fontsize=9.5, color='#555555',
    bbox={"facecolor": "#FFF3CD", "alpha": 0.8, "pad": 5, "edgecolor": "#DDDDDD"}
)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig('PDO_Distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------
# VISUALIZATION SLIDE 3: LINE DISPARITY BARGRAPH
# -------------------------

# -------------------------
# FIRST VS SECOND LINE XG PER 60
# Compare most top-heavy vs most balanced team
# -------------------------

# Get most top-heavy and most balanced teams from disparity data
most_topheavy  = team_disparity.iloc[0]["team"]   # highest disparity ratio
most_balanced  = team_disparity.iloc[-1]["team"]  # lowest disparity ratio

# Pull their line stats
featured_teams = [most_topheavy, most_balanced]
featured_stats = line_stats[line_stats["team"].isin(featured_teams)].copy()
featured_stats["team_label"] = featured_stats["team"].apply(format_team_name)

# -------------------------
# PLOT
# -------------------------
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

bar_width = 0.30
x = np.arange(2)  # two teams

# Extract values in correct order
teams_ordered  = [most_topheavy, most_balanced]
labels_ordered = [format_team_name(t) for t in teams_ordered]

first_line_vals  = [
    featured_stats[(featured_stats["team"] == t) & (featured_stats["line"] == "first_off")]["xg_adj_per60"].values[0]
    for t in teams_ordered
]
second_line_vals = [
    featured_stats[(featured_stats["team"] == t) & (featured_stats["line"] == "second_off")]["xg_adj_per60"].values[0]
    for t in teams_ordered
]

bars1 = ax.bar(
    x - bar_width / 2, first_line_vals,
    width=bar_width,
    color='#2980b9',
    edgecolor='white',
    linewidth=0.8,
    alpha=0.90,
    label='1st Line'
)
bars2 = ax.bar(
    x + bar_width / 2, second_line_vals,
    width=bar_width,
    color='#e67e22',
    edgecolor='white',
    linewidth=0.8,
    alpha=0.90,
    label='2nd Line'
)

# Value labels on bars
for bar in bars1:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{bar.get_height():.2f}",
        ha='center', va='bottom',
        fontsize=11, fontweight='bold', color='#2c3e50'
    )

for bar in bars2:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{bar.get_height():.2f}",
        ha='center', va='bottom',
        fontsize=11, fontweight='bold', color='#2c3e50'
    )

# Disparity ratio annotations below team names
for i, team in enumerate(teams_ordered):
    ratio = team_disparity[team_disparity["team"] == team]["disparity_ratio"].values[0]
    ax.text(
        x[i], -0.18,
        f"Disparity Ratio: {ratio:.2f}",
        ha='center', va='top',
        fontsize=9.5, color='#555555', fontstyle='italic',
        transform=ax.get_xaxis_transform()
    )

# X axis team labels
ax.set_xticks(x)
ax.set_xticklabels(labels_ordered, fontsize=13, fontweight='bold', color='#2c3e50')

# Subtitles under team names
ax.text(x[0], ax.get_ylim()[0] - 0.1, "Most Top-Heavy",
        ha='center', fontsize=9.5, color='#2980b9',
        fontweight='bold', transform=ax.get_xaxis_transform())
ax.text(x[1], ax.get_ylim()[0] - 0.1, "Most Balanced",
        ha='center', fontsize=9.5, color='#e67e22',
        fontweight='bold', transform=ax.get_xaxis_transform())

# Titles and axes
ax.set_title(
    "First vs. Second Line xG per 60",
    fontsize=17, fontweight='bold', pad=18, color='#2c3e50'
)
ax.set_ylabel(
    "Defensive-Adjusted xG per 60",
    fontsize=11, labelpad=10, color='#2c3e50', fontweight='bold'
)

ax.legend(fontsize=11, framealpha=0.8)
ax.tick_params(axis='y', labelsize=10, colors='#555555')

for spine in ax.spines.values():
    spine.set_edgecolor('#DDDDDD')

plt.tight_layout()
plt.savefig('LineDisparity_Comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Check total TOI for a known OT game
print(game_results[game_results["went_ot"] == 1]["ot_duration"].describe())

## -------------------------
# OT DECAY CURVE — SIMPLIFIED
# -------------------------
MAX_OT_SECONDS = 2400
OT_WIN_MAX     = 0.90
OT_WIN_MIN     = 0.55

# Generate smooth curve in minutes for readability
ot_minutes   = np.linspace(0, 60, 500)
ot_seconds_c = ot_minutes * 60
win_weights  = np.where(
    ot_seconds_c <= MAX_OT_SECONDS,
    OT_WIN_MAX - (ot_seconds_c / MAX_OT_SECONDS) * (OT_WIN_MAX - OT_WIN_MIN),
    OT_WIN_MIN
)
loss_weights = 1 - win_weights

# Actual OT game durations in minutes
actual_ot_minutes = game_results[game_results["went_ot"] == 1]["ot_duration"].dropna() / 60
avg_ot_minutes    = actual_ot_minutes.mean()
avg_win_weight    = OT_WIN_MAX - (min(avg_ot_minutes * 60, MAX_OT_SECONDS) / MAX_OT_SECONDS) * (OT_WIN_MAX - OT_WIN_MIN)

# -------------------------
# PLOT
# -------------------------
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

# Curves
ax.plot(ot_minutes, win_weights,
        color='#27ae60', linewidth=2.5, label='OT Win Weight', zorder=3)
ax.plot(ot_minutes, loss_weights,
        color='#c0392b', linewidth=2.5, label='OT Loss Weight', zorder=3)

# Regulation win reference
ax.axhline(1.0, color='#2c3e50', linewidth=1.0,
           linestyle=':', alpha=0.5, zorder=2)
ax.text(58, 1.015, "Regulation Win = 1.0",
        ha='right', fontsize=9, color='#2c3e50', fontstyle='italic')

# Average OT duration line
ax.axvline(avg_ot_minutes, color='#2980b9',
           linewidth=1.5, linestyle='--', zorder=2,
           label=f'Avg OT Duration ({avg_ot_minutes:.1f} min)')

# Dot at average on win curve
ax.scatter([avg_ot_minutes], [avg_win_weight],
           color='#2980b9', s=100, zorder=5,
           edgecolors='white', linewidth=1.2)

# Annotation for average weight at that point
ax.annotate(
    f"Avg OT win weight: {avg_win_weight:.2f}",
    xy=(avg_ot_minutes, avg_win_weight),
    xytext=(avg_ot_minutes + 3, avg_win_weight + 0.05),
    fontsize=9.5, color='#2980b9', fontweight='bold',
    arrowprops=dict(arrowstyle='->', color='#2980b9', lw=1.0)
)

# Start annotations
ax.text(1, OT_WIN_MAX + 0.02,
        f"Win = {OT_WIN_MAX:.2f}",
        fontsize=9, color='#27ae60', fontweight='bold')
ax.text(1, 1 - OT_WIN_MAX - 0.04,
        f"Loss = {1-OT_WIN_MAX:.2f}",
        fontsize=9, color='#c0392b', fontweight='bold')

# End annotations at 40 min cap
ax.text(41, OT_WIN_MIN + 0.01,
        f"Win = {OT_WIN_MIN:.2f}",
        fontsize=9, color='#27ae60', fontweight='bold')
ax.text(41, 1 - OT_WIN_MIN - 0.04,
        f"Loss = {1-OT_WIN_MIN:.2f}",
        fontsize=9, color='#c0392b', fontweight='bold')

# Cap line at 40 minutes
ax.axvline(40, color='#AAAAAA', linewidth=1.0,
           linestyle=':', alpha=0.7, zorder=2)
ax.text(40.5, 0.08, "40 min cap",
        fontsize=8.5, color='#AAAAAA', fontstyle='italic')

# Titles and axes
ax.set_title(
    "OT Result Weight by Game Duration",
    fontsize=17, fontweight='bold', pad=18, color='#2c3e50'
)
ax.set_xlabel(
    "OT Duration (minutes)",
    fontsize=12, labelpad=10, color='#2c3e50', fontweight='bold'
)
ax.set_ylabel(
    "Result Weight",
    fontsize=12, labelpad=10, color='#2c3e50', fontweight='bold'
)

ax.set_xlim(0, 62)
ax.set_ylim(0, 1.08)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.tick_params(axis='both', labelsize=10, colors='#555555')

ax.legend(fontsize=10, framealpha=0.85, loc='upper right',
          bbox_to_anchor=(1.0, 0.95))

for spine in ax.spines.values():
    spine.set_edgecolor('#DDDDDD')

plt.figtext(
    0.5, -0.02,
    "As OT duration increases, win and loss weights converge — the average WHL OT game carries significantly less information than a quick overtime finish.",
    ha="center", fontsize=9.5, color='#555555',
    bbox={"facecolor": "#FFF3CD", "alpha": 0.8, "pad": 5, "edgecolor": "#DDDDDD"}
)
plt.legend(frameon=True)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig('OT_Decay_Curve.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------
# POWER RANKINGS TABLE VISUALIZATION
# Shows ranks 3-5 and 27-29 with highlighted rows for #4 and #28
# Used to set up the matchup probability demonstration
# -------------------------

# Get the teams and their data
def get_ranking_row(rank):
    row = power_rankings[power_rankings["Rank"] == rank].iloc[0]
    return {
        "Rank": rank,
        "Team": format_team_name(row["Team"]),
        "Power Rating": round(row["Power_Rating"], 2)
    }

rows_to_show = [3, 4, 5, None, 27, 28, 29]  # None = ellipsis
table_data   = [get_ranking_row(r) if r else None for r in rows_to_show]

# -------------------------
# PLOT
# -------------------------
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')
ax.axis('off')

# Column headers
col_labels  = ["Rank", "Team", "Power Rating"]
col_widths  = [0.15, 0.55, 0.30]
col_x       = [0.08, 0.23, 0.78]
header_y    = 0.92
row_height  = 0.10

# Draw header
for i, (label, x) in enumerate(zip(col_labels, col_x)):
    ax.text(x, header_y, label,
            fontsize=12, fontweight='bold', color='#2c3e50',
            ha='left' if i > 0 else 'center',
            va='center', transform=ax.transAxes)

# Header underline
ax.plot([0.05, 0.95], [header_y - 0.04, header_y - 0.04],
        color='#2c3e50', linewidth=1.5, transform=ax.transAxes)

# Draw rows
y = header_y - 0.10
for item in table_data:
    if item is None:
        # Ellipsis row
        ax.text(0.50, y, "· · ·",
                fontsize=14, color='#1C1C1C',
                ha='center', va='center',
                transform=ax.transAxes)
        y -= row_height
        continue

    rank      = item["Rank"]
    highlight = rank in [4, 28]

    # Highlight background
    if highlight:
        highlight_box = plt.Rectangle(
            (0.05, y - 0.045), 0.90, row_height,
            transform=ax.transAxes,
            color='#FFFF00', alpha=0.35, zorder=0
        )
        ax.add_patch(highlight_box)

    # Row text
    font_color  = '#2c3e50'
    font_weight = 'bold' if highlight else 'normal'
    font_size   = 11.5

    ax.text(col_x[0], y, f"#{rank}",
            fontsize=font_size, fontweight=font_weight,
            color=font_color, ha='center', va='center',
            transform=ax.transAxes)

    ax.text(col_x[1], y, item["Team"],
            fontsize=font_size, fontweight=font_weight,
            color=font_color, ha='left', va='center',
            transform=ax.transAxes)

    ax.text(col_x[2], y, str(item["Power Rating"]),
            fontsize=font_size, fontweight=font_weight,
            color=font_color, ha='left', va='center',
            transform=ax.transAxes)

    # Light row divider
    if not highlight:
        ax.plot([0.05, 0.95], [y - 0.045, y - 0.045],
                color='#EEEEEE', linewidth=0.8,
                transform=ax.transAxes)

    y -= row_height

# Title
ax.set_title(
    "WHL Power Rankings",
    fontsize=16, fontweight='bold', pad=16, color='#2c3e50'
)

plt.tight_layout()
plt.savefig('PowerRankings_Table.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------
# MATCHUP PROBABILITY BAR
# Thailand (#4, home) vs Vietnam (#28, away)
# -------------------------

# Get win probability
prob_thailand = win_probability("thailand", "vietnam")
prob_vietnam  = 1 - prob_thailand

# Colors
THAILAND_COLOR = '#2980b9'  # blue
VIETNAM_COLOR  = '#c0392b'  # red

# -------------------------
# PLOT
# -------------------------
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 3))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')
ax.axis('off')

bar_y      = 0.45
bar_height = 0.30
bar_left   = 0.05
bar_right  = 0.95
bar_width  = bar_right - bar_left

# Thailand bar (left portion)
thailand_width = bar_width * prob_thailand
ax.add_patch(plt.Rectangle(
    (bar_left, bar_y), thailand_width, bar_height,
    transform=ax.transAxes,
    color=THAILAND_COLOR, zorder=2
))

# Vietnam bar (right portion)
ax.add_patch(plt.Rectangle(
    (bar_left + thailand_width, bar_y), bar_width - thailand_width, bar_height,
    transform=ax.transAxes,
    color=VIETNAM_COLOR, zorder=2
))

# Divider line
ax.plot(
    [bar_left + thailand_width, bar_left + thailand_width],
    [bar_y, bar_y + bar_height],
    color='white', linewidth=2.5,
    transform=ax.transAxes, zorder=3
)

# Probability labels inside bars
ax.text(
    bar_left + thailand_width / 2, bar_y + bar_height / 2,
    f"{prob_thailand:.1%}",
    ha='center', va='center',
    fontsize=15, fontweight='bold', color='white',
    transform=ax.transAxes, zorder=4
)
ax.text(
    bar_left + thailand_width + (bar_width - thailand_width) / 2,
    bar_y + bar_height / 2,
    f"{prob_vietnam:.1%}",
    ha='center', va='center',
    fontsize=15, fontweight='bold', color='white',
    transform=ax.transAxes, zorder=4
)

# Team labels above bar
ax.text(
    bar_left + thailand_width / 2, bar_y + bar_height + 0.08,
    "Thailand  (#4)",
    ha='center', va='bottom',
    fontsize=12, fontweight='bold', color=THAILAND_COLOR,
    transform=ax.transAxes
)
ax.text(
    bar_left + thailand_width + (bar_width - thailand_width) / 2,
    bar_y + bar_height + 0.08,
    "Vietnam  (#28)",
    ha='center', va='bottom',
    fontsize=12, fontweight='bold', color=VIETNAM_COLOR,
    transform=ax.transAxes
)

# Home indicator below bar
ax.text(
    bar_left + thailand_width / 2, bar_y - 0.08,
    "Home",
    ha='center', va='top',
    fontsize=9.5, color='#AAAAAA', fontstyle='italic',
    transform=ax.transAxes
)

# Title
ax.set_title(
    "Matchup Win Probability: Thailand vs. Vietnam",
    fontsize=15, fontweight='bold', pad=16, color='#2c3e50'
)

plt.tight_layout()
plt.savefig('Matchup_Probability.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------
# RESIDUAL PLOT
# Actual wins vs predicted wins based on process metrics
# Positive residual = overperforming (lucky), negative = underperforming (unlucky)
# -------------------------

# Calculate predicted wins for each team
# For each game, sum the win probability for each team across all 82 games

# Home team expected wins
home_expected = game_results[["home_team", "predicted_prob"]].rename(
    columns={"home_team": "team", "predicted_prob": "expected_win"}
)

# Away team expected wins (1 - home win prob)
away_expected = game_results[["away_team", "predicted_prob"]].copy()
away_expected["expected_win"] = 1 - away_expected["predicted_prob"]
away_expected = away_expected[["away_team", "expected_win"]].rename(
    columns={"away_team": "team"}
)

# Sum expected wins per team
expected_wins = (
    pd.concat([home_expected, away_expected])
    .groupby("team")["expected_win"]
    .sum()
    .reset_index()
)
expected_wins.columns = ["team", "expected_wins"]

# Merge with actual wins
residual_df = pd.merge(
    expected_wins,
    pd.DataFrame({"team": team_wins.index, "actual_wins": team_wins.values}),
    on="team"
)

residual_df["residual"]    = residual_df["actual_wins"] - residual_df["expected_wins"]
residual_df["team_label"]  = residual_df["team"].apply(format_team_name)
residual_df                = residual_df.sort_values("residual", ascending=False).reset_index(drop=True)

# Identify top 5 and bottom 5 outliers for labeling
top_outliers    = set(residual_df.nlargest(5, "residual")["team"])
bottom_outliers = set(residual_df.nsmallest(5, "residual")["team"])
label_teams     = top_outliers | bottom_outliers

# -------------------------
# PLOT
# -------------------------
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

# Color by residual
colors = ['#27ae60' if r > 0 else '#c0392b' for r in residual_df["residual"]]

# Scatter plot
ax.scatter(
    residual_df["expected_wins"],
    residual_df["actual_wins"],
    color=colors,
    s=120,
    alpha=0.88,
    edgecolors='white',
    linewidth=0.8,
    zorder=3
)

# Perfect prediction line (y = x)
min_val = min(residual_df["expected_wins"].min(), residual_df["actual_wins"].min()) - 1
max_val = max(residual_df["expected_wins"].max(), residual_df["actual_wins"].max()) + 1

ax.plot([min_val, max_val], [min_val, max_val],
        color='#2c3e50', linewidth=1.5,
        linestyle='--', alpha=0.6, zorder=2)

# Zone shading
ax.fill_between([min_val, max_val], [min_val, max_val], [max_val, max_val],
                alpha=0.03, color='#27ae60', zorder=0)
ax.fill_between([min_val, max_val], [min_val, max_val], [min_val, min_val],
                alpha=0.03, color='#c0392b', zorder=0)

# Zone labels
ax.text(min_val + 0.3, max_val - 0.5,
        "Overperforming\n(likely to regress)",
        fontsize=9.5, color='#27ae60', fontweight='bold', va='top')
ax.text(max_val - 0.3, min_val + 0.5,
        "Underperforming\n(potential sleepers)",
        fontsize=9.5, color='#c0392b', fontweight='bold', va='bottom', ha='right')

# Label outliers only
for _, row in residual_df.iterrows():
    if row["team"] in label_teams:
        ha     = 'left' if row["residual"] > 0 else 'right'
        offset = (6, 4) if row["residual"] > 0 else (-6, 4)
        ax.annotate(
            row["team_label"],
            xy=(row["expected_wins"], row["actual_wins"]),
            xytext=offset,
            textcoords='offset points',
            fontsize=8.5, fontweight='bold',
            color='#2c3e50', ha=ha,
            bbox=dict(boxstyle='round,pad=0.15',
                      facecolor='white', alpha=0.6, edgecolor='none')
        )

# Titles and axes
ax.set_title(
    "Actual vs. Expected Wins: Identifying Lucky and Unlucky Teams",
    fontsize=16, fontweight='bold', pad=18, color='#2c3e50'
)
ax.set_xlabel(
    "Expected Wins (Process-Based Model)",
    fontsize=12, labelpad=10, color='#2c3e50', fontweight='bold'
)
ax.set_ylabel(
    "Actual Wins",
    fontsize=12, labelpad=10, color='#2c3e50', fontweight='bold'
)

ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)
ax.legend(fontsize=10, framealpha=0.85, loc='upper left')
ax.tick_params(axis='both', labelsize=10, colors='#555555')

for spine in ax.spines.values():
    spine.set_edgecolor('#DDDDDD')

plt.tight_layout()
plt.savefig('Residual_Plot.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------
# WIN PERCENTAGE DISTRIBUTION
# Shows how narrow the spread is across all 32 teams
# Reinforces high-variance nature of hockey
# -------------------------

win_pct_df = pd.DataFrame({
    "team"    : standings.index,
    "win_pct" : standings["Wins"] / 82
}).sort_values("win_pct", ascending=False).reset_index(drop=True)

league_avg  = win_pct_df["win_pct"].mean()
top_win_pct = win_pct_df["win_pct"].max()
bot_win_pct = win_pct_df["win_pct"].min()
spread      = top_win_pct - bot_win_pct

# -------------------------
# PLOT
# -------------------------
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(11, 4))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

# Dots
ax.scatter(
    win_pct_df["win_pct"],
    np.zeros(len(win_pct_df)),
    color='#2c3e50',
    s=120,
    alpha=0.85,
    edgecolors='white',
    linewidth=0.8,
    zorder=3
)

# League average line
ax.axvline(league_avg, color='#2980b9', linewidth=1.8,
           linestyle='--', zorder=2)
ax.text(league_avg, 0.55,
        f"League Avg\n{league_avg:.1%}",
        ha='center', va='bottom', fontsize=9.5,
        color='#2980b9', fontweight='bold',
        transform=ax.get_xaxis_transform())

# Spread bracket
ax.set_xlabel(
    "Win Percentage",
    fontsize=11, labelpad=50, color='#2c3e50', fontweight='bold'
)

# Spread bracket
ax.annotate("",
    xy=(top_win_pct, -0.55), xycoords=('data', 'axes fraction'),
    xytext=(bot_win_pct, -0.55), textcoords=('data', 'axes fraction'),
    arrowprops=dict(arrowstyle='<->', color='#c0392b', lw=1.8)
)
ax.text((top_win_pct + bot_win_pct) / 2, -0.68,
        f"Spread: {spread:.1%}",
        ha='center', va='top', fontsize=9.5,
        color='#c0392b', fontweight='bold',
        transform=ax.get_xaxis_transform())


# Best and worst reference lines
ax.axvline(top_win_pct, color='#AAAAAA', linewidth=1.0,
           linestyle=':', zorder=1)
ax.axvline(bot_win_pct, color='#AAAAAA', linewidth=1.0,
           linestyle=':', zorder=1)

ax.text(top_win_pct, 0.55, f"Best\n{top_win_pct:.1%}",
        ha='center', va='bottom', fontsize=9,
        color='#AAAAAA', fontstyle='italic',
        transform=ax.get_xaxis_transform())
ax.text(bot_win_pct, 0.55, f"Worst\n{bot_win_pct:.1%}",
        ha='center', va='bottom', fontsize=9,
        color='#AAAAAA', fontstyle='italic',
        transform=ax.get_xaxis_transform())

# Clean up y axis
ax.set_yticks([])
ax.set_ylim(-1, 1)

# X axis formatting
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.tick_params(axis='x', labelsize=10, colors='#555555')

for spine in ax.spines.values():
    spine.set_edgecolor('#DDDDDD')

# Titles
ax.set_title(
    "WHL Win Percentage Distribution Across All 32 Teams",
    fontsize=15, fontweight='bold', pad=16, color='#2c3e50'
)
ax.set_xlabel(
    "Win Percentage",
    fontsize=11, labelpad=30, color='#2c3e50', fontweight='bold'
)

plt.figtext(
    0.5, -0.08,
    "The gap between 1st and 32nd is smaller than you'd expect — in hockey, any team can beat any team on a given night.",
    ha="center", fontsize=10, color='#555555',
    bbox={"facecolor": "#FFF3CD", "alpha": 0.8, "pad": 5, "edgecolor": "#DDDDDD"}
)

plt.tight_layout(rect=[0, 0.15, 1, 1.25])
plt.savefig('WinPct_Distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------
# COMPONENT HEATMAP
# Top 10 and bottom 10 teams by Power Rating
# Z-scores for each component, sorted by Power Rating
# -------------------------

# Build component dataframe
heatmap_data = pd.DataFrame({
    "team"          : team_stats.index,
    "Power_Rating"  : team_stats["Power_Rating"],
    "Process"       : team_stats["Process_Score"],
    "Goaltending"   : team_stats["Goalie_Score"],
    "Special Teams" : team_stats["Special_Teams_Score"],
    "Results"       : team_stats["Results_Score"]
}).sort_values("Power_Rating", ascending=False).reset_index(drop=True)

# Top 10 and bottom 10
top10    = heatmap_data.head(10)
bottom10 = heatmap_data.tail(10)
plot_df  = pd.concat([top10, bottom10]).reset_index(drop=True)

# Format team names
plot_df["team_label"] = plot_df["team"].apply(format_team_name)

# Matrix for heatmap
components  = ["Process", "Goaltending", "Special Teams", "Results"]
matrix      = plot_df[components].values
team_labels = plot_df["team_label"].tolist()

# -------------------------
# PLOT
# -------------------------
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 10))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

# Custom diverging colormap — green to white to red
from matplotlib.colors import TwoSlopeNorm
import matplotlib.colors as mcolors

cmap = plt.cm.RdYlGn
norm = TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5)

im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')

# Cell annotations — z-score values
for i in range(len(team_labels)):
    for j in range(len(components)):
        val        = matrix[i, j]
        text_color = 'white' if abs(val) > 1.5 else '#2c3e50'
        ax.text(j, i, f"{val:.2f}",
                ha='center', va='center',
                fontsize=10, fontweight='bold',
                color=text_color)

# Divider line between top 10 and bottom 10
ax.axhline(9.5, color='#2c3e50', linewidth=2.0, zorder=5)

# Top 10 / Bottom 10 labels on right side
ax.text(len(components) - 0.5 + 0.6, 4.5, "Top 10",
        ha='left', va='center', fontsize=10,
        fontweight='bold', color='#27ae60',
        rotation=90, transform=ax.transData)
ax.text(len(components) - 0.5 + 0.6, 14.5, "Bottom 10",
        ha='left', va='center', fontsize=10,
        fontweight='bold', color='#c0392b',
        rotation=90, transform=ax.transData)

# Axes
ax.set_xticks(range(len(components)))
ax.set_xticklabels(components, fontsize=12,
                   fontweight='bold', color='#2c3e50')
ax.set_yticks(range(len(team_labels)))
ax.set_yticklabels(team_labels, fontsize=10,
                   fontweight='bold', color='#2c3e50')

ax.tick_params(axis='both', length=0)
ax.set_xlim(-0.5, len(components) - 0.5)

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label("Z-Score", fontsize=10,
               fontweight='bold', color='#2c3e50')
cbar.ax.tick_params(labelsize=9, colors='#555555')

# Title
ax.set_title(
    "What Makes WHL Teams Elite?\nPower Rating Components by Team",
    fontsize=15, fontweight='bold', pad=18, color='#2c3e50'
)

for spine in ax.spines.values():
    spine.set_edgecolor('#DDDDDD')

plt.tight_layout()
plt.savefig('Component_Heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

components = {
    "Process"       : team_stats["Process_Score"],
    "Goaltending"   : team_stats["Goalie_Score"],
    "Special Teams" : team_stats["Special_Teams_Score"],
    "Results"       : team_stats["Results_Score"]
}

print("Correlation of each component with Power Rating:")
print("-" * 40)
for name, scores in components.items():
    corr = scores.corr(team_stats["Power_Rating"])
    print(f"{name:<15}: {corr:.4f}")

actual_std = standings["Wins"].std()
theoretical_std = np.sqrt(82 * 0.5 * 0.5)

print(f"Actual std of wins: {actual_std:.2f}")
print(f"Theoretical random std: {theoretical_std:.2f}")
print(f"Ratio: {actual_std/theoretical_std:.2f}")

# -------------------------
# COMPONENT CORRELATION BAR CHART
# Shows how strongly each component predicts Power Rating
# Goaltending highlighted as the surprising weak link
# -------------------------

corr_data = {
    "Process"       : 0.8700,
    "Special Teams" : 0.8655,
    "Results"       : 0.8472,
    "Goaltending"   : 0.6972
}

labels = list(corr_data.keys())
values = list(corr_data.values())
colors = ['#2c3e50' if label != 'Goaltending' else '#c0392b' for label in labels]

# -------------------------
# PLOT
# -------------------------
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

bars = ax.bar(
    labels, values,
    color=colors,
    edgecolor='white',
    linewidth=0.8,
    alpha=0.90,
    width=0.55
)

# Value labels on bars
for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{val:.2f}",
        ha='center', va='bottom',
        fontsize=12, fontweight='bold',
        color='#2c3e50'
    )

# Goaltending annotation
ax.annotate(
    "Goaltending is\n0.17 below next\nlowest predictor",
    xy=(3, 0.6972),
    xytext=(2.3, 0.75),
    fontsize=9.5, color='#c0392b', fontweight='bold',
    arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.2)
)

# Reference line at goaltending level
ax.axhline(0.6972, color='#c0392b', linewidth=1.0,
           linestyle=':', alpha=0.5, zorder=1)

# Y axis range
ax.set_ylim(0.5, 0.95)
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))

# Titles and axes
ax.set_title(
    "How Strongly Does Each Component\nPredict WHL Team Quality?",
    fontsize=15, fontweight='bold', pad=18, color='#2c3e50'
)
ax.set_ylabel(
    "Correlation with Power Rating",
    fontsize=11, labelpad=10, color='#2c3e50', fontweight='bold'
)
ax.set_xlabel("")

ax.tick_params(axis='x', labelsize=12, colors='#2c3e50')
ax.tick_params(axis='y', labelsize=10, colors='#555555')

for label in ax.get_xticklabels():
    if label.get_text() == 'Goaltending':
        label.set_color('#c0392b')
        label.set_fontweight('bold')

for spine in ax.spines.values():
    spine.set_edgecolor('#DDDDDD')

plt.figtext(
    0.5, -0.02,
    "Goaltending correlates at 0.70 vs 0.87 for process and special teams — the most volatile component evens out over a full season.",
    ha="center", fontsize=9.5, color='#555555',
    bbox={"facecolor": "#FFF3CD", "alpha": 0.8, "pad": 5, "edgecolor": "#DDDDDD"}
)

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig('Component_Correlations.png', dpi=300, bbox_inches='tight')
plt.show()

actual_wins_series = standings["Wins"]

print("Correlation of each component with Actual Wins:")
print("-" * 40)
for name, scores in components.items():
    corr = scores.corr(actual_wins_series)
    print(f"{name:<15}: {corr:.4f}")

# -------------------------
# COMPONENT CORRELATION WITH ACTUAL WINS
# Shows how strongly each component predicts real game outcomes
# Excludes Results since it contains wins by design
# -------------------------

corr_actual = {
    "Process"       : 0.6118,
    "Special Teams" : 0.6232,
    "Goaltending"   : 0.7044,
}

labels = list(corr_actual.keys())
values = list(corr_actual.values())

# Goaltending highlighted as the surprising strongest predictor
colors = ['#2c3e50' if label != 'Goaltending' else '#e67e22' for label in labels]

# -------------------------
# PLOT
# -------------------------
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

bars = ax.bar(
    labels, values,
    color=colors,
    edgecolor='white',
    linewidth=0.8,
    alpha=0.90,
    width=0.50
)

# Value labels on bars
for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.003,
        f"{val:.2f}",
        ha='center', va='bottom',
        fontsize=13, fontweight='bold',
        color='#2c3e50'
    )

# Goaltending annotation
ax.annotate(
    "Strongest predictor\nof actual game outcomes",
    xy=(2, 0.7044),
    xytext=(1.4, 0.72),
    fontsize=9.5, color='#e67e22', fontweight='bold',
    arrowprops=dict(arrowstyle='->', color='#e67e22', lw=1.2)
)

# Reference lines
ax.axhline(0.7044, color='#e67e22', linewidth=1.0,
           linestyle=':', alpha=0.5, zorder=1)

# Y axis
ax.set_ylim(0.50, 0.78)
ax.set_yticks([0.50, 0.55, 0.60, 0.65, 0.70, 0.75])
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x:.2f}")
)

# Titles and axes
ax.set_title(
    "Which Components Actually Predict Wins?",
    fontsize=15, fontweight='bold', pad=18, color='#2c3e50'
)
ax.set_ylabel(
    "Correlation with Actual Wins",
    fontsize=11, labelpad=10, color='#2c3e50', fontweight='bold'
)

ax.tick_params(axis='x', labelsize=12, colors='#2c3e50')
ax.tick_params(axis='y', labelsize=10, colors='#555555')

# Highlight goaltending x label
for label in ax.get_xticklabels():
    if label.get_text() == 'Goaltending':
        label.set_color('#e67e22')
        label.set_fontweight('bold')

for spine in ax.spines.values():
    spine.set_edgecolor('#DDDDDD')

plt.figtext(
    0.5, -0.02,
    "Results excluded — contains win percentage by design (r=0.98). "
    "Goaltending is the strongest predictor of actual game outcomes "
    "despite being the most volatile component across teams.",
    ha="center", fontsize=9, color='#555555',
    bbox={"facecolor": "#FFF3CD", "alpha": 0.8, "pad": 5,
          "edgecolor": "#DDDDDD"}
)

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig('Component_Actual_Wins_Correlation.png', dpi=300,
            bbox_inches='tight')
plt.show()

# -------------------------
# CALIBRATION PLOT — CONCLUSION SLIDE
# -------------------------
from sklearn.calibration import calibration_curve

actual    = game_results["home_win"]
predicted = game_results["predicted_prob"]

prob_true, prob_pred = calibration_curve(
    actual, predicted, n_bins=10, strategy='uniform'
)

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

# Perfect calibration line
ax.plot([0.35, 0.75], [0.35, 0.75],
        linestyle='--', color='#AAAAAA',
        linewidth=1.5, label='Perfect Calibration', zorder=1)

# Model calibration line
ax.plot(prob_pred, prob_true,
        marker='o', linewidth=2.5,
        color='#2c3e50', markersize=8,
        markeredgecolor='white', markeredgewidth=1.0,
        label='Our Model', zorder=3)

# AUC annotation
ax.text(0.05, 0.95,
        "AUC = 0.606",
        fontsize=11, fontweight='bold',
        color='#2c3e50',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.4',
                  facecolor='white', alpha=0.8,
                  edgecolor='#DDDDDD'))

# Titles and axes
ax.set_title(
    "Model Calibration",
    fontsize=14, fontweight='bold', pad=14, color='#2c3e50'
)
ax.set_xlabel(
    "Predicted Win Probability",
    fontsize=11, labelpad=8,
    color='#2c3e50', fontweight='bold'
)
ax.set_ylabel(
    "Observed Win Rate",
    fontsize=11, labelpad=8,
    color='#2c3e50', fontweight='bold'
)

ax.set_xlim(0.35, 0.75)
ax.set_ylim(0.35, 0.75)
ax.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x:.0%}")
)
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x:.0%}")
)

ax.legend(fontsize=10, framealpha=0.85, loc='upper right')
ax.tick_params(axis='both', labelsize=10, colors='#555555')

for spine in ax.spines.values():
    spine.set_edgecolor('#DDDDDD')

plt.figtext(
    0.5, -0.02,
    "When our model predicts 65%, teams win ~65% of the time — probabilities are trustworthy across all prediction ranges.",
    ha="center", fontsize=9.5, color='#555555',
    bbox={"facecolor": "#FFF3CD", "alpha": 0.8,
          "pad": 5, "edgecolor": "#DDDDDD"}
)

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig('Calibration_Plot_Conclusion.png', dpi=300,
            bbox_inches='tight')
plt.show()
