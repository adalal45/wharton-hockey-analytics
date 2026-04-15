# Wharton-Hockey-Analytics-Competition

2024 Wharton Sports Analytics Competition — Top 6% Internationally (500+ Teams)

# Overview
This project was submitted to the Wharton Sports Analytics Competition, where it placed in the top 6% out of 500+ international teams. Using a fictional hockey league dataset, the analysis was presented directly to the league commissioner as a full analytical report with actionable findings.
The project tackled three core questions:
1. Matchup prediction — Can we accurately predict game outcomes and win probabilities between teams?
2. Power rankings — How do teams rank relative to each other beyond just win-loss record?
3. Line disparity — Does the quality gap between a team's first and second lines meaningfully affect overall team performance?

# Methods
· Logistic regression to model and predict head-to-head matchup outcomes and win probabilities
· Linear regression to quantify the relationship between line disparity and team quality
· Power ranking model to score and rank teams using performance metrics
· Data visualization (matplotlib, seaborn) to communicate findings clearly to a non-technical audience

# Tools & Libraries
· Python · pandas · matplotlib · seaborn · scikit-learn
· Colab Notebook

# Key Findings
· Probability model predicted win rates closely match observed outcomes across all buckets (AUC = 0.606)
· Line disparity was not found to be a statistically significant predictor of overall team power
· Power rankings revealed meaningful separation between tiers of teams not captured by raw standings

# Repository Structure
wharton-hockey-analytics/
├── analysis.ipynb        # Main notebook with full analysis
├── data/                 # Dataset(s) used
├── figures/              # Exported visualizations
└── README.md

# Presentation
Full slide deck presented to the league commissioner: [[Link to Google Slides]](https://docs.google.com/presentation/d/10VPTREwxN7Do8WYaXcFRqNKhd64VyQqNztOSBSTgUKs/edit?usp=sharing)
