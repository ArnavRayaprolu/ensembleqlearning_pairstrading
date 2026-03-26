from graphviz import Digraph

g = Digraph(name='Pairs Trading Ensemble Q-Learning', format='png')

# --- A4 sizing for 0.83" margins: usable area 6.61 x 10.03 inches ---
g.graph_attr['size'] = '6.61,10.03!'
g.graph_attr['margin'] = '0.0'
g.graph_attr['dpi'] = '300'         # Good for print publications
g.graph_attr['pad'] = '0.2'         # Tight but readable

# --- (Rest of your diagram code from above, unchanged) ---

# Data nodes
g.node('RAW', 'Raw NIFTY50 Hourly Prices\n(Jan 2024–Jul 2025)', shape='folder', style='filled', fillcolor='lightgray')
g.node('CLEAN', 'Preprocessing\n(Missing Value Imputation,\nTime Sync)', shape='parallelogram', fillcolor='lightblue')
g.node('PAIR', 'Pair Selection (1,225 pairs)\nComposite Score:\n• Engle-Granger Cointegration\n• Z-Score Cross Frequency', shape='parallelogram', fillcolor='lightyellow')
g.node('PSEL', 'Top 5 Pairs\n(Composite Ranking)', shape='box', fillcolor='lightyellow')
g.node('ZSCORE', 'Spread & Z-score\nFeature Calculation', shape='box', fillcolor='white')
g.node('DS', 'Train/Test Split', shape='parallelogram', fillcolor='white')

# RL block
g.node('ENV', 'Custom Gym Trading Environment\nPairsTradingEnv\n(state, reward, capital, trade constraints)', shape='parallelogram', fillcolor='lightgreen')
g.node('STATE', 'State Encoding\n(z-score window [15 bins x 3],\npos code = 3)', shape='box', fillcolor='honeydew2')
g.node('QENS', 'Ensemble Q-Learning\n(n=3 Q-tables,\nOptimistic Init., Epsilon-Greedy)', shape='folder', fillcolor='lightgoldenrod1')
g.node('HPO', 'Bayesian Hyperparam.\nTuning (Optuna)', shape='ellipse', fillcolor='bisque')
g.node('ACT', 'Action Selection\n(Enter/Exit/Hold/Alt)', shape='box', fillcolor='palegoldenrod')
g.node('RWD', 'Reward Calculation\n(Activity, Entry, Exit,\nForced Exit, Growth)', shape='box', fillcolor='palegreen')
g.node('UPDATE', 'Q-table Update', shape='box', fillcolor='peachpuff')

# Eval/output
g.node('OOS', 'Out-of-Sample Eval\n(2025 Jan–Jul)', shape='parallelogram', fillcolor='azure2')
g.node('LOG', 'Performance Metrics:\nReturn, Sharpe, Win Rate,\nDrawdown\n(Time Series Logging)', shape='note', fillcolor='lightcyan')
g.node('PLOT', 'Visualization & Reporting:\nCapital Curves, Std Dev Bands\n(Export to PNG & TXT File)', shape='note', fillcolor='lightsteelblue')

# Arrows between stages
g.edge('RAW', 'CLEAN', label='procure\nvia Yahoo API')
g.edge('CLEAN', 'PAIR', label='all pairs\n(1,225)')
g.edge('PAIR', 'PSEL', label='composite\nscore')
g.edge('PSEL', 'ZSCORE', label='5 pairs')
g.edge('ZSCORE', 'DS', label='features')
g.edge('DS', 'ENV', label='z-score')
g.edge('HPO', 'QENS', label='best\nparams')
g.edge('STATE', 'ENV', label='compact state', style='dashed')
g.edge('ENV', 'STATE', label='sliding\nz-score\nwindow', style='dashed')
g.edge('ENV', 'ACT', label='call step()', style='dotted')
g.edge('QENS', 'ACT', label='policy\nQ(s,a)', style='bold')
g.edge('ACT', 'ENV', label='action')
g.edge('ENV', 'RWD', label='reward,cap')
g.edge('RWD', 'QENS', label='update')
g.edge('QENS', 'UPDATE')
g.edge('ENV', 'QENS', label='episodic\ninteraction', style='dashed')
g.edge('QENS', 'OOS', label='mean Q-table')
g.edge('OOS', 'LOG', label='capital,\ntrades')
g.edge('LOG', 'PLOT', label='metrics\n& curves')

# Group: Data, RL core, outputs
with g.subgraph(name='cluster_data') as d:
    d.attr(label='Data Pipeline & Pair Selection', color='lightblue')
    d.node('RAW')
    d.node('CLEAN')
    d.node('PAIR')
    d.node('PSEL')
    d.node('ZSCORE')
    d.node('DS')
    
with g.subgraph(name='cluster_rl') as d:
    d.attr(label='RL Agent Training & Interaction', color='lightgreen')
    d.node('ENV')
    d.node('STATE')
    d.node('QENS')
    d.node('HPO')
    d.node('ACT')
    d.node('RWD')
    d.node('UPDATE')
    
with g.subgraph(name='cluster_eval') as d:
    d.attr(label='Evaluation & Reporting', color='lightcyan')
    d.node('OOS')
    d.node('LOG')
    d.node('PLOT')

g.attr(rankdir='TB')  # Portrait orientation for A4, top to bottom

g.render('enhanced_pairstrading_model_a4', format='png', view=True)
