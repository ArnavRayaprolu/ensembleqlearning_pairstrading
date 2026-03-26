import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
import random
import json

# Load data
z_train = pd.read_csv("zscore_spreads.csv", index_col=0, parse_dates=True)
z_eval = pd.read_csv("zscore_spreads2.csv", index_col=0, parse_dates=True)
price_train = pd.read_csv("nifty50_hourly_prices.csv", index_col=0, parse_dates=True)
price_eval = pd.read_csv("nifty50_hourly_prices2.csv", index_col=0, parse_dates=True)

# Load best parameters
with open("best_hyperparameters.json", "r") as f:
    best_params = json.load(f)

alpha = float(best_params.get("alpha", 0.015))
gamma = float(best_params.get("gamma", 0.98))
eps = float(best_params.get("eps", 0.2))
decay = float(best_params.get("decay", 0.9999))
entry_zscore = float(best_params.get("entry_zscore", 0.01))

# Optimistic initialization value for Q-table
OPTIMISTIC_Q = 10.0  # You may adjust this depending on expected rewards

# Ensemble size
ENSEMBLE_SIZE = 3

WINDOW = 3
Z_BINS = 15
HOLD_POS = 3
STATE_SIZE = (Z_BINS ** WINDOW) * HOLD_POS

class PairsTradingEnv(gym.Env):
    def __init__(self, z_scores, tc=0.0009, lmb=0.2, cap0=100000, entry_zscore=0.01):
        super().__init__()
        self.z = z_scores
        self.n = len(z_scores)
        self.tc = tc
        self.lmb = lmb
        self.cap0 = cap0
        self.max_dd = 0.25 * cap0
        self.min_holding = 1
        self.cooldown_steps = 1
        self.entry_zscore = entry_zscore
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Discrete(STATE_SIZE)
        self.reset()

    def reset(self):
        self.i = WINDOW
        self.cap = self.cap0
        self.pos = 0
        self.entry_z = None
        self.entry_i = None
        self.entry_amt = 0
        self.last_exit_i = -1000
        self.cap_history = [self.cap]
        self.trades = []
        return self._get_state()

    def step(self, action):
        done = False
        reward = -0.001
        z_cur = self.z[self.i]
        z_prev = self.z[self.i - 1] if self.i > 0 else 0
        cooldown = (self.i - self.last_exit_i) < self.cooldown_steps
        holding = self.i - (self.entry_i if self.entry_i is not None else self.i)

        if self.pos == 0 and not cooldown:
            if action in [1, 2, 3, 4]:
                can_enter = abs(z_cur) >= self.entry_zscore or (z_cur * z_prev < 0)
                if can_enter:
                    self.pos = 1 if action in [1, 2] else -1
                    amt = self.cap
                    self.entry_amt = amt
                    self.entry_z = z_cur
                    self.entry_i = self.i
                    self.cap -= self.tc * amt
                    self.trades.append(('entry', self.i, self.pos, z_cur, amt))
                    reward += abs(z_cur) * 0.5 + 0.05

        elif self.pos != 0 and action == 5 and holding >= self.min_holding:
            closer_to_mean = abs(z_cur) < abs(self.entry_z)
            pnl = self.entry_amt * self.pos * (self.entry_z - z_cur) / 10
            profit_ratio = pnl / self.entry_amt
            if closer_to_mean and profit_ratio > 0.0005:
                self.cap += pnl - self.tc * self.entry_amt
                reward += 50 * profit_ratio + 0.3 * (1 - abs(z_cur))
                self.trades.append(('exit', self.i, self.pos, z_cur, pnl))
                self.pos = 0
                self.entry_amt = 0
                self.entry_z = None
                self.entry_i = None
                self.last_exit_i = self.i
            else:
                reward -= 0.2

        if self.pos != 0:
            too_far = abs(z_cur - self.entry_z) > 7
            too_long = holding > 60
            if too_far or too_long:
                pnl = self.entry_amt * self.pos * (self.entry_z - z_cur) / 10
                self.cap += pnl - self.tc * self.entry_amt
                self.trades.append(('forced_exit', self.i, self.pos, z_cur, pnl))
                reward -= 1.0
                self.pos = 0
                self.entry_amt = 0
                self.last_exit_i = self.i

        if self.cap > self.cap0:
            reward += 0.05 * (self.cap - self.cap0) / self.cap0

        if self.cap0 - self.cap > self.max_dd or self.i >= self.n - 1:
            done = True

        self.i += 1
        self.cap_history.append(self.cap)
        return self._get_state(), reward, done, {}

    def _get_state(self):
        zvals = self.z[self.i - WINDOW:self.i]
        zidx = [min(Z_BINS - 1, max(0, int((z + 5) / 10 * Z_BINS))) for z in zvals]
        hist = sum(zidx[j] * (Z_BINS ** j) for j in range(WINDOW))
        pos_idx = {0: 0, 1: 1, -1: 2}[self.pos]
        return hist * HOLD_POS + pos_idx

def train_single_q(env, N, alpha, gamma, eps, decay, optimistic_value=15.0):
    Q = np.full((STATE_SIZE, env.action_space.n), optimistic_value, dtype=np.float64)
    episode_rewards = []
    for ep in range(N):
        s = env.reset()
        done = False
        total_reward = 0
        while not done:
            if random.random() < eps:
                a = random.randint(0, env.action_space.n - 1)
            else:
                a = np.argmax(Q[s])
            ns, r, done, _ = env.step(a)
            Q[s, a] += alpha * (r + gamma * np.max(Q[ns]) - Q[s, a])
            s = ns
            total_reward += r
        episode_rewards.append(total_reward)
        eps = max(0.01, eps * decay)
    return Q, episode_rewards

def compute_total_return(hist):
    return ((hist[-1] - hist[0]) / hist[0]) * 100

def compute_sharpe_ratio(hist, risk_free_rate=0.0679, periods_per_year=853):
    returns = np.diff(hist) / hist[:-1]
    excess_returns = returns - (risk_free_rate / periods_per_year)
    if returns.std() == 0:
        return 0.0
    return (excess_returns.mean() / returns.std()) * np.sqrt(periods_per_year)

def compute_max_drawdown(hist):
    hist = np.array(hist)
    running_max = np.maximum.accumulate(hist)
    drawdown = (hist - running_max) / running_max
    return drawdown.min()

# Main evaluation
selected_pairs = [
    "BPCL_CIPLA",
    "CIPLA_NTPC",
    "ADANIENT_TATACONSUM",
    "EICHERMOT_M&M",
    "HDFCBANK_ICICIBANK"
]

with open("evaluation_results.txt", "w") as f:
    for pair in selected_pairs:
        print(f"\nEvaluating {pair}", file=f)
        z_train_series = z_train[pair].values
        z_eval_series = z_eval[pair].values

        results = []
        all_capitals_runs = []

        for run in range(10):  # 5 evaluation runs per pair
            Q_ensemble = []
            for ens in range(ENSEMBLE_SIZE):
                rng = np.random.default_rng()  # For maximum randomness (system entropy)
                env = PairsTradingEnv(z_train_series, entry_zscore=entry_zscore)
                Q, episode_rewards = train_single_q(
                    env, N=50000, alpha=alpha, gamma=gamma, eps=eps, decay=decay,
                    optimistic_value=OPTIMISTIC_Q, 
                )
                Q_ensemble.append(Q)

            Qs_stacked = np.array(Q_ensemble)
            Q_avg = Qs_stacked.mean(axis=0)

            plt.figure(figsize=(10, 4))
            plt.plot(episode_rewards, label='Episode Reward (last ensemble)')
            plt.title(f'{pair} – Run {run+1} Episode Reward (Ensemble Q)')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True)
            plt.tight_layout()
            plt.legend()
            plt.savefig(f"{pair}_run{run+1}_reward_curve.png", dpi=200)
            plt.close()

            env_eval = PairsTradingEnv(z_eval_series, entry_zscore=entry_zscore)
            s = env_eval.reset()
            done = False
            while not done:
                a = np.argmax(Q_avg[s])
                s, _, done, _ = env_eval.step(a)

            trades = env_eval.trades
            hist = env_eval.cap_history

            plt.figure(figsize=(10, 4))
            plt.plot(hist, label=f'Run {run+1} Capital')
            plt.title(f'{pair} – Capital Over Time (Run {run+1})')
            plt.xlabel('Time Steps')
            plt.ylabel('Capital')
            plt.grid(True)
            plt.tight_layout()
            plt.legend()
            plt.savefig(f"{pair}_run{run+1}_capital.png", dpi=200)
            plt.close()

            wins = sum(1 for t in trades if t[0] in ('exit', 'forced_exit') and t[4] > 0)
            losses = sum(1 for t in trades if t[0] in ('exit', 'forced_exit') and t[4] <= 0)
            total = wins + losses
            ret = compute_total_return(hist)
            sharpe = compute_sharpe_ratio(hist)
            max_dd = compute_max_drawdown(hist)
            total_trades = total

            print(f"Run {run+1}: Return: {ret:.2f}%, Sharpe: {sharpe:.2f}, Win Rate: {100*wins/total:.2f}%, Total Trades: {total_trades}, Max Drawdown: {max_dd:.2%}", file=f)
            f.flush()

            results.append((ret, sharpe, max_dd, total_trades))
            all_capitals_runs.append(hist)

        min_len = min(len(c) for c in all_capitals_runs)
        trimmed_runs = np.array([c[:min_len] for c in all_capitals_runs])
        np.save(f"{pair}_all_capitals.npy", trimmed_runs)

        returns, sharpes, max_dds, total_trades_list = zip(*results)
        print(f"\n>> {pair} Summary over {len(returns)} runs:", file=f)
        print(f"Average Return: {np.mean(returns):.2f}% ± {np.std(returns):.2f}%", file=f)
        print(f"Average Sharpe: {np.mean(sharpes):.2f} ± {np.std(sharpes):.2f}", file=f)
        print(f"Average Max Drawdown: {np.mean(max_dds):.2%} ± {np.std(max_dds):.2%}", file=f)
        print(f"Average Total Trades: {np.mean(total_trades_list):.2f} ± {np.std(total_trades_list):.2f}", file=f)
        f.flush()

        avg_capital = trimmed_runs.mean(axis=0)
        std_capital = trimmed_runs.std(axis=0)
        np.save(f"{pair}_capitals.npy", avg_capital)

        plt.figure(figsize=(13, 4))
        plt.plot(avg_capital, label='Mean Capital')
        plt.fill_between(
            range(min_len),
            avg_capital - std_capital,
            avg_capital + std_capital,
            color='gray',
            alpha=0.3,
            label='±1 Std Dev'
        )
        plt.title(f"{pair} – Capital over Evaluation Period")
        plt.xlabel("Time Steps")
        plt.ylabel("Capital")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{pair}_capital_plot.png", dpi=300)
        plt.close()

# === Combined Total Capital Plot (with padding for different time lengths) ===

all_capitals_runs = []
for pair in selected_pairs:
    try:
        capitals = np.load(f"{pair}_all_capitals.npy")
        print(f"Loaded {pair}_all_capitals.npy with shape {capitals.shape}")
        all_capitals_runs.append(capitals)
    except FileNotFoundError:
        print(f"Warning: Skipping {pair}, no capital data found.")
        continue

if all_capitals_runs:
    max_len = max(c.shape[1] for c in all_capitals_runs)
    padded = []
    for capitals in all_capitals_runs:
        runs, length = capitals.shape
        if length < max_len:
            pad = np.tile(capitals[:, -1:], (1, max_len - length))
            padded_capitals = np.concatenate([capitals, pad], axis=1)
        else:
            padded_capitals = capitals
        padded.append(padded_capitals)
    stacked = np.stack(padded)  # shape: (n_pairs, runs, max_len)
    total_capital_runs = np.sum(stacked, axis=0)
    mean_total_capital = total_capital_runs.mean(axis=0)
    std_total_capital = total_capital_runs.std(axis=0)

    initial_capital = mean_total_capital[0]
    final_capital = mean_total_capital[-1]
    total_return = (final_capital - initial_capital) / initial_capital * 100

    risk_free_rate = 0.0679
    periods_per_year = 853
    returns = np.diff(mean_total_capital) / mean_total_capital[:-1]
    excess_returns = returns - (risk_free_rate / periods_per_year)
    if returns.std() == 0:
        portfolio_sharpe = 0.0
    else:
        portfolio_sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(periods_per_year)
    portfolio_max_dd = compute_max_drawdown(mean_total_capital)

    print(f"Portfolio Max Drawdown: {portfolio_max_dd:.2%}")
    print(f"Portfolio Total Return: {final_capital:.2f}")
    print(f"Portfolio Total Return (%): {total_return:.2f}%")
    print(f"Portfolio Sharpe Ratio: {portfolio_sharpe:.2f}")

    plt.figure(figsize=(14, 4))
    plt.plot(mean_total_capital, label='Combined Avg Capital')
    plt.fill_between(
        range(len(mean_total_capital)),
        mean_total_capital - std_total_capital,
        mean_total_capital + std_total_capital,
        color='gray',
        alpha=0.3,
        label='±1 Std Dev'
    )
    plt.title("Combined Capital Over Evaluation Period (All Pairs)")
    plt.xlabel("Time Steps")
    plt.ylabel("Total Portfolio Capital")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("total_combined_capital_plot.png", dpi=300)
    plt.close()
