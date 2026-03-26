import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
import random
import json
import statsmodels.api as sm
import logging
import time
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

logger.info("Starting script execution")

# Load data
logger.info("Loading data files")
try:
    z_train = pd.read_csv("zscore_spreads.csv", index_col=0, parse_dates=True)
    z_eval = pd.read_csv("zscore_spreads2.csv", index_col=0, parse_dates=True)
    price_train = pd.read_csv("nifty50_hourly_prices.csv", index_col=0, parse_dates=True)
    price_eval = pd.read_csv("nifty50_hourly_prices2.csv", index_col=0, parse_dates=True)
    logger.info(f"Loaded z_train: {z_train.shape}, z_eval: {z_eval.shape}, price_train: {price_train.shape}, price_eval: {price_eval.shape}")
except FileNotFoundError as e:
    logger.error(f"Data file not found: {e}")
    exit(1)

# Load best parameters
logger.info("Loading hyperparameters")
try:
    with open("best_hyperparameters.json", "r") as f:
        best_params = json.load(f)
    logger.info(f"Loaded hyperparameters: {best_params}")
except FileNotFoundError:
    logger.error("best_hyperparameters.json not found")
    exit(1)

alpha = float(best_params.get("alpha", 0.015))
gamma = float(best_params.get("gamma", 0.98))
eps = float(best_params.get("eps", 0.2))
decay = float(best_params.get("decay", 0.9999))
entry_zscore = float(best_params.get("entry_zscore", 0.01))
logger.info(f"Parameters: alpha={alpha}, gamma={gamma}, eps={eps}, decay={decay}, entry_zscore={entry_zscore}")

# Optimistic initialization value for Q-table
OPTIMISTIC_Q = 10.0

# Ensemble size
ENSEMBLE_SIZE = 3

WINDOW = 3
Z_BINS = 15
HOLD_POS = 3
STATE_SIZE = (Z_BINS ** WINDOW) * HOLD_POS
logger.info(f"Environment setup: WINDOW={WINDOW}, Z_BINS={Z_BINS}, HOLD_POS={HOLD_POS}, STATE_SIZE={STATE_SIZE}")

class PairsTradingEnv(gym.Env):
    def __init__(self, z_scores, prices_stock1, prices_stock2, beta, tc=0.0009, lmb=0.2, cap0=100000, entry_zscore=0.01):
        super().__init__()
        self.z = z_scores
        self.n = len(z_scores)
        self.prices1 = prices_stock1
        self.prices2 = prices_stock2
        self.beta = beta
        self.tc = tc
        self.lmb = lmb
        self.cap0 = cap0
        self.max_dd = 0.25 * cap0
        self.min_holding = 1
        self.cooldown_steps = 1
        self.entry_zscore = entry_zscore
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Discrete(STATE_SIZE)
        # Validate data lengths
        if not (len(self.prices1) == len(self.prices2) == self.n):
            logger.error(f"Length mismatch: z_scores={len(self.z)}, prices_stock1={len(self.prices1)}, prices_stock2={len(self.prices2)}")
            raise ValueError("Length mismatch between z_scores, prices_stock1, and prices_stock2")
        logger.info(f"Initialized environment with {self.n} time steps, beta={self.beta}")
        self.reset()

    def reset(self):
        self.i = WINDOW
        self.cash = self.cap0
        self.shares1 = 0.0
        self.shares2 = 0.0
        self.pos = 0
        self.entry_z = None
        self.entry_i = None
        self.entry_amt = 0
        self.entry_cap = None
        self.last_exit_i = -1000
        self.cap_history = [self.cap0]
        self.trades = []
        return self._get_state()

    def get_current_cap(self):
        return self.cash + self.shares1 * self.prices1[self.i] + self.shares2 * self.prices2[self.i]

    def step(self, action):
        done = False
        reward = -0.001
        z_cur = self.z[self.i]
        z_prev = self.z[self.i - 1] if self.i > 0 else 0
        cooldown = (self.i - self.last_exit_i) < self.cooldown_steps
        holding = self.i - self.entry_i if self.entry_i is not None else 0

        current_p1 = self.prices1[self.i]
        current_p2 = self.prices2[self.i]

        if self.pos == 0 and not cooldown:
            if action in [1, 2, 3, 4]:
                can_enter = abs(z_cur) >= self.entry_zscore or (z_cur * z_prev < 0)
                if can_enter:
                    self.pos = 1 if action in [1, 2] else -1
                    dollar = self.cap0
                    self.entry_amt = dollar
                    if self.pos == 1:
                        # short spread: short stock1, long stock2
                        shares2 = dollar / current_p2
                        shares1 = -self.beta * shares2
                    else:
                        # long spread: long stock1, short stock2
                        shares1 = dollar / current_p1
                        shares2 = -self.beta * shares1
                    self.shares1 = shares1
                    self.shares2 = shares2
                    dollar_long = abs(self.shares1 if self.shares1 > 0 else self.shares2) * (current_p1 if self.shares1 > 0 else current_p2)
                    dollar_short = abs(self.shares1 if self.shares1 < 0 else self.shares2) * (current_p1 if self.shares1 < 0 else current_p2)
                    tc_cost = self.tc * (dollar_long + dollar_short)
                    self.cash -= tc_cost
                    self.cash -= (self.shares1 * current_p1 + self.shares2 * current_p2)
                    self.entry_cap = self.get_current_cap()
                    self.entry_z = z_cur
                    self.entry_i = self.i
                    self.trades.append(('entry', self.i, self.pos, z_cur, dollar_long + dollar_short))
                    reward += abs(z_cur) * 0.5 + 0.05

        elif self.pos != 0 and action == 5 and holding >= self.min_holding:
            closer_to_mean = abs(z_cur) < abs(self.entry_z)
            # Exit
            delta_shares1 = -self.shares1
            delta_shares2 = -self.shares2
            dollar_long = abs(delta_shares1 if delta_shares1 > 0 else delta_shares2) * (current_p1 if delta_shares1 > 0 else current_p2)
            dollar_short = abs(delta_shares1 if delta_shares1 < 0 else delta_shares2) * (current_p1 if delta_shares1 < 0 else current_p2)
            tc_cost = self.tc * (dollar_long + dollar_short)
            self.cash -= tc_cost
            self.cash -= (delta_shares1 * current_p1 + delta_shares2 * current_p2)
            self.shares1 = 0.0
            self.shares2 = 0.0
            pnl = self.cash - self.entry_cap
            profit_ratio = pnl / self.entry_amt
            if closer_to_mean and profit_ratio > 0.0005:
                reward += 50 * profit_ratio + 0.3 * (1 - abs(z_cur))
                self.trades.append(('exit', self.i, self.pos, z_cur, pnl))
            else:
                reward -= 0.2
                self.trades.append(('exit', self.i, self.pos, z_cur, pnl))
            self.pos = 0
            self.entry_amt = 0
            self.entry_z = None
            self.entry_i = None
            self.last_exit_i = self.i

        if self.pos != 0:
            too_far = abs(z_cur - self.entry_z) > 7
            too_long = holding > 60
            if too_far or too_long:
                # Forced exit
                delta_shares1 = -self.shares1
                delta_shares2 = -self.shares2
                dollar_long = abs(delta_shares1 if delta_shares1 > 0 else delta_shares2) * (current_p1 if delta_shares1 > 0 else current_p2)
                dollar_short = abs(delta_shares1 if delta_shares1 < 0 else self.shares2) * (current_p1 if delta_shares1 < 0 else current_p2)
                tc_cost = self.tc * (dollar_long + dollar_short)
                self.cash -= tc_cost
                self.cash -= (delta_shares1 * current_p1 + delta_shares2 * current_p2)
                self.shares1 = 0.0
                self.shares2 = 0.0
                pnl = self.cash - self.entry_cap
                self.trades.append(('forced_exit', self.i, self.pos, z_cur, pnl))
                reward -= 1.0
                self.pos = 0
                self.entry_amt = 0
                self.last_exit_i = self.i

        self.cap = self.get_current_cap()
        if self.cap > self.cap0:
            reward += 0.05 * (self.cap - self.cap0) / self.cap0

        if self.cap0 - self.cap > self.max_dd or self.i >= self.n - 1:
            done = True

        self.cap_history.append(self.cap)
        self.i += 1
        return self._get_state(), reward, done, {}

    def _get_state(self):
        zvals = self.z[self.i - WINDOW:self.i]
        zidx = [min(Z_BINS - 1, max(0, int((z + 5) / 10 * Z_BINS))) for z in zvals]
        hist = sum(zidx[j] * (Z_BINS ** j) for j in range(WINDOW))
        pos_idx = {0: 0, 1: 1, -1: 2}[self.pos]
        return hist * HOLD_POS + pos_idx

def train_single_q(env, N, alpha, gamma, eps, decay, optimistic_value=15.0):
    logger.info(f"Starting Q-table training with N={N} episodes")
    start_time = time.time()
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
        if (ep + 1) % 100 == 0:
            logger.info(f"Completed episode {ep + 1}/{N}, total reward: {total_reward:.2f}, time elapsed: {time.time() - start_time:.2f}s")
    logger.info(f"Finished training, total time: {time.time() - start_time:.2f}s")
    return Q, episode_rewards

def compute_total_return(hist):
    return ((hist[-1] - hist[0]) / hist[0]) * 100

def compute_sharpe_ratio(hist, risk_free_rate=0.0679, periods_per_year=1500):
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
logger.info(f"Selected pairs: {selected_pairs}")

generated_files = []

with open("evaluation_results.txt", "w") as f:
    for pair in selected_pairs:
        logger.info(f"Evaluating pair: {pair}")
        print(f"\nEvaluating {pair}", file=f)
        stock1, stock2 = pair.split("_")
        # Validate data existence
        if stock1 not in price_train.columns or stock2 not in price_train.columns:
            logger.warning(f"Stock {stock1} or {stock2} not found in price_train")
            print(f"Error: Stock {stock1} or {stock2} not found in price_train", file=f)
            continue
        if stock1 not in price_eval.columns or stock2 not in price_eval.columns:
            logger.warning(f"Stock {stock1} or {stock2} not found in price_eval")
            print(f"Error: Stock {stock1} or {stock2} not found in price_eval", file=f)
            continue
        if pair not in z_train.columns or pair not in z_eval.columns:
            logger.warning(f"Pair {pair} not found in z_train or z_eval")
            print(f"Error: Pair {pair} not found in z_train or z_eval", file=f)
            continue

        # Compute beta from training data
        logger.info(f"Computing beta for {pair}")
        try:
            log_p1 = np.log(price_train[stock1].dropna())
            log_p2 = np.log(price_train[stock2].dropna())
            X = sm.add_constant(log_p2)
            model = sm.OLS(log_p1, X).fit()
            beta = model.params.iloc[1]
            logger.info(f"Beta for {pair}: {beta}")
        except Exception as e:
            logger.error(f"Error computing beta for {pair}: {e}")
            print(f"Error computing beta for {pair}: {e}", file=f)
            continue

        results = []
        all_capitals_runs = []
        pair_files = []

        for run in range(10):
            logger.info(f"Starting run {run + 1}/10 for {pair}")
            start_run_time = time.time()
            Q_ensemble = []
            for ens in range(ENSEMBLE_SIZE):
                logger.info(f"Training ensemble {ens + 1}/{ENSEMBLE_SIZE} for run {run + 1}")
                rng = np.random.default_rng()
                env = PairsTradingEnv(z_scores=z_train[pair].values, 
                                     prices_stock1=price_train[stock1].values, 
                                     prices_stock2=price_train[stock2].values, 
                                     beta=beta, 
                                     entry_zscore=entry_zscore)
                Q, episode_rewards = train_single_q(
                    env, N=1000, alpha=alpha, gamma=gamma, eps=eps, decay=decay,
                    optimistic_value=OPTIMISTIC_Q
                )
                Q_ensemble.append(Q)

            Qs_stacked = np.array(Q_ensemble)
            Q_avg = Qs_stacked.mean(axis=0)
            logger.info(f"Computed average Q-table for run {run + 1}")

            # Plot episode rewards
            reward_plot = f"{pair}_run{run + 1}_reward_curve.png"
            plt.figure(figsize=(10, 4))
            plt.plot(episode_rewards, label='Episode Reward (last ensemble)')
            plt.title(f'{pair} – Run {run + 1} Episode Reward (Ensemble Q)')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True)
            plt.tight_layout()
            plt.legend()
            plt.savefig(reward_plot, dpi=200)
            plt.close()
            logger.info(f"Saved reward plot: {reward_plot}")
            pair_files.append(reward_plot)

            # Evaluation
            logger.info(f"Starting evaluation for run {run + 1}")
            env_eval = PairsTradingEnv(z_scores=z_eval[pair].values, 
                                      prices_stock1=price_eval[stock1].values, 
                                      prices_stock2=price_eval[stock2].values, 
                                      beta=beta, 
                                      entry_zscore=entry_zscore)
            s = env_eval.reset()
            done = False
            while not done:
                a = np.argmax(Q_avg[s])
                s, _, done, _ = env_eval.step(a)

            trades = env_eval.trades
            hist = env_eval.cap_history
            logger.info(f"Evaluation complete for run {run + 1}, trades: {len(trades)}, capital history length: {len(hist)}")

            # Plot capital history
            capital_plot = f"{pair}_run{run + 1}_capital.png"
            plt.figure(figsize=(10, 4))
            plt.plot(hist, label=f'Run {run + 1} Capital')
            plt.title(f'{pair} – Capital Over Time (Run {run + 1})')
            plt.xlabel('Time Steps')
            plt.ylabel('Capital')
            plt.grid(True)
            plt.tight_layout()
            plt.legend()
            plt.savefig(capital_plot, dpi=200)
            plt.close()
            logger.info(f"Saved capital plot: {capital_plot}")
            pair_files.append(capital_plot)

            wins = sum(1 for t in trades if t[0] in ('exit', 'forced_exit') and t[4] > 0)
            losses = sum(1 for t in trades if t[0] in ('exit', 'forced_exit') and t[4] <= 0)
            total = wins + losses if wins + losses > 0 else 1
            ret = compute_total_return(hist)
            sharpe = compute_sharpe_ratio(hist)
            max_dd = compute_max_drawdown(hist)
            total_trades = total

            logger.info(f"Run {run + 1} metrics: Return={ret:.2f}%, Sharpe={sharpe:.2f}, Win Rate={100*wins/total:.2f}%, Trades={total_trades}, Max DD={max_dd:.2%}")
            print(f"Run {run + 1}: Return: {ret:.2f}%, Sharpe: {sharpe:.2f}, Win Rate: {100*wins/total:.2f}%, Total Trades: {total_trades}, Max Drawdown: {max_dd:.2%}", file=f)
            f.flush()

            results.append((ret, sharpe, max_dd, total_trades))
            all_capitals_runs.append(hist)
            logger.info(f"Run {run + 1} completed in {time.time() - start_run_time:.2f}s")

        min_len = min(len(c) for c in all_capitals_runs)
        trimmed_runs = np.array([c[:min_len] for c in all_capitals_runs])
        all_capitals_file = f"{pair}_all_capitals.npy"
        np.save(all_capitals_file, trimmed_runs)
        logger.info(f"Saved all capitals: {all_capitals_file}")
        pair_files.append(all_capitals_file)

        returns, sharpes, max_dds, total_trades_list = zip(*results)
        logger.info(f"{pair} Summary: Avg Return={np.mean(returns):.2f}% ± {np.std(returns):.2f}%, Avg Sharpe={np.mean(sharpes):.2f} ± {np.std(sharpes):.2f}, Avg Max DD={np.mean(max_dds):.2%} ± {np.std(max_dds):.2%}, Avg Trades={np.mean(total_trades_list):.2f} ± {np.std(total_trades_list):.2f}")
        print(f"\n>> {pair} Summary over {len(returns)} runs:", file=f)
        print(f"Average Return: {np.mean(returns):.2f}% ± {np.std(returns):.2f}%", file=f)
        print(f"Average Sharpe: {np.mean(sharpes):.2f} ± {np.std(sharpes):.2f}", file=f)
        print(f"Average Max Drawdown: {np.mean(max_dds):.2%} ± {np.std(max_dds):.2%}", file=f)
        print(f"Average Total Trades: {np.mean(total_trades_list):.2f} ± {np.std(total_trades_list):.2f}", file=f)
        f.flush()

        avg_capital = trimmed_runs.mean(axis=0)
        std_capital = trimmed_runs.std(axis=0)
        avg_capital_file = f"{pair}_capitals.npy"
        np.save(avg_capital_file, avg_capital)
        logger.info(f"Saved average capital: {avg_capital_file}")
        pair_files.append(avg_capital_file)

        # Plot average capital
        avg_capital_plot = f"{pair}_capital_plot.png"
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
        plt.savefig(avg_capital_plot, dpi=300)
        plt.close()
        logger.info(f"Saved average capital plot: {avg_capital_plot}")
        pair_files.append(avg_capital_plot)

        logger.info(f"Files generated for {pair}: {pair_files}")
        generated_files.extend(pair_files)

# Combined Total Capital Plot
logger.info("Generating combined capital plot")
all_capitals_runs = []
for pair in selected_pairs:
    try:
        capitals_file = f"{pair}_all_capitals.npy"
        capitals = np.load(capitals_file)
        logger.info(f"Loaded {capitals_file} with shape {capitals.shape}")
        all_capitals_runs.append(capitals)
    except FileNotFoundError:
        logger.warning(f"Skipping {pair}, no capital data found: {capitals_file}")
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
    stacked = np.stack(padded)
    total_capital_runs = np.sum(stacked, axis=0)
    mean_total_capital = total_capital_runs.mean(axis=0)
    std_total_capital = total_capital_runs.std(axis=0)

    initial_capital = mean_total_capital[0]
    final_capital = mean_total_capital[-1]
    total_return = (final_capital - initial_capital) / initial_capital * 100

    risk_free_rate = 0.0679
    periods_per_year = 1500
    returns = np.diff(mean_total_capital) / mean_total_capital[:-1]
    excess_returns = returns - (risk_free_rate / periods_per_year)
    if returns.std() == 0:
        portfolio_sharpe = 0.0
    else:
        portfolio_sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(periods_per_year)
    portfolio_max_dd = compute_max_drawdown(mean_total_capital)

    logger.info(f"Portfolio Metrics: Max Drawdown={portfolio_max_dd:.2%}, Total Return={final_capital:.2f}, Total Return (%): {total_return:.2f}%, Sharpe={portfolio_sharpe:.2f}")
    print(f"Portfolio Max Drawdown: {portfolio_max_dd:.2%}")
    print(f"Portfolio Total Return: {final_capital:.2f}")
    print(f"Portfolio Total Return (%): {total_return:.2f}%")
    print(f"Portfolio Sharpe Ratio: {portfolio_sharpe:.2f}")

    combined_plot = "total_combined_capital_plot.png"
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
    plt.savefig(combined_plot, dpi=300)
    plt.close()
    logger.info(f"Saved combined capital plot: {combined_plot}")
    generated_files.append(combined_plot)

logger.info(f"Script completed. Generated files: {generated_files}")
logger.info("Output files include:")
logger.info(f"- evaluation_results.txt: Evaluation metrics for all pairs")
logger.info(f"- training_log.txt: Detailed execution log")
for file in generated_files:
    logger.info(f"- {file}")