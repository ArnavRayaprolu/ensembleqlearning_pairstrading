import pandas as pd
import numpy as np
import optuna
import gym
from gym import spaces
import random
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message="The reported value is ignored because this `step`")


# === Load Data ===
z_train = pd.read_csv("zscore_spreads.csv", index_col=0, parse_dates=True)
z_eval = pd.read_csv("zscore_spreads2.csv", index_col=0, parse_dates=True)

# === Constants ===
WINDOW = 3
Z_BINS = 15
HOLD_POS = 3
STATE_SIZE = (Z_BINS ** WINDOW) * HOLD_POS

# === Environment and Utilities ===
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

def train_single_q(env, trial, N=8000, alpha=0.02, gamma=0.97, eps=0.8, decay=0.995, prune_interval=500):
    Q = np.zeros((STATE_SIZE, env.action_space.n))
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
        # Prune every prune_interval episodes
        if trial is not None and (ep + 1) % prune_interval == 0:
            intermediate_value = np.mean(episode_rewards[-prune_interval:])
            trial.report(intermediate_value, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return Q

def compute_total_return(hist):
    return ((hist[-1] - hist[0]) / hist[0]) * 100

def compute_sharpe_ratio(hist, risk_free_rate=0.0679, periods_per_year=855):
    returns = np.diff(hist) / hist[:-1]
    excess_returns = returns - (risk_free_rate / periods_per_year)
    if returns.std() == 0:
        return 0.0
    sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(periods_per_year)
    return sharpe

# === Bayesian Optimization for all pairs ===
selected_pairs = [
    "BPCL_CIPLA",
]

def objective(trial):
    N = 8000
    # alpha: 0.01 to 0.05, step 0.005
    alpha = trial.suggest_categorical('alpha', [round(x, 3) for x in np.arange(0.01, 0.051, 0.005)])
    # gamma: 0.90 to 0.99, step 0.01
    gamma = trial.suggest_categorical('gamma', [round(x, 2) for x in np.arange(0.90, 0.991, 0.01)])
    # epsilon: 0.6 to 1.0, step 0.05
    eps = trial.suggest_categorical('eps', [round(x, 2) for x in np.arange(0.6, 1.01, 0.05)])
    # decay: 0.980 to 0.999, step 0.001
    decay = trial.suggest_categorical('decay', [round(x, 3) for x in np.arange(0.980, 0.9991, 0.001)])
    # entry_zscore: 0.01 to 2.0, step 0.01
    entry_zscore = trial.suggest_categorical('entry_zscore', [round(x, 2) for x in np.arange(0.01, 1.01, 0.05)])

    returns, sharpes = [], []
    n_runs = 15  # Number of runs per pair per trial (increase for more robust results)
    for pair in selected_pairs:
        z_train_series = z_train[pair].values
        z_eval_series = z_eval[pair].values
        run_returns, run_sharpes = [], []
        with tqdm(total=n_runs, desc=f"Trial {trial.number} {pair}", leave=False, ncols=70) as inner_pbar:
            for _ in range(n_runs):
                env_train = PairsTradingEnv(z_train_series, entry_zscore=entry_zscore)
                Q = train_single_q(env_train, trial, N=N, alpha=alpha, gamma=gamma, eps=eps, decay=decay)
                env_eval = PairsTradingEnv(z_eval_series, entry_zscore=entry_zscore)
                s = env_eval.reset()
                done = False
                while not done:
                    a = np.argmax(Q[s])
                    s, _, done, _ = env_eval.step(a)
                hist = env_eval.cap_history
                run_returns.append(compute_total_return(hist))
                run_sharpes.append(compute_sharpe_ratio(hist))
                inner_pbar.update(1)
        returns.append(np.mean(run_returns))
        sharpes.append(np.mean(run_sharpes))
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    score = mean_return - 3 * std_return  # Weighted score
    return score

if __name__ == "__main__":
    n_trials = 60 # Set your number of trials here

    # Use Optuna median pruner for early stopping
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2, interval_steps=1)
    study = optuna.create_study(direction="maximize", pruner=pruner)

    # Set up tqdm progress bar for trials
    pbar = tqdm(total=n_trials, desc="Optuna Trials", ncols=80)

    def tqdm_callback(study, trial):
        pbar.update(1)

    study.optimize(objective, n_trials=n_trials, callbacks=[tqdm_callback])

    pbar.close()

    print("Best parameters:", study.best_params)
    print("Best score:", study.best_value)
    # Save best parameters for later use
    with open("best_hyperparameters.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
