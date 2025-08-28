import json
import glob
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def find_latest_history(prefix='mlp', folder='.'):
    pattern = os.path.join(folder, f'*{prefix}*_training_history.json')
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def plot_convergence(history_path: str, out_path: str = 'figures/convergence_f1.png'):
    with open(history_path, 'r') as f:
        data = json.load(f)

    if 'mlp' not in data:
        raise RuntimeError('No mlp history found in file')

    mlp = data['mlp']
    eval_rewards = mlp.get('eval_rewards', [])
    rewards = mlp.get('rewards', [])
    eval_f1s = mlp.get('eval_f1s', [])
    eval_indices = mlp.get('eval_episode_indices', [])

    # Prefer plotting F1s if available
    if eval_f1s:
        y = np.array(eval_f1s)
        ylabel = 'F1'
        title = 'Convergence: Eval F1 vs Episodes'
        if eval_indices:
            x = np.array(eval_indices)
        else:
            total_eps = len(rewards) if rewards else (len(eval_f1s) * 1)
            x = np.linspace(1, total_eps, num=len(eval_f1s), dtype=int)
    else:
        if not eval_rewards:
            raise RuntimeError('No eval_rewards or eval_f1s found to plot')
        y = np.array(eval_rewards)
        ylabel = 'Mean Eval Reward'
        title = 'Convergence: Eval Reward vs Episodes'
        total_eps = len(rewards) if rewards else (len(eval_rewards) * 1)
        x = np.linspace(1, total_eps, num=len(eval_rewards), dtype=int)

    Path(os.path.dirname(out_path) or '.').mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y, marker='o', linestyle='-')
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)

    best_idx = int(np.argmax(y))
    best_val = float(y[best_idx])
    best_ep = int(x[best_idx])
    plt.scatter([best_ep], [best_val], color='red')
    plt.annotate(f'Best: {best_val:.2f} @ ep {best_ep}', xy=(best_ep, best_val), xytext=(best_ep, best_val + 2), arrowprops=dict(arrowstyle='->'))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f'Plot saved: {out_path}')
    print(f'Best eval value: {best_val:.3f} at episode {best_ep}')


if __name__ == '__main__':
    latest = find_latest_history(prefix='mlp', folder='.')
    if latest is None:
        print('No mlp training history file found in workspace.')
        raise SystemExit(1)
    out = 'figures/convergence_f1.png'
    plot_convergence(latest, out)
    