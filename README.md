# A/B Testing Homework - Hripsime Soghomonyan


## Project Structure

- `Bandit.py`: Contains implementations of both algorithms and the experiment logic.
- `csv/`: Stores the reward logs from each algorithm as CSV files.
- `png/`: Contains plots showing learning progress, rewards, and regrets.
- `bonus_suggestion.txt`: My suggestion for a better or extended implementation.

## 📊 Visualizations

The following plots are generated:
- Reward per trial (linear & log scale)
- Cumulative reward comparison (EG vs. TS)
- Cumulative regret comparison
- Individual reward trajectories

## How to Run

```bash
python Bandit.py
