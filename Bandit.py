from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Bandit(ABC):

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self, arm, reward):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        pass


class Visualization():

    def __init__(self, eg_csv='EpsilonGreedy_rewards.csv', ts_csv='ThompsonSampling_rewards.csv'):
        self.eg_df = pd.read_csv(eg_csv)
        self.ts_df = pd.read_csv(ts_csv)

    def plot1(self):
        # linear
        plt.figure()
        plt.plot(self.eg_df['Reward'], alpha=0.6, label='Epsilon-Greedy')
        plt.plot(self.ts_df['Reward'], alpha=0.6, label='Thompson Sampling')
        plt.xlabel("Trials")
        plt.ylabel("Reward")
        plt.title("Reward per Trial (Linear Scale)")
        plt.legend()
        plt.grid(True)
        plt.savefig("Reward_Per_Trial_Linear.png")
        plt.close()

        # log scale
        plt.figure()
        plt.plot(self.eg_df['Reward'], alpha=0.6, label='Epsilon-Greedy')
        plt.plot(self.ts_df['Reward'], alpha=0.6, label='Thompson Sampling')
        plt.xlabel("Trials")
        plt.ylabel("Reward")
        plt.title("Reward per Trial (Log Scale)")
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.savefig("Reward_Per_Trial_Log.png")
        plt.close()

    def plot2(self):
        # Cumulative rewards
        eg_cum_reward = self.eg_df['Reward'].cumsum()
        ts_cum_reward = self.ts_df['Reward'].cumsum()

        plt.figure()
        plt.plot(eg_cum_reward, label='Epsilon-Greedy')
        plt.plot(ts_cum_reward, label='Thompson Sampling')
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Rewards Comparison")
        plt.legend()
        plt.grid(True)
        plt.savefig("Cumulative_Reward_Comparison.png")
        plt.close()

        # Cumulative regret
        eg_regret = max(Bandit_Reward) * len(self.eg_df) - eg_cum_reward
        ts_regret = max(Bandit_Reward) * len(self.ts_df) - ts_cum_reward

        plt.figure()
        plt.plot(eg_regret, label='Epsilon-Greedy')
        plt.plot(ts_regret, label='Thompson Sampling')
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Regret")
        plt.title("Cumulative Regret Comparison")
        plt.legend()
        plt.grid(True)
        plt.savefig("Cumulative_Regret_Comparison.png")
        plt.close()


class EpsilonGreedy(Bandit):
    def __init__(self, rewards, epsilon=1.0):
        self.rewards = rewards
        self.epsilon = epsilon
        self.counts = np.zeros(len(rewards))
        self.values = np.zeros(len(rewards))
        self.total_reward = 0
        self.reward_history = []
        self.choice_history = []
        self.regret_history = []

    def __repr__(self):
        return f"EpsilonGreedy(epsilon={self.epsilon})"

    def pull(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.rewards))
        else:
            return np.argmax(self.values)

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward
        self.total_reward += reward
        self.reward_history.append(reward)
        self.choice_history.append(arm)
        regret = max(self.rewards) - self.rewards[arm]
        self.regret_history.append(regret)

    def experiment(self, trials=20000):
        for t in range(1, trials + 1):
            self.epsilon = 1 / t
            arm = self.pull()
            reward = np.random.normal(loc=self.rewards[arm], scale=1.0)
            self.update(arm, reward)

    def report(self):
        df = pd.DataFrame({
            "Bandit": self.choice_history,
            "Reward": self.reward_history,
            "Algorithm": ["EpsilonGreedy"] * len(self.reward_history)
        })
        df.to_csv("EpsilonGreedy_rewards.csv", index=False)

        avg_reward = np.mean(self.reward_history)
        avg_regret = np.mean(self.regret_history)

        logger.info(f"[EpsilonGreedy] Average Reward: {avg_reward:.4f}")
        logger.info(f"[EpsilonGreedy] Cumulative Reward: {self.total_reward:.4f}")
        logger.info(f"[EpsilonGreedy] Cumulative Regret: {sum(self.regret_history):.4f}")

        self.plot_learning()

    def plot_learning(self):
        plt.figure()
        plt.plot(np.cumsum(self.reward_history))
        plt.title("Cumulative Reward - Epsilon Greedy")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.grid(True)
        plt.savefig("EpsilonGreedy_cumulative_reward.png")
        plt.close()


class ThompsonSampling(Bandit):
    def __init__(self, rewards):
        self.rewards = rewards
        self.means = np.zeros(len(rewards))
        self.lambda_prior = np.ones(len(rewards))
        self.tau = 1.0
        self.total_reward = 0
        self.reward_history = []
        self.choice_history = []
        self.regret_history = []

    def __repr__(self):
        return "ThompsonSampling()"

    def pull(self):
        samples = np.random.normal(self.means, 1 / np.sqrt(self.lambda_prior))
        return np.argmax(samples)

    def update(self, arm, reward):
        self.lambda_prior[arm] += self.tau
        self.means[arm] = (self.means[arm] * (self.lambda_prior[arm] - self.tau) + reward * self.tau) / self.lambda_prior[arm]
        self.total_reward += reward
        self.reward_history.append(reward)
        self.choice_history.append(arm)
        regret = max(self.rewards) - self.rewards[arm]
        self.regret_history.append(regret)

    def experiment(self, trials=20000):
        for _ in range(trials):
            arm = self.pull()
            reward = np.random.normal(loc=self.rewards[arm], scale=1.0)
            self.update(arm, reward)

    def report(self):
        df = pd.DataFrame({
            "Bandit": self.choice_history,
            "Reward": self.reward_history,
            "Algorithm": ["ThompsonSampling"] * len(self.reward_history)
        })
        df.to_csv("ThompsonSampling_rewards.csv", index=False)

        avg_reward = np.mean(self.reward_history)
        avg_regret = np.mean(self.regret_history)

        logger.info(f"[ThompsonSampling] Average Reward: {avg_reward:.4f}")
        logger.info(f"[ThompsonSampling] Cumulative Reward: {self.total_reward:.4f}")
        logger.info(f"[ThompsonSampling] Cumulative Regret: {sum(self.regret_history):.4f}")

        self.plot_learning()

    def plot_learning(self):
        plt.figure()
        plt.plot(np.cumsum(self.reward_history))
        plt.title("Cumulative Reward - Thompson Sampling")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.grid(True)
        plt.savefig("ThompsonSampling_cumulative_reward.png")
        plt.close()


def comparison():
    viz = Visualization()
    viz.plot1()
    viz.plot2()
    logger.info("Comparison plots saved: linear, log, cumulative reward, and regret.")


if __name__=='__main__':
   
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")


if __name__ == '__main__':
    Bandit_Reward = [1, 2, 3, 4]

    eg = EpsilonGreedy(Bandit_Reward)
    eg.experiment()
    eg.report()

    ts = ThompsonSampling(Bandit_Reward)
    ts.experiment()
    ts.report()

    # viz = Visualization()
    # viz.plot1()
    # viz.plot2()

    comparison()


