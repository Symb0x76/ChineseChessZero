import cchess
import random
import pickle
import time
import argparse
import numpy as np
from game import Game
from collections import deque
from net import PolicyValueNet
from tools import recovery_state_mcts_prob

# from torch.utils.data import DataLoader, Dataset
from parameters import (
    PLAYOUT,
    C_PUCT,
    BATCH_SIZE,
    EPOCHS,
    KL_TARG,
    BUFFER_SIZE,
    GAME_BATCH_NUM,
    UPDATE_INTERVAL,
    DATA_PATH,
    MODEL_PATH,
    CHECK_FREQ,
)


class TrainPipeline:
    def __init__(self, init_model=None):
        # 训练参数
        self.board = cchess.Board()
        self.game = Game(self.board)
        self.n_playout = PLAYOUT  # 每次移动的模拟次数
        self.c_puct = C_PUCT
        self.learning_rate = 1e-3  # 学习率
        self.lr_multiplier = 1  # 基于KL自适应的调整学习率
        self.temp = 1.0  # 温度
        self.batch_size = BATCH_SIZE  # 批次大小
        self.epochs = EPOCHS  # 每次训练的轮数
        self.kl_targ = KL_TARG  # kl散度控制
        self.check_freq = CHECK_FREQ  # 保存模型的频率
        self.game_batch_num = GAME_BATCH_NUM  # 每次训练的游戏数量
        # self.best_win_ratio = 0.0
        # self.pure_mcts_playout_num = 500
        self.buffer_size = BUFFER_SIZE  # 经验池大小
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.train_iters = 0  # 训练迭代计数
        self.data_iters = 0  # 数据收集迭代计数
        if init_model:
            try:
                self.policy_value_net = PolicyValueNet(model_file=init_model)
                print(f"[{time.strftime('%H:%M:%S')}] 已加载模型: {init_model}")
            except Exception as e:
                # 从零开始训练
                print(
                    f"[{time.strftime('%H:%M:%S')}] 模型 {init_model} 加载失败: {str(e)}，从零开始训练"
                )
                self.policy_value_net = PolicyValueNet()
        else:
            print(f"[{time.strftime('%H:%M:%S')}] 从零开始训练")
            self.policy_value_net = PolicyValueNet()

    def policy_update(self):
        """更新策略价值网络"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        # print(mini_batch[0][1],mini_batch[1][1])
        mini_batch = [recovery_state_mcts_prob(data) for data in mini_batch]
        state_batch = [data[0] for data in mini_batch]
        state_batch = np.array(state_batch).astype("float32")

        mcts_probs_batch = [data[1] for data in mini_batch]
        mcts_probs_batch = np.array(mcts_probs_batch).astype("float32")

        winner_batch = [data[2] for data in mini_batch]
        winner_batch = np.array(winner_batch).astype("float32")

        # 旧的策略，旧的价值函数
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learning_rate * self.lr_multiplier,
            )
            # 新的策略，新的价值函数
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)

            kl = np.mean(
                np.sum(
                    old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1,
                )
            )
            if kl > self.kl_targ * 4:  # 如果KL散度很差，则提前终止
                break

        # 自适应调整学习率
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        # print(old_v.flatten(),new_v.flatten())
        explained_var_old = 1 - np.var(
            np.array(winner_batch) - old_v.flatten()
        ) / np.var(np.array(winner_batch))
        explained_var_new = 1 - np.var(
            np.array(winner_batch) - new_v.flatten()
        ) / np.var(np.array(winner_batch))

        print(
            (
                f"[{time.strftime('%H:%M:%S')}] kl:{kl:.6f},"
                f"lr_multiplier:{self.lr_multiplier:.6f},"
                f"loss:{loss:.6f},"
                f"entropy:{entropy:.6f},"
                f"explained_var_old:{explained_var_old:.6f},"
                f"explained_var_new:{explained_var_new:.6f}"
            )
        )
        return loss, entropy

    def save_train_state(self):
        """保存训练状态"""
        train_state = {
            "train_iters": self.train_iters,
            "data_iters": self.data_iters,
            "lr_multiplier": self.lr_multiplier,
        }
        with open("train_state.pkl", "wb") as f:
            pickle.dump(train_state, f)

    def load_train_state(self):
        """加载训练状态"""
        try:
            with open("train_state.pkl", "rb") as f:
                train_state = pickle.load(f)
                self.train_iters = train_state["train_iters"]
                self.data_iters = train_state["data_iters"]
                self.lr_multiplier = train_state["lr_multiplier"]
            print(
                f"[{time.strftime('%H:%M:%S')}] 已加载训练状态: 训练迭代 {self.train_iters}, 数据迭代 {self.data_iters}"
            )
            return True
        except:
            print(f"[{time.strftime('%H:%M:%S')}] 无法加载训练状态，从头开始训练")
            return False

    def run(self):
        """开始训练"""
        try:
            # 尝试加载之前的训练状态
            self.load_train_state()

            while self.train_iters < self.game_batch_num:
                # 加载最新数据
                new_data = False
                while not new_data:
                    try:
                        with open(DATA_PATH, "rb") as data_dict:
                            data_file = pickle.load(data_dict)
                            self.data_buffer = data_file["data_buffer"]
                            current_data_iters = data_file["iters"]

                            # 检查是否有新数据
                            if current_data_iters > self.data_iters:
                                self.data_iters = current_data_iters
                                new_data = True
                                print(
                                    f"[{time.strftime('%H:%M:%S')}] 已载入新数据，数据迭代: {self.data_iters}, 缓冲区大小: {len(self.data_buffer)}"
                                )
                            else:
                                print(
                                    f"[{time.strftime('%H:%M:%S')}] 等待新数据... 当前数据迭代: {self.data_iters}, 训练迭代: {self.train_iters}"
                                )
                                time.sleep(UPDATE_INTERVAL)
                    except Exception as e:
                        print(
                            f"[{time.strftime('%H:%M:%S')}] 加载数据失败: {str(e)}，10秒后重试"
                        )
                        time.sleep(10)

                # 执行训练
                print(
                    f"[{time.strftime('%H:%M:%S')}] 训练迭代 {self.train_iters}, 数据迭代 {self.data_iters}"
                )
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()

                    # 保存模型和训练状态
                    self.policy_value_net.save_model(MODEL_PATH)
                    self.train_iters += 1
                    self.save_train_state()

                    # 定期保存检查点
                    if self.train_iters % self.check_freq == 0:
                        # win_ratio = self.policy_evaluate()
                        # print("current self-play batch: {},win_ratio: {}".format(i + 1, win_ratio))
                        # self.policy_value_net.save_model('./current_policy.model')
                        # if win_ratio > self.best_win_ratio:
                        #     print(f"[{time.strftime('%H:%M:%S')}] New best policy!!!!!!!!")
                        #     self.best_win_ratio = win_ratio
                        #     # update the best_policy
                        #     self.policy_value_net.save_model('./best_policy.model')
                        #     if (self.best_win_ratio == 1.0 and
                        #             self.pure_mcts_playout_num < 5000):
                        #         self.pure_mcts_playout_num += 1000
                        #         self.best_win_ratio = 0.0
                        print(
                            f"[{time.strftime('%H:%M:%S')}] 保存检查点: 训练迭代 {self.train_iters}"
                        )
                        self.policy_value_net.save_model(
                            f"models/current_policy_batch{self.train_iters}.pkl"
                        )
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] 数据不足，等待更多数据")
                    time.sleep(UPDATE_INTERVAL)
        except KeyboardInterrupt:
            print(f"\r[{time.strftime('%H:%M:%S')}] 保存训练状态并退出")
            self.save_train_state()


if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="收集中国象棋自对弈数据")
    parser.add_argument(
        "--model", type=str, default="current_policy.pkl", help="初始化模型路径"
    )
    args = parser.parse_args()
    training_pipeline = TrainPipeline(init_model=args.model)
    training_pipeline.run()
