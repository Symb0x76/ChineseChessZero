import cchess
import random
import pickle
import time
import numpy as np
from collect import Game
from collections import deque
from net import PolicyValueNet
from tools import zip_array
from parameters import (
    PLAYOUT,
    C_PUCT,
    BATCH_SIZE,
    EPOCHS,
    KL_TARG,
    BUFFER_SIZE,
    GAME_BATCH_NUM,
    UPDATE_INTERVAL,
    DATA_BUFFER_PATH,
    MODEL_PATH,
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
        self.check_freq = 100  # 保存模型的频率
        self.game_batch_num = GAME_BATCH_NUM  # 每次训练的游戏数量
        self.best_win_ratio = 0.0
        # self.pure_mcts_playout_num = 500
        self.buffer_size = BUFFER_SIZE  # 经验池大小
        self.data_buffer = deque(maxlen=self.buffer_size)
        if init_model:
            try:
                self.policy_value_net = PolicyValueNet(model_file=init_model)
                print("已加载上次最终模型")
            except:
                # 从零开始训练
                print("模型路径不存在，从零开始训练")
                self.policy_value_net = PolicyValueNet()
        else:
            print("从零开始训练")
            self.policy_value_net = PolicyValueNet()

    def policy_updata(self):
        """更新策略价值网络"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        # print(mini_batch[0][1],mini_batch[1][1])
        mini_batch = [zip_array.recovery_state_mcts_prob(data) for data in mini_batch]
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
                self.learn_rate * self.lr_multiplier,
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
                "kl:{:.5f},"
                "lr_multiplier:{:.3f},"
                "loss:{},"
                "entropy:{},"
                "explained_var_old:{:.9f},"
                "explained_var_new:{:.9f}"
            ).format(
                kl,
                self.lr_multiplier,
                loss,
                entropy,
                explained_var_old,
                explained_var_new,
            )
        )
        return loss, entropy

    def run(self):
        """开始训练"""
        try:
            for i in range(self.game_batch_num):
                while True:
                    try:
                        with open(DATA_BUFFER_PATH, "rb") as data_dict:
                            data_file = pickle.load(data_dict)
                            self.data_buffer = data_file["data_buffer"]
                            self.iters = data_file["iters"]
                            del data_file
                        print("已载入数据")
                        break
                    except:
                        time.sleep(30)

                print("step i {}: ".format(self.iters))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_updata()
                    # 保存模型
                    self.policy_value_net.save_model(MODEL_PATH)

                time.sleep(UPDATE_INTERVAL)  # 每10分钟更新一次模型

                if (i + 1) % self.check_freq == 0:
                    # win_ratio = self.policy_evaluate()
                    # print("current self-play batch: {},win_ratio: {}".format(i + 1, win_ratio))
                    # self.policy_value_net.save_model('./current_policy.model')
                    # if win_ratio > self.best_win_ratio:
                    #     print("New best policy!!!!!!!!")
                    #     self.best_win_ratio = win_ratio
                    #     # update the best_policy
                    #     self.policy_value_net.save_model('./best_policy.model')
                    #     if (self.best_win_ratio == 1.0 and
                    #             self.pure_mcts_playout_num < 5000):
                    #         self.pure_mcts_playout_num += 1000
                    #         self.best_win_ratio = 0.0
                    print("current self-play batch: {}".format(i + 1))
                    self.policy_value_net.save_model(
                        "models/current_policy_batch{}.model".format(i + 1)
                    )
        except KeyboardInterrupt:
            print("\n\rquit")


training_pipeline = TrainPipeline(init_model="current_policy.pkl")
training_pipeline.run()
