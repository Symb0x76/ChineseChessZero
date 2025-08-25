import cchess
import torch
import h5py
import time
import os
import argparse
import numpy as np
from game import Game
from net import PolicyValueNet
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from dataset import PickleDataset
from parameters import (
    PLAYOUT,
    C_PUCT,
    BATCH_SIZE,
    EPOCHS,
    KL_TARG,
    UPDATE_INTERVAL,
    PICKLE_PATH,
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
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = 500
        self.train_iters = 0  # 训练迭代计数
        self.data_iters = 0  # 数据收集迭代计数
        self.dataset = None  # 数据集
        self.dataloader = None  # 数据加载器
        self.pickle_path = PICKLE_PATH  # 优化后的pickle数据路径
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
        # CUDA Check
        self.device = self.policy_value_net.device
        print(f"[{time.strftime('%H:%M:%S')}] 训练设备: {self.device}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            scaler = GradScaler("cuda")

        # 创建优化的pickle数据集
        if self.dataloader is None:
            print(f"[{time.strftime('%H:%M:%S')}] 加载优化的pickle数据集: {self.pickle_path}")
            self.dataset = PickleDataset(self.pickle_path)
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,  # Windows下避免多进程pickle报错
                pin_memory=True,
            )
            print(f"[{time.strftime('%H:%M:%S')}] 数据集加载完成，共 {len(self.dataset)} 个样本")

        # 记录损失&熵
        total_loss = 0.0
        total_entropy = 0.0
        batch_count = 0

        # 读取数据
        for _ in range(self.epochs):
            self.policy_value_net.optimizer.zero_grad()
            epoch_loss = 0.0
            epoch_entropy = 0.0
            batch_in_epoch = 0

            for state_batch, mcts_probs_batch, winner_batch in self.dataloader:
                # 转换数据类型
                state_batch = state_batch.float().to(self.policy_value_net.device)
                mcts_probs_batch = mcts_probs_batch.float().to(
                    self.policy_value_net.device
                )
                winner_batch = winner_batch.float().to(self.policy_value_net.device)

                # 旧策略&旧价值函数
                if torch.cuda.is_available():
                    with autocast("cuda"):
                        old_probs, old_v = self.policy_value_net.policy_value(
                            state_batch
                        )

                        # 前向传播
                        log_act_probs, value = self.policy_value_net.policy_value_net(
                            state_batch
                        )
                        value = value.flatten()

                        # 计算损失
                        value_loss = torch.nn.functional.mse_loss(
                            input=value, target=winner_batch
                        )
                        policy_loss = -torch.mean(
                            torch.sum(
                                mcts_probs_batch * log_act_probs,
                                dim=1,
                            )
                        )
                        loss = value_loss + policy_loss

                    # 反向传播
                    scaler.scale(loss).backward()
                    scaler.step(self.policy_value_net.optimizer)
                    scaler.update()

                    # 计算熵
                    with torch.no_grad():
                        entropy = -torch.mean(
                            torch.sum(torch.exp(log_act_probs) * log_act_probs, dim=1)
                        )
                        entropy = entropy.item()

                    # 新的策略&新的价值函数
                    with autocast("cuda"):
                        new_probs, new_v = self.policy_value_net.policy_value(
                            state_batch
                        )
                else:
                    old_probs, old_v = self.policy_value_net.policy_value(state_batch)

                    # 前向传播
                    loss, entropy = self.policy_value_net.train_step(
                        state_batch,
                        mcts_probs_batch,
                        winner_batch,
                        self.learning_rate * self.lr_multiplier,
                    )

                    # 新的策略&新的价值函数
                    new_probs, new_v = self.policy_value_net.policy_value(state_batch)

                    # 将数据移回CPU进行KL散度计算
                old_probs_cpu = (
                    old_probs.cpu().numpy() if torch.is_tensor(old_probs) else old_probs
                )
                new_probs_cpu = (
                    new_probs.cpu().numpy() if torch.is_tensor(new_probs) else new_probs
                )

                kl = np.mean(
                    np.sum(
                        old_probs_cpu
                        * (
                            np.log(old_probs_cpu + 1e-10)
                            - np.log(new_probs_cpu + 1e-10)
                        ),
                        axis=1,
                    )
                )

                epoch_loss += loss
                epoch_entropy += entropy
                batch_count += 1

                # 如果 KL 散度很差，则提前终止
                if kl > self.kl_targ * 4:
                    break

            # 计算该轮平均损失和熵
            if batch_in_epoch > 0:
                epoch_loss /= batch_in_epoch
                epoch_entropy /= batch_in_epoch
                total_loss += epoch_loss
                total_entropy += epoch_entropy
                batch_count += 1

        # 自适应调整学习率
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        # 计算平均损失和熵
        avg_loss = total_loss / max(1, batch_count)
        avg_entropy = total_entropy / max(1, batch_count)

        # 将数据移至CPU计算解释方差
        winner_batch_cpu = (
            winner_batch.cpu().numpy()
            if torch.is_tensor(winner_batch)
            else winner_batch
        )
        old_v_cpu = (
            old_v.cpu().flatten().numpy() if torch.is_tensor(old_v) else old_v.flatten()
        )
        new_v_cpu = (
            new_v.cpu().flatten().numpy() if torch.is_tensor(new_v) else new_v.flatten()
        )

        # 计算解释方差
        explained_var_old = 1 - np.var(winner_batch_cpu - old_v_cpu) / np.var(
            winner_batch_cpu
        )
        explained_var_new = 1 - np.var(winner_batch_cpu - new_v_cpu) / np.var(
            winner_batch_cpu
        )

        print(
            (
                f"[{time.strftime('%H:%M:%S')}] kl:{kl:.9f},"
                f"lr_multiplier:{self.lr_multiplier:.6f},"
                f"loss:{avg_loss:.6f},"
                f"entropy:{avg_entropy:.6f},"
                f"explained_var_old:{explained_var_old:.6f},"
                f"explained_var_new:{explained_var_new:.6f}"
            )
        )
        return avg_loss, avg_entropy

    def policy_evaluate(self):
        """评估当前策略的胜率（暂时返回一个模拟值）"""
        # 这里应该实现实际的策略评估逻辑
        # 暂时返回一个模拟的胜率值
        return 0.6

    def save_train_state(self):
        """保存训练状态到 pickle 文件"""
        import pickle
        train_state_path = "train_state.pkl"
        state = {
            "train_iters": self.train_iters,
            "data_iters": self.data_iters,
            "lr_multiplier": self.lr_multiplier,
        }
        try:
            with open(train_state_path, "wb") as f:
                pickle.dump(state, f)
            print(f"[{time.strftime('%H:%M:%S')}] 训练状态已保存到 {train_state_path}")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] 保存训练状态失败: {str(e)}")

    def load_train_state(self):
        """从 pickle 文件加载训练状态"""
        import pickle
        train_state_path = "train_state.pkl"
        try:
            if os.path.exists(train_state_path):
                with open(train_state_path, "rb") as f:
                    state = pickle.load(f)
                self.train_iters = state.get("train_iters", 0)
                self.data_iters = state.get("data_iters", 0)
                self.lr_multiplier = state.get("lr_multiplier", 1.0)
                print(
                    f"[{time.strftime('%H:%M:%S')}] 已加载训练状态: 训练迭代 {self.train_iters}, 数据迭代 {self.data_iters}"
                )
                return True
            else:
                print(f"[{time.strftime('%H:%M:%S')}] 训练状态文件不存在，从头开始训练")
                return False
        except Exception as e:
            print(
                f"[{time.strftime('%H:%M:%S')}] 无法加载训练状态: {str(e)}，从头开始训练"
            )
        return False

    def run(self):
        """开始训练"""
        try:
            # 尝试加载之前的训练状态
            self.load_train_state()

            # 检查优化的pickle数据文件是否存在
            if not os.path.exists(self.pickle_path):
                print(f"[{time.strftime('%H:%M:%S')}] 错误: 优化的数据文件 {self.pickle_path} 不存在")
                print(f"[{time.strftime('%H:%M:%S')}] 请先运行 convert.py 生成优化的数据文件")
                return

            while True:
                if self.dataset is None:
                    print(f"[{time.strftime('%H:%M:%S')}] 首次加载pickle数据集")
                    self.dataset = PickleDataset(self.pickle_path)
                    self.dataloader = DataLoader(
                        self.dataset,
                        self.batch_size,
                        shuffle=True,
                        num_workers=0,  # Windows下避免多进程pickle报错
                        pin_memory=True,
                    )
                    print(f"[{time.strftime('%H:%M:%S')}] 数据集加载完成，共 {len(self.dataset)} 个样本")

                print(f"[{time.strftime('%H:%M:%S')}] 训练迭代 {self.train_iters}")
                if len(self.dataset) > self.batch_size:
                    loss, entropy = self.policy_update()
                    self.policy_value_net.save_model(MODEL_PATH)
                    self.train_iters += 1
                    self.save_train_state()
                    if self.train_iters % self.check_freq == 0:
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
