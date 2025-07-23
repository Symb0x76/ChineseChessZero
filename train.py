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
from parameters import (
    PLAYOUT,
    C_PUCT,
    BATCH_SIZE,
    EPOCHS,
    KL_TARG,
    GAME_BATCH_NUM,
    UPDATE_INTERVAL,
    DATA_PATH,
    MODEL_PATH,
    CHECK_FREQ,
)


class ChessDataset(Dataset):
    """象棋对弈数据集类 (HDF5 版本)"""

    def __init__(self, data_path, max_items=None):
        self.data_path = data_path
        self.max_items = max_items

        # 计算数据集大小和创建索引映射
        self.game_indices = []
        self.total_moves = 0

        # 临时打开文件获取元数据
        with h5py.File(data_path, "r") as h5f:
            # 获取游戏数量
            games_count = h5f.attrs.get("iters", 0)
            print(f"[{time.strftime('%H:%M:%S')}] 数据集中共有 {games_count} 局游戏")

            # 遍历所有游戏，计算总步数
            for game_idx in range(games_count):
                game_group = h5f.get(f"game_{game_idx}")
                if game_group is not None:
                    # 通过状态数据集的第一维获取步数，而不是通过属性
                    if "states" in game_group:
                        steps = game_group["states"].shape[0]
                        # 为每一步创建索引映射 (游戏索引, 步骤索引)
                        for step_idx in range(steps):
                            self.game_indices.append((game_idx, step_idx))
                        self.total_moves += steps
                    else:
                        print(
                            f"[{time.strftime('%H:%M:%S')}] 警告: game_{game_idx} 缺少states数据集"
                        )

            # 如果设置了最大项目数，限制大小
            if max_items and max_items < len(self.game_indices):
                self.game_indices = self.game_indices[:max_items]
                print(f"[{time.strftime('%H:%M:%S')}] 限制数据集大小为 {max_items} 条")

            self.length = len(self.game_indices)
            print(f"[{time.strftime('%H:%M:%S')}] 数据集总共包含 {self.length} 步棋")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 确保索引在有效范围内
        if idx < 0 or idx >= self.length:
            raise IndexError(f"索引 {idx} 超出范围 (0-{self.length-1})")

        # 获取游戏索引和步骤索引
        game_idx, step_idx = self.game_indices[idx]

        # 每次获取数据时打开和关闭文件
        with h5py.File(self.data_path, "r") as h5f:
            # 获取对应游戏组
            game_group = h5f[f"game_{game_idx}"]

            # 从游戏组中读取数据
            state = game_group["states"][step_idx]
            mcts_prob = game_group["mcts_probs"][step_idx]
            winner = game_group["winners"][step_idx]

            return state, mcts_prob, winner


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
        self.train_iters = 0  # 训练迭代计数
        self.data_iters = 0  # 数据收集迭代计数
        self.dataset = None  # 数据集
        self.dataloader = None  # 数据加载器
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

        # 创建dataloader
        if self.dataloader is None:
            self.dataset = ChessDataset(DATA_PATH)
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )

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

    def save_train_state(self):
        """保存训练状态到 HDF5 文件"""
        try:
            train_state_path = "train_state.h5"
            with h5py.File(train_state_path, "w") as f:
                # 保存训练状态作为属性
                f.attrs["train_iters"] = self.train_iters
                f.attrs["data_iters"] = self.data_iters
                f.attrs["lr_multiplier"] = self.lr_multiplier
            print(f"[{time.strftime('%H:%M:%S')}] 训练状态已保存到 {train_state_path}")
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] 保存训练状态失败: {str(e)}")

    def load_train_state(self):
        """从 HDF5 文件加载训练状态"""
        train_state_path = "train_state.h5"
        try:
            if os.path.exists(train_state_path):
                with h5py.File(train_state_path, "r") as f:
                    # 读取属性
                    self.train_iters = f.attrs.get("train_iters", 0)
                    self.data_iters = f.attrs.get("data_iters", 0)
                    self.lr_multiplier = f.attrs.get("lr_multiplier", 1.0)
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

            while self.train_iters < self.game_batch_num:
                # 加载最新数据
                try:
                    with h5py.File(DATA_PATH, "r") as data_file:
                        # 检查iters值
                        current_data_iters = data_file.get("iters", 0)

                        # 检查是否有新数据
                        if current_data_iters > self.data_iters:
                            # 更新数据迭代计数
                            self.data_iters = current_data_iters
                            print(
                                f"[{time.strftime('%H:%M:%S')}] 检测到新数据，数据迭代: {self.data_iters}, 重新创建数据集"
                            )
                            if (
                                hasattr(self, "dataset")
                                and self.dataset is not None
                                and hasattr(self.dataset, "h5f")
                            ):
                                self.dataset.h5f.close()

                            self.dataset = ChessDataset(DATA_PATH)
                            self.dataloader = DataLoader(
                                self.dataset,
                                self.batch_size,
                                shuffle=True,
                                num_workers=4,
                                pin_memory=True,
                            )
                        else:
                            if self.dataset is None:
                                print(
                                    f"[{time.strftime('%H:%M:%S')}] 首次加载数据，数据迭代: {current_data_iters}"
                                )
                                self.dataset = ChessDataset(DATA_PATH)
                                self.dataloader = DataLoader(
                                    self.dataset,
                                    self.batch_size,
                                    shuffle=True,
                                    num_workers=4,
                                    pin_memory=True,
                                )
                                self.data_iters = current_data_iters
                            else:
                                print(
                                    f"[{time.strftime('%H:%M:%S')}] 等待新数据... 当前数据迭代: {self.data_iters}, 训练迭代: {self.train_iters}"
                                )
                                # time.sleep(UPDATE_INTERVAL)
                except Exception as e:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] 加载数据失败: {str(e)}, 10秒后重试"
                    )
                    time.sleep(10)

                # 执行训练
                print(
                    f"[{time.strftime('%H:%M:%S')}] 训练迭代 {self.train_iters}, 数据迭代 {self.data_iters}"
                )
                if len(self.dataset) > self.batch_size:
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
