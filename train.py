import cchess
import pickle
import torch
import os
import argparse
import sys
import numpy as np
import copy
from tools import log
from game import Game
from net import PolicyValueNet
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from dataset import NpyMemmapDataset
from parameters import (
    PLAYOUT,
    C_PUCT,
    BATCH_SIZE,
    EPOCHS,
    KL_TARG,
    CHECK_FREQ,
    DATA_DIR,
    MODEL_DIR,
)


class TrainPipeline:
    def __init__(self, init_model: str | None = None):
        """初始化训练流水线"""
        # 基础参数
        self.board = cchess.Board()
        self.game = Game(self.board)
        self.n_playout = PLAYOUT
        self.c_puct = C_PUCT
        self.learning_rate = 1e-3
        self.lr_multiplier = 1.0
        self.temp = 1.0
        self.batch_size = BATCH_SIZE
        self.epochs = EPOCHS
        self.kl_targ = KL_TARG
        self.check_freq = CHECK_FREQ
        # 策略标签平滑 & 熵守护
        self.label_smoothing = 0.05  # 5% 平滑
        self.min_entropy_guard = 1.0  # 若更新后熵低于该值则回滚
        # 计数 / 数据
        self.train_iters = 0
        self.data_iters = 0
        self.dataset = None
        self.dataloader = None
        # 训练数据目录（与 convert.py 一致：states.npy/mcts.npy/winners.npy）
        self.data_dir = DATA_DIR
        # 当前模型路径由 MODEL_DIR 推导
        self.current_policy_path = os.path.join(MODEL_DIR, "current_policy.pkl")

        # 模型
        if init_model:
            try:
                self.policy_value_net = PolicyValueNet(model=init_model)
                log(f"Loaded model: {init_model}")
            except Exception as e:
                log(
                    f"Failed to load model {init_model}: {e}. Start training from scratch",
                    level="WARNING",
                )
                self.policy_value_net = PolicyValueNet()
        else:
            log("Start training blankly")
            self.policy_value_net = PolicyValueNet()

    def policy_update(self):
        """Run one policy/value update step and return avg loss and entropy"""
        device = self.policy_value_net.device
        if not hasattr(self, "_device_printed"):
            log(f"Device: {device}")
            self._device_printed = True

        use_cuda = torch.cuda.is_available()
        scaler = GradScaler("cuda") if use_cuda else None
        if use_cuda:
            torch.cuda.empty_cache()

        # DataLoader 准备
        if self.dataloader is None:
            # collate_fn: stack NumPy arrays then copy into writable tensors
            def npy_collate(batch):
                states, mcts, winners = zip(*batch)
                states = torch.tensor(np.stack(states), dtype=torch.float32)
                mcts = torch.tensor(np.stack(mcts), dtype=torch.float32)
                winners = torch.tensor(np.array(winners), dtype=torch.float32)
                return states, mcts, winners

            log(f"Loading .npy dataset from: {self.data_dir}")
            states_path = os.path.join(self.data_dir, "states.npy")
            if not os.path.exists(states_path):
                log(f"Not found: {states_path}", level="ERROR")
                log("Please run convert.py to generate .npy data", level="ERROR")
                raise FileNotFoundError(states_path)
            self.dataset = NpyMemmapDataset(self.data_dir)
            # 在 CUDA 上启用 pin_memory 能提升 H2D 传输性能
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=use_cuda,
                collate_fn=npy_collate,
            )
            log(f"Dataset loaded: {len(self.dataset)} samples")

        # 累计器
        total_loss = total_entropy = 0.0
        total_policy_loss = total_value_loss = 0.0
        total_batches = 0
        last_old_v = last_new_v = last_winner_batch = None
        last_kl = 0.0
        # winners 分布
        win_neg = win_zero = win_pos = 0

        for epoch in range(self.epochs):
            epoch_loss = epoch_entropy = 0.0
            epoch_policy_loss = epoch_value_loss = 0.0

            for batch_idx, (state_batch, mcts_probs_batch, winner_batch) in enumerate(
                self.dataloader
            ):
                # 校验 mcts_probs 归一性
                sums = mcts_probs_batch.sum(dim=1)
                if not ((sums > 0.99) & (sums < 1.01)).all():
                    raise ValueError("mcts_probs_batch rows must sum to 1 (±0.01)")
                if torch.isnan(mcts_probs_batch).any():
                    raise ValueError("mcts_probs_batch contains NaN")

                # 设置学习率（可动态调整）
                current_lr = self.learning_rate * self.lr_multiplier
                for g in self.policy_value_net.optimizer.param_groups:
                    g["lr"] = current_lr

                state_batch = state_batch.float().to(device)
                mcts_probs_batch = mcts_probs_batch.float().to(device)
                winner_batch = winner_batch.float().to(device)

                # 旧策略 (KL 基线)
                old_probs, old_v = self.policy_value_net.policy_value(state_batch)
                # 切回训练模式 (policy_value 内部设置 eval)
                self.policy_value_net.policy_value_net.train()

                self.policy_value_net.optimizer.zero_grad()
                # 备份参数用于潜在回滚（熵塌陷）
                backup_weights = {
                    k: v.clone()
                    for k, v in self.policy_value_net.policy_value_net.state_dict().items()
                }
                backup_opt_state = copy.deepcopy(
                    self.policy_value_net.optimizer.state_dict()
                )
                if use_cuda:
                    with autocast("cuda"):
                        log_act_probs, value = self.policy_value_net.policy_value_net(
                            state_batch
                        )
                        value = value.flatten()
                        value_loss = torch.nn.functional.mse_loss(value, winner_batch)
                        if self.label_smoothing > 0:
                            eps = self.label_smoothing
                            smooth_target = (
                                1 - eps
                            ) * mcts_probs_batch + eps / mcts_probs_batch.size(1)
                        else:
                            smooth_target = mcts_probs_batch
                        policy_loss = -torch.mean(
                            torch.sum(smooth_target * log_act_probs, dim=1)
                        )
                        loss = value_loss + policy_loss
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.policy_value_net.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.policy_value_net.policy_value_net.parameters(), 5.0
                    )
                    scaler.step(self.policy_value_net.optimizer)
                    scaler.update()
                else:
                    log_act_probs, value = self.policy_value_net.policy_value_net(
                        state_batch
                    )
                    value = value.flatten()
                    value_loss = torch.nn.functional.mse_loss(value, winner_batch)
                    if self.label_smoothing > 0:
                        eps = self.label_smoothing
                        smooth_target = (
                            1 - eps
                        ) * mcts_probs_batch + eps / mcts_probs_batch.size(1)
                    else:
                        smooth_target = mcts_probs_batch
                    policy_loss = -torch.mean(
                        torch.sum(smooth_target * log_act_probs, dim=1)
                    )
                    loss = value_loss + policy_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.policy_value_net.policy_value_net.parameters(), 5.0
                    )
                    self.policy_value_net.optimizer.step()
                # 数值稳定性检查
                if (
                    torch.isnan(loss)
                    or torch.isinf(loss)
                    or torch.isnan(log_act_probs).any()
                    or torch.isinf(log_act_probs).any()
                ):
                    self.policy_value_net.policy_value_net.load_state_dict(
                        backup_weights
                    )
                    self.policy_value_net.optimizer.load_state_dict(backup_opt_state)
                    old_mult = self.lr_multiplier
                    self.lr_multiplier = max(0.05, self.lr_multiplier / 2)
                    continue

                # 新策略 (for KL)
                new_probs, new_v = self.policy_value_net.policy_value(state_batch)
                # policy_value 会将模型切到 eval，这里切回 train，避免对后续训练步骤造成影响
                self.policy_value_net.policy_value_net.train()
                last_old_v, last_new_v, last_winner_batch = old_v, new_v, winner_batch

                # winners 分布累计
                with torch.no_grad():
                    win_neg += (winner_batch < 0).sum().item()
                    win_zero += (winner_batch == 0).sum().item()
                    win_pos += (winner_batch > 0).sum().item()

                # KL(old||new)
                last_kl = float(
                    np.mean(
                        np.sum(
                            old_probs
                            * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                            axis=1,
                        )
                    )
                )

                with torch.no_grad():
                    entropy = -torch.mean(
                        torch.sum(torch.exp(log_act_probs) * log_act_probs, dim=1)
                    ).item()

                # 累计 epoch
                l = float(loss.item())
                epoch_loss += l
                epoch_entropy += entropy
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                total_batches += 1

                # 熵塌陷保护：若熵过低且目标分布非极端一热则回滚
                if entropy < self.min_entropy_guard:
                    non_zero_targets = (
                        (mcts_probs_batch > 0).sum(dim=1).float().mean().item()
                    )
                    if non_zero_targets > 1.5:  # 目标分布不是基本一热
                        self.policy_value_net.policy_value_net.load_state_dict(
                            backup_weights
                        )
                        self.policy_value_net.optimizer.load_state_dict(
                            backup_opt_state
                        )
                        old_multiplier = self.lr_multiplier
                        self.lr_multiplier = max(0.1, self.lr_multiplier / 2)
                        continue  # 不计入统计

                if last_kl > self.kl_targ * 4:
                    # 高 KL: 降低 lr_multiplier 并继续，不整轮早停
                    old_multiplier = self.lr_multiplier
                    self.lr_multiplier = max(0.05, self.lr_multiplier / 1.5)
                    # 保持训练，未早停

            # 汇总 epoch
            total_loss += epoch_loss
            total_entropy += epoch_entropy
            total_policy_loss += epoch_policy_loss
            total_value_loss += epoch_value_loss

        # 动态 lr_multiplier 调整
        if last_kl > self.kl_targ * 2 and self.lr_multiplier > 0.05:
            self.lr_multiplier = max(0.05, self.lr_multiplier / 1.2)
        elif last_kl < self.kl_targ / 2 and self.lr_multiplier < 2.0:
            self.lr_multiplier = min(2.0, self.lr_multiplier * 1.2)

        denom = max(1, total_batches)
        avg_loss = total_loss / denom
        avg_entropy = total_entropy / denom
        avg_policy_loss = total_policy_loss / denom
        avg_value_loss = total_value_loss / denom

        # 解释方差
        if last_winner_batch is not None:
            w = last_winner_batch.detach().cpu().numpy()
            old_v_arr = (
                last_old_v.flatten()
                if isinstance(last_old_v, np.ndarray)
                else last_old_v.cpu().flatten().numpy()
            )
            new_v_arr = (
                last_new_v.flatten()
                if isinstance(last_new_v, np.ndarray)
                else last_new_v.cpu().flatten().numpy()
            )
            explained_var_old = 1 - np.var(w - old_v_arr) / (np.var(w) + 1e-12)
            explained_var_new = 1 - np.var(w - new_v_arr) / (np.var(w) + 1e-12)
        else:
            explained_var_old = explained_var_new = 0.0

        log(
            f"kl:{last_kl:.9f},lr_multiplier:{self.lr_multiplier:.6f},"
            f"loss:{avg_loss:.6f},policy_loss:{avg_policy_loss:.6f},value_loss:{avg_value_loss:.6f},"
            f"entropy:{avg_entropy:.6f},explained_var_old:{explained_var_old:.6f},explained_var_new:{explained_var_new:.6f}"
        )

        total_w = win_neg + win_zero + win_pos
        if total_w > 0:
            log(
                f"Winners distribution: total={total_w} -1:{win_neg} ({win_neg/total_w:.2%}) 0:{win_zero} ({win_zero/total_w:.2%}) 1:{win_pos} ({win_pos/total_w:.2%})"
            )
        else:
            log("Winners distribution: no data", level="WARNING")
        return avg_loss, avg_entropy

    '''
    def policy_evaluate(self):
        """评估当前策略的胜率（暂时返回一个模拟值）"""
        # 这里应该实现实际的策略评估逻辑
        # 暂时返回一个模拟的胜率值
        return 0.6
        '''

    def save_train_state(self):
        """Save training state to a pickle file"""
        os.makedirs(MODEL_DIR, exist_ok=True)
        train_state_path = os.path.join(MODEL_DIR, "train_state.pkl")
        state = {
            "train_iters": self.train_iters,
            "data_iters": self.data_iters,
            "lr_multiplier": self.lr_multiplier,
        }
        try:
            with open(train_state_path, "wb") as f:
                pickle.dump(state, f)
        except Exception as e:
            log(f"Failed to save training state: {str(e)}", level="ERROR")

    def load_train_state(self):
        """Load training state from a pickle file"""
        train_state_path = os.path.join(MODEL_DIR, "train_state.pkl")
        try:
            if os.path.exists(train_state_path):
                with open(train_state_path, "rb") as f:
                    state = pickle.load(f)
                self.train_iters = state.get("train_iters", 0)
                self.data_iters = state.get("data_iters", 0)
                self.lr_multiplier = state.get("lr_multiplier", 1.0)
                log(
                    f"Loaded training state: train_iters={self.train_iters}, data_iters={self.data_iters}"
                )
                return True
            else:
                log("Training state not found; starting blankly", level="WARNING")
                return False
        except Exception as e:
            log(
                f"Failed to load training state: {str(e)}; starting blankly",
                level="ERROR",
            )
            return False

    def run(self):
        """Start training"""
        try:
            # 尝试加载之前的训练状态
            self.load_train_state()

            while True:
                if self.dataset is None:
                    log("Loading .npy dataset")
                    states_path = os.path.join(self.data_dir, "states.npy")
                    if not os.path.exists(states_path):
                        log(f"Not found: {states_path}", level="ERROR")
                        log(
                            "Please run convert.py to generate .npy data", level="ERROR"
                        )
                        sys.exit(1)
                    self.dataset = NpyMemmapDataset(self.data_dir)
                    use_cuda = torch.cuda.is_available()

                    # Use the same collate_fn as above to avoid read-only NumPy warnings
                    def npy_collate(batch):
                        states, mcts, winners = zip(*batch)
                        states = torch.tensor(np.stack(states), dtype=torch.float32)
                        mcts = torch.tensor(np.stack(mcts), dtype=torch.float32)
                        winners = torch.tensor(np.array(winners), dtype=torch.float32)
                        return states, mcts, winners

                    self.dataloader = DataLoader(
                        self.dataset,
                        self.batch_size,
                        shuffle=True,
                        pin_memory=use_cuda,
                        collate_fn=npy_collate,
                    )

                log(f"Training iteration {self.train_iters}")
                if len(self.dataset) > self.batch_size:
                    loss, entropy = self.policy_update()
                    os.makedirs(MODEL_DIR, exist_ok=True)
                    self.policy_value_net.save_model(self.current_policy_path)
                    self.train_iters += 1
                    self.save_train_state()
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
                        log(f"Saved checkpoint: training iteration {self.train_iters}")
                        os.makedirs(MODEL_DIR, exist_ok=True)
                        self.policy_value_net.save_model(
                            os.path.join(
                                MODEL_DIR, f"current_policy_batch{self.train_iters}.pkl"
                            )
                        )
                else:
                    log("Insufficient data; exiting", level="ERROR")
                    sys.exit(1)
        except KeyboardInterrupt:
            log("Saving training state and exiting")
            self.save_train_state()
            log(f"Training state saved to {os.path.join(MODEL_DIR, 'train_state.pkl')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="收集中国象棋自对弈数据")
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(MODEL_DIR, "current_policy.pkl"),
        help="初始化模型路径",
    )
    args = parser.parse_args()
    training_pipeline = TrainPipeline(init_model=args.model)
    training_pipeline.run()
