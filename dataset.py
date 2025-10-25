import os
import numpy as np
from torch.utils.data import Dataset


class NpyMemmapDataset(Dataset):
    """
    基于三个 .npy (.npy + 内存映射) 的象棋数据集：
    - 目录模式（推荐，与 convert.py 一致）：
        states.npy, mcts.npy, winners.npy
    - 兼容旧模式（stem 前缀）：
        <stem>_states.npy, <stem>_mcts.npy, <stem>_winners.npy

    仅按索引切片，从磁盘内存映射读取，几乎不占额外内存。
    """

    def __init__(self, path: str):
        # 目录模式
        if os.path.isdir(path):
            states_path = os.path.join(path, "states.npy")
            mcts_path = os.path.join(path, "mcts.npy")
            winners_path = os.path.join(path, "winners.npy")
        else:
            base = os.path.splitext(path)[0]
            states_path = f"{base}_states.npy"
            mcts_path = f"{base}_mcts.npy"
            winners_path = f"{base}_winners.npy"

        if not (
            os.path.exists(states_path)
            and os.path.exists(mcts_path)
            and os.path.exists(winners_path)
        ):
            raise FileNotFoundError(
                "未找到 .npy 数据文件，请先运行 convert.py 生成：\n"
                f"  - {states_path}\n  - {mcts_path}\n  - {winners_path}"
            )

        self.states_path = states_path
        self.mcts_path = mcts_path
        self.winners_path = winners_path
        self._reload_memmaps()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        state = self.states[idx]
        mcts_prob = self.mcts[idx]
        winner = self.winners[idx]
        return state, mcts_prob, winner

    def _reload_memmaps(self):
        self.states = np.load(self.states_path, mmap_mode="r")
        self.mcts = np.load(self.mcts_path, mmap_mode="r")
        self.winners = np.load(self.winners_path, mmap_mode="r")
        if not (len(self.states) == len(self.mcts) == len(self.winners)):
            raise ValueError(
                f"数据长度不一致: states={len(self.states)}, mcts={len(self.mcts)}, winners={len(self.winners)}"
            )
        self.length = len(self.states)

    def __getstate__(self):
        state = self.__dict__.copy()
        # numpy.memmap 对象不可安全地通过 pickle 共享，传路径即可
        state["states"] = None
        state["mcts"] = None
        state["winners"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._reload_memmaps()
