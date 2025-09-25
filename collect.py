import cchess
import os
import argparse
import h5py
import numpy as np
from net import PolicyValueNet
from mcts import MCTS_AI
from game import Game
from parameters import C_PUCT, PLAYOUT, DATA_DIR, MODEL_DIR
from tools import (
    move_id2move_action,
    move_action2move_id,
    flip,
    log,
)

"""
产生的数据应该为17*7*10*9的数组
17 = 8*2+1 (8表示各方当前状态以及前七步状态, 1表示当前玩家的指示, 红时为全1, 黑时为全0)
7 棋子指示通道数(7种)
10*9 棋盘长和宽
"""


# 定义整个对弈收集数据流程
class CollectPipeline:
    def __init__(self, init_model=None):
        self.board = cchess.Board()
        self.game = Game(self.board)
        self.temp = 1.0
        self.n_playout = PLAYOUT
        self.c_puct = C_PUCT
        self.init_model = init_model
        self.iters = 0
        self.mcts_ai = None
        self.policy_value_net = None  # 延迟初始化
        self.data_path = os.path.join(DATA_DIR, "data.h5")
        # Load iters
        if os.path.exists(self.data_path):
            try:
                with h5py.File(self.data_path, "r") as h5f:
                    self.iters = h5f.attrs.get("iters", 0)
                    log(f"Current game count: {self.iters}")
            except Exception as e:
                log(f"Failed to load current game count: {str(e)}", "ERROR")

    # 加载模型
    def load_model(self):
        if self.policy_value_net is None:
            model_path = self.init_model if self.init_model else MODEL_DIR
            try:
                self.policy_value_net = PolicyValueNet(model=model_path)
                log(f"Loaded model: {model_path}")
            except Exception as e:
                log(f"Failed to load model {model_path}: {str(e)}", "ERROR")
                self.policy_value_net = PolicyValueNet()
            self.mcts_ai = MCTS_AI(
                self.policy_value_net.policy_value_fn,
                c_puct=self.c_puct,
                n_playout=self.n_playout,
                is_selfplay=True,
            )

    def preprocess(self, play_data):
        """play_data为迭代器, 每个元素都是(red_states, black_states, mcts_prob, winner)
        预处理将red_state和black_state堆叠为17通道的状态
        直接返回(states, mcts_probs, winners)
        """

        # 记录对局总步数
        processed_data = []
        self.episode_len = 0

        for red_states, black_states, mcts_prob, winner in play_data:
            self.episode_len += 1

            # 走子方指示层
            if self.board.turn == cchess.RED:
                current_player = np.ones((1, 7, 10, 9), dtype=np.float16)
            else:
                current_player = np.zeros((1, 7, 10, 9), dtype=np.float16)

            # 堆叠出完整状态
            states = np.concatenate((red_states, black_states), axis=0)
            states = np.concatenate((states, current_player), axis=0)

            # 确保mcts_prob是NumPy数组
            if not isinstance(mcts_prob, np.ndarray):
                mcts_prob = np.array(mcts_prob)

            # 验证和修复概率分布
            prob_sum = np.sum(mcts_prob)
            if prob_sum <= 0:
                log(f"mcts_prob sum is {prob_sum}; skipping this step", "WARNING")
                continue
            elif abs(prob_sum - 1.0) > 1e-6:
                log(f"mcts_prob sum is {prob_sum:.6f}; normalizing", "INFO")
                mcts_prob = mcts_prob / prob_sum

            # 验证概率分布是否过于集中（类似one-hot）
            max_prob = np.max(mcts_prob)
            non_zero_count = np.sum(mcts_prob > 1e-8)
            if max_prob > 0.99 and non_zero_count <= 3:
                game_idx = getattr(self, "current_game_index", "?")
                log(
                    f"Game {game_idx} | mcts_prob over-centered: max={max_prob:.4f}, non-zero elements={non_zero_count}",
                    "WARNING",
                )

            # 堆叠出完整数据
            processed_data.append((states, mcts_prob, winner))

        return processed_data

    def flip_data(self, data):
        """对数据进行翻转操作"""
        # 创建映射表
        flip_map = np.array(
            [
                move_action2move_id[flip(move_id2move_action[i])]
                for i in range(len(move_id2move_action))
            ]
        )
        data_flip = []
        for states, mcts_prob, winner in data:
            states_flip = []
            for state in states:
                states_flip.append(np.flip(state, axis=2))
            mcts_prob_flip = mcts_prob[flip_map]
            data_flip.append((states_flip, mcts_prob_flip, winner))
        return data + data_flip

    def collect_data(self, is_shown=False):
        """Collect self-play data"""
        self.load_model()  # 从本体处加载最新模型
        # 保存当前对局索引用于日志展示
        self.current_game_index = self.iters + 1
        play_data = self.game.start_self_play(
            self.mcts_ai, is_shown=is_shown, game_index=self.current_game_index
        )
        play_data = self.preprocess(play_data)
        play_data = self.flip_data(play_data)

        try:
            # 创建 HDF5 文件并直接保存数据
            with h5py.File(self.data_path, "a") as h5f:
                current_iter = h5f.attrs.get("iters", 0)
                game_group = h5f.create_group(f"game_{current_iter}")

                # 直接使用gzip压缩
                game_group.create_dataset(
                    "states",
                    data=[state for state, _, _ in play_data],
                    compression="gzip",
                )
                game_group.create_dataset(
                    "mcts_probs",
                    data=[prob for _, prob, _ in play_data],
                    compression="gzip",
                )
                game_group.create_dataset(
                    "winners", data=[winner for _, _, winner in play_data]
                )

                # 更新游戏索引
                h5f.attrs["iters"] = current_iter + 1
                self.iters = current_iter + 1

                log(f"Saved data to {self.data_path}")

            # 释放内存
            del play_data
        except Exception as e:
            log(f"Failed to save data: {str(e)}", "ERROR")

        return self.iters

    def run(self, is_shown=False):
        """Start data collection"""
        try:
            while True:
                iters = self.collect_data(is_shown=is_shown)
                log(f"Episode {iters}, steps {self.episode_len}")
        except KeyboardInterrupt:
            log("Exit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="收集中国象棋自对弈数据")
    parser.add_argument(
        "--show", action="store_true", default=False, help="是否显示棋盘对弈过程"
    )
    parser.add_argument(
        "--model", type=str, default="current_policy.pkl", help="初始化模型路径"
    )
    args = parser.parse_args()
    collecting_pipeline = CollectPipeline(init_model=args.model)
    collecting_pipeline.run(is_shown=args.show)
