import cchess
import time
import os
import argparse
import pickle
import shelve
import numpy as np
from net import PolicyValueNet
from mcts import MCTS_AI
from game import Game
from parameters import C_PUCT, PLAYOUT, DATA_PATH, MODEL_PATH
from tools import (
    move_id2move_action,
    move_action2move_id,
    flip,
    zip_state,
    zip_probs,
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
        # 对弈参数
        self.temp = 1  # 温度
        self.n_playout = PLAYOUT  # 每次移动的模拟次数
        self.c_puct = C_PUCT
        self.init_model = init_model  # 初始化模型路径
        self.iters = 0
        self.mcts_ai = None
        self.policy_value_net = None  # 延迟初始化
        # 尝试从已有文件加载 iters
        if os.path.exists(DATA_PATH):
            try:
                with shelve.open(DATA_PATH) as data_file:
                    self.iters = data_file.get("iters", 0)
                    print(
                        f"[{time.strftime('%H:%M:%S')}] 成功加载当前对局数: {self.iters}"
                    )
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] 加载当前对局数失败: {str(e)}")

    # 从主体加载模型
    def load_model(self):
        if self.policy_value_net is None:  # 仅初始化一次
            model_path = self.init_model if self.init_model else MODEL_PATH
            try:
                self.policy_value_net = PolicyValueNet(model_file=model_path)
                print(f"[{time.strftime('%H:%M:%S')}] 已加载模型: {model_path}")
            except Exception as e:
                print(
                    f"[{time.strftime('%H:%M:%S')}] 加载模型 {model_path} 失败: {str(e)}"
                )
                self.policy_value_net = PolicyValueNet()
                print(f"[{time.strftime('%H:%M:%S')}] 已加载初始模型")
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
            # 左右对称翻转红方和黑方状态
            states_flip = []
            for state in states:
                states_flip.append(np.flip(state, axis=2))
            # 向量化操作翻转概率
            mcts_prob_flip = mcts_prob[flip_map]
            data_flip.append((states_flip, mcts_prob_flip, winner))
        return data + data_flip

    def compress(self, data):
        """将预处理后的对局数据压缩为稀疏数组格式

        Args:
            data (numpy.ndarray): 预处理后的对局数据

        Returns:
            压缩后的稀疏数组
        """
        # 压缩数据
        compressed_data = []
        for states, mcts_probs, winner in data:
            compressed_data.append((zip_state(states), zip_probs(mcts_probs), winner))
        return compressed_data

    def collect_data(self, is_shown=False):
        """收集自我对弈的数据"""
        self.load_model()  # 从本体处加载最新模型
        play_data = self.game.start_self_play(self.mcts_ai, is_shown=is_shown)
        play_data = self.preprocess(play_data)  # 预处理数据
        play_data = self.flip_data(play_data)  # 翻转数据
        play_data = self.compress(play_data)  # 压缩数据
        self.iters += 1  # 更新迭代次数

        try:
            with shelve.open(DATA_PATH) as data_file:
                data_file["data_" + str(self.iters)] = play_data
            print(f"[{time.strftime('%H:%M:%S')}] 成功保存数据到 {DATA_PATH}")
            del play_data  # 释放内存
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] 保存数据失败: {str(e)}")
        return self.iters

    def run(self, is_shown=False):
        """开始收集数据"""
        try:
            while True:
                iters = self.collect_data(is_shown=is_shown)
                print(
                    f"[{time.strftime('%H:%M:%S')}] 第 {iters} 局, 共 {self.episode_len} 步"
                )
        except KeyboardInterrupt:
            print(f"\r[{time.strftime('%H:%M:%S')}] 退出")


if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="收集中国象棋自对弈数据")
    parser.add_argument(
        "--show", action="store_true", default=False, help="是否显示棋盘对弈过程"
    )
    parser.add_argument(
        "--model", type=str, default="current_policy.pkl", help="初始化模型路径"
    )
    args = parser.parse_args()
    # 创建数据收集管道实例
    collecting_pipeline = CollectPipeline(init_model=args.model)
    collecting_pipeline.run(is_shown=args.show)
