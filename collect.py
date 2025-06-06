import pickle
import cchess
import time
import os
import copy
import numpy as np
from collections import deque
from net import PolicyValueNet
from mcts import MCTS_AI
from IPython.display import SVG, display
from parameters import C_PUCT, PLAYOUT, BUFFER_SIZE, DATA_BUFFER_PATH, MODEL_PATH
from tools import (
    decode_board,
    move_id2move_action,
    move_action2move_id,
    is_tie,
    zip_state_mcts_prob,
    flip,
)


class Game(object):
    """
    在cchess.Board类基础上定义Game类, 用于启动并控制一整局对局的完整流程,
    收集对局过程中的数据，以及进行棋盘的展示
    """

    def __init__(self, board):
        self.board = board

    # 可视化棋盘
    def graphic(self, board):
        print("player1 take: ", "RED" if cchess.RED else "BLACK")
        print("player0 take: ", "BLACK" if cchess.RED else "RED")
        svg = cchess.svg.board(
            board,
            size=450,
            coordinates=True,
            axes_type=1,
            checkers=board.checkers(),
            orientation=cchess.RED,
        )
        display(SVG(svg))

    # 用于人机对战，人人对战等
    def start_play(self, player1, player0, is_shown=True):
        """
        启动一场对局

        Args:
            player1: 玩家1(红方)
            player0: 玩家0(黑方)
            先手玩家1
            is_shown: 是否显示棋盘

        Returns:
            winner: 获胜方, True (cchess.RED) 或 False (cchess.BLACK) 或 None (平局)
        """

        # 初始化棋盘
        self.board = cchess.Board()

        # 设置玩家(默认玩家1先手)
        player1.set_player_idx(1)
        player0.set_player_idx(0)
        players = {cchess.RED: player1, cchess.BLACK: player0}

        # 显示初始棋盘
        if is_shown:
            self.graphic(self.board)

        # 开始游戏循环
        while True:
            player_in_turn = players[self.board.turn]
            move = player_in_turn.get_action(self.board)

            # 执行移动
            self.board.push(move)

            # 更新显示
            if is_shown:
                self.graphic(self.board)

            # 检查游戏是否结束
            if self.board.is_game_over():
                outcome = self.board.outcome()
                if outcome.winner is not None:
                    winner = outcome.winner
                    if is_shown:
                        winner_name = "RED" if winner == cchess.RED else "BLACK"
                        print(f"Game end. Winner is {winner_name}")
                else:
                    winner = -1
                    if is_shown:
                        print("Game end. Tie")
                return winner

    # 使用蒙特卡洛树搜索开始自我对弈，存储游戏状态（状态，蒙特卡洛落子概率，胜负手）三元组用于神经网络训练
    def start_self_play(self, player, is_shown=False, temp=1e-3):
        """
        开始自我对弈，用于收集训练数据

        Args:
            player: 自我对弈的玩家(MCTS_AI)
            is_shown: 是否显示棋盘
            temp: 温度参数，控制探索度

        Returns:
            winner: 获胜方
            play_data: 包含(state, mcts_prob, winner)的元组列表，用于训练
        """
        # 初始化棋盘
        self.board = cchess.Board()

        # 初始化数据收集列表
        states, mcts_probs, current_players = [], [], []

        # 开始自我对弈
        move_count = 0

        while True:
            move_count += 1

            # 每20步输出一次耗时
            if move_count % 20 == 0:
                start_time = time.time()
                move, move_probs = player.get_action(
                    self.board, temp=temp, return_prob=True
                )
                print(f"第{move_count}步耗时: {time.time() - start_time:.2f}秒")
            else:
                move, move_probs = player.get_action(
                    self.board, temp=temp, return_prob=True
                )

            # 保存自我对弈的数据
            current_state = decode_board(self.board)
            states.append(current_state)
            mcts_probs.append(move_probs)
            current_players.append(self.board.turn)

            # 执行一步落子
            self.board.push(cchess.Move.from_uci(move_id2move_action[move]))

            # 显示当前棋盘状态
            if is_shown:
                self.graphic(self.board)

            # 检查游戏是否结束
            if self.board.is_game_over() or is_tie(self.board):
                # 处理游戏结束情况
                outcome = self.board.outcome() if self.board.is_game_over() else None

                # 初始化胜负信息
                winner_z = np.zeros(len(current_players))

                if outcome and outcome.winner is not None:
                    winner = outcome.winner
                    # 根据胜方设置奖励
                    for i, player_id in enumerate(current_players):
                        winner_z[i] = 1.0 if player_id == winner else -1.0

                    if is_shown:
                        winner_name = "RED" if winner == cchess.RED else "BLACK"
                        print(f"Game end. Winner is: {winner_name}")
                else:
                    # 平局情况
                    winner = -1
                    if is_shown:
                        print("Game end. Tie")

                # 重置蒙特卡洛根节点
                player.reset_player()

                # 返回胜方和游戏数据
                return winner, zip(states, mcts_probs, winner_z)


# 定义整个对弈收集数据流程
class CollectPipeline:
    def __init__(self, init_model=None):
        self.board = cchess.Board()
        self.game = Game(self.board)
        # 对弈参数
        self.temp = 1  # 温度
        self.n_playout = PLAYOUT  # 每次移动的模拟次数
        self.c_puct = C_PUCT
        self.buffer_size = BUFFER_SIZE  # 经验池大小
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iters = 0
        self.init_model = init_model  # 初始模型路径

    # 从主体加载模型
    def load_model(self):
        try:
            self.policy_value_net = PolicyValueNet(model_file=MODEL_PATH)
            print("已加载最新模型")
        except:
            self.policy_value_net = PolicyValueNet()
            print("已加载初始模型")
        self.mcts_ai = MCTS_AI(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            is_selfplay=True,
        )

    def mirror_data(self, play_data):
        """左右对称变换，扩充数据集一倍，加速一倍训练速度"""
        mirror_data = []
        # 棋盘形状 [15, 10, 9], 走子概率，赢家
        for state, mcts_prob, winner in play_data:
            # 原始数据
            mirror_data.append(zip_state_mcts_prob((state, mcts_prob, winner)))
            # 水平翻转后的数据
            state_flip = state.transpose([1, 2, 0])[:, ::-1, :].transpose([2, 0, 1])
            mcts_prob_flip = copy.deepcopy(mcts_prob)
            for i in range(len(mcts_prob_flip)):
                # 水平翻转后，走子概率也需要翻转
                mcts_prob_flip[i] = mcts_prob[
                    move_action2move_id[flip(move_id2move_action[i])]
                ]
            mirror_data.append(
                zip_state_mcts_prob((state_flip, mcts_prob_flip, winner))
            )
        return mirror_data

    def collect_data(self, n_games=1):
        """收集自我对弈的数据"""
        for i in range(n_games):
            self.load_model()  # 从本体处加载最新模型
            winner, play_data = self.game.start_self_play(
                self.mcts_ai, is_shown=False
            )  # 开始自我对弈
            play_data = list(play_data)  # 转换为列表
            self.episode_len = len(play_data)  # 记录每盘对局长度
            # 增加数据
            play_data = self.mirror_data(play_data)
            if os.path.exists(DATA_BUFFER_PATH):
                while True:
                    try:
                        with open(DATA_BUFFER_PATH, "rb") as data_dict:
                            data_file = pickle.load(data_dict)
                            self.data_buffer = deque(maxlen=self.buffer_size)
                            self.data_buffer.extend(data_file["data_buffer"])
                            self.iters = data_file["iters"]
                            del data_file
                            self.iters += 1
                            self.data_buffer.extend(play_data)
                        print("成功载入数据")
                        break
                    except:
                        time.sleep(30)
            else:
                self.data_buffer.extend(play_data)
                self.iters += 1
            data_dict = {"data_buffer": self.data_buffer, "iters": self.iters}
            with open(DATA_BUFFER_PATH, "wb") as data_file:
                pickle.dump(data_dict, data_file)
        return self.iters

    def run(self):
        """开始收集数据"""
        try:
            while True:
                iters = self.collect_data()
                print("batch i: {}, episode_len: {}".format(iters, self.episode_len))
        except KeyboardInterrupt:
            print("\n\rquit")


collecting_pipeline = CollectPipeline(init_model="current_policy.pkl")
collecting_pipeline.run()
