import cchess
import time
import numpy as np
from IPython.display import display, SVG
from tools import decode_board, move_id2move_action, is_tie


class Game(object):
    """
    在cchess.Board类基础上定义Game类, 用于启动并控制一整局对局的完整流程,
    收集对局过程中的数据，以及进行棋盘的展示
    """

    def __init__(self, board):
        self.board = board

    # 可视化棋盘
    def graphic(self, board):
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] player1 take: ",
            "RED" if cchess.RED else "BLACK",
        )
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] player0 take: ",
            "BLACK" if cchess.RED else "RED",
        )
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
                        print(
                            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Game end. Winner is {winner_name}"
                        )
                else:
                    winner = -1
                    if is_shown:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Game end. Tie")
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
                print(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 第{move_count}步耗时: {time.time() - start_time:.2f}秒"
                )
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
                        print(
                            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Game end. Winner is: {winner_name}"
                        )
                else:
                    # 平局情况
                    winner = -1
                    if is_shown:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Game end. Tie")

                # 重置蒙特卡洛根节点
                player.reset_player()

                # 返回胜方和游戏数据
                return winner, zip(states, mcts_probs, winner_z)
