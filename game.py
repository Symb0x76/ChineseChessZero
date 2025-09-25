import cchess
import cchess.svg
import time
import numpy as np
from IPython.display import display, SVG
from tools import decode_board, move_id2move_action, is_tie, log
from rich.progress import Progress, BarColumn, TextColumn
from frontend import get_chess_window


class Game(object):
    """
    在cchess.Board类基础上定义Game类, 用于启动并控制一整局对局的完整流程,
    收集对局过程中的数据，以及进行棋盘的展示
    """

    def __init__(self, board):
        self.board = board
        self.red_states = None
        self.black_states = None
        self.reset_states_history()

    def reset_states_history(self):
        # 初始化棋盘状态时调用
        if hasattr(self, "board") and self.board:
            init_red_state, init_black_state = decode_board(self.board)
        else:
            init_red_state = np.zeros((7, 10, 9), dtype=np.float16)
            init_black_state = np.zeros((7, 10, 9), dtype=np.float16)

        # 为每方存储最近8步的状态
        self.red_states = [init_red_state.copy() for _ in range(8)]
        self.black_states = [init_black_state.copy() for _ in range(8)]
        # print(f"red_states shape: {np.array(self.red_states).shape}")

    def update_states_history(self):
        # 获取当前棋盘状态
        red_state, black_state = decode_board(self.board)

        # 更新状态历史
        self.red_states.pop()
        self.red_states.insert(0, red_state)
        self.black_states.pop()
        self.black_states.insert(0, black_state)

    # 可视化棋盘
    def graphic(self, board):
        svg = cchess.svg.board(
            board,
            size=720,
            coordinates=True,
            axes_type=0,
            checkers=board.checkers(),
            lastmove=board.peek() if board.peek() else None,
            orientation=cchess.RED,
        )
        # 获取当前玩家

        current_player = "红方" if board.turn == cchess.RED else "黑方"
        status_text = f"当前走子: {current_player} - 步数: {len(board.move_stack)}"

        # 尝试在窗口中显示
        try:
            window = get_chess_window()
            if window:
                window.update_board(svg, status_text)
                # 给窗口一点时间更新
                time.sleep(0.01)
            else:
                # 如果窗口创建失败，回退到终端显示
                display(SVG(svg))
        except ImportError:
            # 如果无法导入窗口函数，回退到终端显示
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

            # 更新状态历史
            self.update_states_history()

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
                        log(f"Game over. Winner: {winner_name}")
                else:
                    winner = -1
                    if is_shown:
                        log("Game over. Draw")
                return winner

    # 使用蒙特卡洛树搜索开始自我对弈，存储游戏状态（状态，蒙特卡洛落子概率，胜负手）三元组用于神经网络训练
    def start_self_play(
        self, player, is_shown=False, temp=1.0, game_index: int | None = None
    ):
        """
        开始自我对弈，用于收集训练数据

        Args:
            player: 自我对弈的玩家(MCTS_AI)
            is_shown: 是否显示棋盘
            temp: 温度参数，控制探索度

        Returns:
            训练数据列表
        """
        # 初始化棋盘
        self.board = cchess.Board()
        self.reset_states_history()
        mcts_probs, current_players = [], []
        move_count = 0
        avg_step_time = 0.0

        # 开始自我对弈
        while True:
            move_count += 1

            # 动态调整温度：前30步使用较高温度增加探索，后续降低
            current_temp = temp if move_count <= 30 else max(0.1, temp * 0.5)

            # 进度条：每步都会显示 playout 进度和平均用时
            step_start_time = time.time()
            with Progress(
                TextColumn("Game: {task.fields[game]}"),
                TextColumn("| Step: {task.fields[step]}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total} playouts"),
                TextColumn("| avg: {task.fields[avg]:.2f}s/step"),
                transient=True,
            ) as progress:
                task = progress.add_task(
                    "mcts",
                    total=player.mcts.n_playout,
                    game=(game_index if game_index is not None else "-"),
                    step=move_count,
                    avg=avg_step_time if avg_step_time > 0 else 0.0,
                )
                move, move_probs = player.get_action(
                    self.board,
                    temp=current_temp,
                    return_prob=True,
                    on_playout=lambda d: progress.update(task, advance=d),
                )
            step_time = time.time() - step_start_time
            avg_step_time = (avg_step_time * (move_count - 1) + step_time) / move_count

            # 验证概率分布
            prob_sum = np.sum(move_probs)
            if prob_sum > 0:
                move_probs = move_probs / prob_sum  # 归一化确保概率和为1
            else:
                log(f"move_probs are all zero at step {move_count}", "WARNING")
                continue

            # 保存自我对弈的数据
            mcts_probs.append(move_probs)
            current_players.append(self.board.turn)
            self.update_states_history()

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
                        winner_z[i] = 1 if player_id == winner else -1

                    if is_shown:
                        winner_name = "RED" if winner == cchess.RED else "BLACK"
                        log(f"Game over. Winner: {winner_name}")
                else:
                    # 平局情况
                    winner = 0
                    if is_shown:
                        log("Game over. Draw")

                # 重置蒙特卡洛根节点
                player.reset_player()

                # 返回游戏数据
                return [
                    (self.red_states, self.black_states, mcts_probs[i], winner_z[i])
                    for i in range(len(mcts_probs))
                ]
