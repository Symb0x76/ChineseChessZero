import cchess
import numpy as np
import os
import sys
import time
from rich.console import Console
from parameters import LOG_LEVEL

_console: Console | None = None


def log(message: str, level: str = "INFO", log_path: str | None = None):
    """Print a log line using Rich and persist to a .log file.

    - Console output is filtered by parameters.LOG_LEVEL (numeric 1..5).
    - File output is always written for all levels.
    - File path rule: <log_path or project_root/logs>/<script_basename>.log
    """
    global _console
    if _console is None:
        _console = Console()
    console = _console

    # Normalize level and styles
    lvl = (level or "INFO").upper()
    style_map = {
        "DEBUG": "dim",
        "INFO": "cyan",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold red",
    }
    level_rank = {"DEBUG": 1, "INFO": 2, "WARNING": 3, "ERROR": 4, "CRITICAL": 5}

    # Console threshold from parameters (numeric only), clamp 1..5
    try:
        console_threshold = int(LOG_LEVEL)
    except Exception:
        console_threshold = 2
    console_threshold = max(1, min(5, console_threshold))
    current_rank = level_rank.get(lvl, 2)

    # Build file path with default project_root/logs
    try:
        script_path = os.path.abspath(sys.argv[0] or "app")
        script_base = os.path.splitext(os.path.basename(script_path))[0] or "app"
        project_root = os.path.dirname(os.path.abspath(__file__))
        default_dir = os.path.join(project_root, "logs")
        target_dir = log_path or default_dir
        os.makedirs(target_dir, exist_ok=True)
        file_path = os.path.join(target_dir, f"{script_base}.log")
    except Exception:
        # Fallback to CWD/logs/app.log, then CWD/app.log
        try:
            fallback_dir = os.path.join(os.getcwd(), "logs")
            os.makedirs(fallback_dir, exist_ok=True)
            file_path = os.path.join(fallback_dir, "app.log")
        except Exception:
            file_path = os.path.join(os.getcwd(), "app.log")

    # Always write to file (all levels)
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"{ts} | {lvl:<8} | {message}\n")
    except Exception:
        console.log(f"Failed to write to {file_path}", style="red")

    # Console output, filtered by LOG_LEVEL
    if current_rank >= console_threshold:
        console.log(f"[{lvl}] {message}", style=style_map.get(lvl, "cyan"))


def decode_board(board):
    """
    将棋盘状态转换为神经网络输入格式的一层

    参数:
        board: cchess.Board对象

    返回:
        两个形状均为 [7, 10, 9] 的numpy数组, 分别代表红方和黑方棋子
    """
    # 初始化两个全零数组，各7个通道（7种棋子），分别表示红方和黑方
    red_state = np.zeros((7, 10, 9), dtype=np.int8)
    black_state = np.zeros((7, 10, 9), dtype=np.int8)

    # 遍历棋盘上的每个位置
    for i in range(10):
        for j in range(9):
            square = j + i * 9
            piece = board.piece_at(square)
            # print(piece)
            if piece:
                # 获取棋子类型和颜色
                piece_type = piece.piece_type
                color = piece.color

                # 根据棋子类型和颜色设置对应通道的值
                channel_idx = piece_type - 1
                if color == cchess.RED:
                    red_state[channel_idx, i, j] = 1
                else:
                    black_state[channel_idx, i, j] = 1

    return red_state, black_state


def is_tie(board):
    """
    判断游戏是否平局

    参数:
        board: cchess.Board对象

    返回:
        True 如果游戏结束且平局，否则 False
    """
    return (
        board.is_insufficient_material()
        or board.is_fourfold_repetition()
        or board.is_sixty_moves()
    )


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


# 走子翻转的函数，用来扩充我们的数据
def flip(string):
    """
    翻转棋盘走法字符串

    参数:
        string: 棋盘走法字符串

    返回:
        翻转后的棋盘走法字符串
    """
    # 定义翻转映射
    flip_map_dict = {
        "a": "i",
        "b": "h",
        "c": "g",
        "d": "f",
        "e": "e",
        "f": "d",
        "g": "c",
        "h": "b",
        "i": "a",
    }

    # 使用列表推导式进行翻转
    flip_str = "".join(
        [
            flip_map_dict[string[index]] if index in [0, 2] else string[index]
            for index in range(4)
        ]
    )

    return flip_str


# print(flip_map("d9e8"))  # 输出: f9e8
# 拿到所有合法走子的集合，2086长度，也就是神经网络预测的走子概率向量的长度
# 第一个字典：move_id到move_action
# 第二个字典：move_action到move_id
# 例如：move_id:0 --> move_action:'a0a1' 即红车上一步
def get_all_legal_moves():
    _move_id2move_action = {}
    _move_action2move_id = {}
    column = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    row = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    # 士的全部走法
    advisor_labels = [
        "d0e1",
        "e1d0",
        "f0e1",
        "e1f0",
        "d2e1",
        "e1d2",
        "f2e1",
        "e1f2",
        "d9e8",
        "e8d9",
        "f9e8",
        "e8f9",
        "d7e8",
        "e8d7",
        "f7e8",
        "e8f7",
    ]
    # 象的全部走法
    bishop_labels = [
        "a2c0",
        "c0a2",
        "a2c4",
        "c4a2",
        "c0e2",
        "e2c0",
        "c4e2",
        "e2c4",
        "e2g0",
        "g0e2",
        "e2g4",
        "g4e2",
        "g0i2",
        "i2g0",
        "g4i2",
        "i2g4",
        "a7c5",
        "c5a7",
        "a7c9",
        "c9a7",
        "c5e7",
        "e7c5",
        "c9e7",
        "e7c9",
        "e7g5",
        "g5e7",
        "e7g9",
        "g9e7",
        "g5i7",
        "i7g5",
        "g9i7",
        "i7g9",
    ]
    idx = 0
    for l1 in range(10):
        for n1 in range(9):
            destinations = (
                [(t, n1) for t in range(10)]
                + [(l1, t) for t in range(9)]
                + [
                    (l1 + a, n1 + b)
                    for (a, b) in [
                        (-2, -1),
                        (-1, -2),
                        (-2, 1),
                        (1, -2),
                        (2, -1),
                        (-1, 2),
                        (2, 1),
                        (1, 2),
                    ]
                ]
            )  # 马走日
            for l2, n2 in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(10) and n2 in range(9):
                    action = column[n1] + row[l1] + column[n2] + row[l2]
                    _move_id2move_action[idx] = action
                    _move_action2move_id[action] = idx
                    idx += 1

    for action in advisor_labels:
        _move_id2move_action[idx] = action
        _move_action2move_id[action] = idx
        idx += 1

    for action in bishop_labels:
        _move_id2move_action[idx] = action
        _move_action2move_id[action] = idx
        idx += 1

    # print(idx)  # 2086
    return _move_id2move_action, _move_action2move_id


move_id2move_action, move_action2move_id = get_all_legal_moves()
