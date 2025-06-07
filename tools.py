import cchess


def decode_board(board):
    """
    将棋盘状态转换为神经网络输入格式

    参数:
        board: cchess.Board对象

    返回:
        一个形状为 [channels, height, width] 的numpy数组
    """
    # 初始化一个全零数组，15个通道（7种棋子×2方 + 1个当前玩家指示器）
    state = np.zeros((15, 10, 9), dtype=np.int8)

    # 遍历棋盘上的每个位置
    for i in range(10):
        for j in range(9):
            square = j + i * 9  # 使用正确的索引计算方式
            piece = board.piece_at(square)
            # print(piece)
            if piece:
                # 获取棋子类型和颜色
                piece_type = piece.piece_type
                color = piece.color

                # 设置对应通道的值为1
                # 红方棋子在通道0-6，黑方棋子在通道7-13
                channel_idx = piece_type - 1
                if color == cchess.BLACK:
                    channel_idx += 7

                state[channel_idx, i, j] = 1.0

    # 设置当前玩家指示器
    if board.turn == cchess.RED:
        state[14, :, :] = 1
    elif board.turn == cchess.BLACK:
        state[14, :, :] = 0

    return state


# print(decode_board(board))


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


# print(is_tie(board))


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


import numpy as np


def zip_array(array, data=0.0):
    """
    将数组压缩为稀疏数组格式

    参数:
        array: 二维numpy数组
        data: 需要压缩的值, 默认为0.

    返回:
        压缩后的列表
    """
    rows, cols = array.shape
    zip_res = [[rows, cols]]

    for i in range(rows):
        for j in range(cols):
            if array[i, j] != data:
                zip_res.append([i, j, array[i, j]])

    return zip_res  # 直接返回列表，不转换为numpy数组


def recovery_array(array, data=0.0):
    """
    从稀疏数组恢复为二维数组

    参数:
        array: 压缩后的列表或numpy数组
        data: 填充的默认值, 默认为0.

    返回:
        恢复后的二维numpy数组
    """
    # 将array转换为列表进行操作，确保兼容性
    array_list = array.tolist() if isinstance(array, np.ndarray) else array

    rows, cols = array_list[0]
    recovery_res = np.full((int(rows), int(cols)), data)

    for i in range(1, len(array_list)):
        row_idx = int(array_list[i][0])
        col_idx = int(array_list[i][1])
        recovery_res[row_idx, col_idx] = array_list[i][2]

    return recovery_res


# (state, mcts_prob, winner) ((15,10,9),2086,1) => ((15,90),(2,1043),1)
def zip_state_mcts_prob(tuple):
    state, mcts_prob, winner = tuple
    state = state.reshape((15, -1))
    mcts_prob = mcts_prob.reshape((2, -1))
    state = zip_array(state)
    mcts_prob = zip_array(mcts_prob)
    return state, mcts_prob, winner


def recovery_state_mcts_prob(tuple):
    state, mcts_prob, winner = tuple
    state = recovery_array(state)
    mcts_prob = recovery_array(mcts_prob)
    state = state.reshape((15, 10, 9))
    mcts_prob = mcts_prob.reshape(2086)
    return state, mcts_prob, winner


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


def create_window_visualization():
    """创建基于Web浏览器的棋盘可视化，零依赖方案"""
    try:
        import webbrowser
        import os
        import tempfile
        import threading
        import time

        class BrowserWindow:
            def __init__(self):
                self.temp_dir = tempfile.mkdtemp(prefix="chess_viz_")
                self.html_path = os.path.join(self.temp_dir, "chess.html")
                self.current_svg = ""
                self.status_text = ""
                self.browser_opened = False

                # 创建初始HTML文件
                self._create_html()

                # 启动自动刷新线程
                self.running = True
                self.update_thread = threading.Thread(target=self._auto_update)
                self.update_thread.daemon = True
                self.update_thread.start()

            def _create_html(self):
                """创建自动刷新的HTML页面"""
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Chinese Chess</title>
                    <meta http-equiv="refresh" content="1">
                    <style>
                        body {{ display: flex; flex-direction: column; align-items: center;
                                font-family: Arial, sans-serif; background-color: #f5f5f5; }}
                        .board {{ margin: 20px; }}
                        .status {{ font-weight: bold; font-size: 16px; margin: 10px; }}
                    </style>
                </head>
                <body>
                    <div class="board">{self.current_svg}</div>
                    <div class="status">{self.status_text}</div>
                </body>
                </html>
                """
                with open(self.html_path, "w", encoding="utf-8") as f:
                    f.write(html_content)

            def _auto_update(self):
                """定期更新HTML文件"""
                last_svg = ""
                last_status = ""

                while self.running:
                    if self.current_svg != last_svg or self.status_text != last_status:
                        self._create_html()
                        last_svg = self.current_svg
                        last_status = self.status_text
                    time.sleep(0.5)

            def update_board(self, svg_content, status_text=""):
                self.current_svg = svg_content
                self.status_text = status_text

                if not self.browser_opened:
                    webbrowser.open(f"file://{self.html_path}")
                    self.browser_opened = True

        # 单例模式
        window_instance = None

        def get_window():
            nonlocal window_instance

            if window_instance is None:
                window_instance = BrowserWindow()

            return window_instance

        return get_window

    except ImportError as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 错误: {e}")

        # 回退方案
        def dummy_window():
            return None

        return dummy_window


# 创建全局窗口获取函数
get_chess_window = create_window_visualization()
