import cchess
import threading
import time
import webbrowser
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn


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


def create_window_visualization(host="0.0.0.0", port=8000):
    """创建基于纯HTTP的棋盘可视化服务器，避免使用WebSocket"""
    try:
        # 存储当前棋盘状态的全局变量
        current_svg = ""
        status_text = ""
        last_update_time = time.time()

        # 线程化HTTP服务器，支持并发连接
        class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
            """处理请求的线程化版本"""

            daemon_threads = True

        class ChessHTTPHandler(SimpleHTTPRequestHandler):
            """处理棋盘HTTP请求的处理程序"""

            def do_GET(self):
                """处理GET请求"""
                # 根据路径提供不同的响应
                if self.path == "/" or self.path == "/index.html":
                    # 主页，提供棋盘界面
                    self.send_html_response()
                elif self.path == "/board":
                    # 棋盘数据API，提供JSON格式的当前棋盘状态
                    self.send_board_data()
                elif self.path.startswith("/poll"):
                    # 轮询端点，使用长轮询或条件请求减少刷新频率
                    self.handle_polling()
                else:
                    # 404 - 找不到请求的资源
                    self.send_error(404, "Resource not found")

            def send_html_response(self):
                """发送主HTML页面"""
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <title>Chinese Chess</title>
                    <style>
                        body {{ 
                            display: flex; 
                            flex-direction: column; 
                            align-items: center;
                            font-family: Arial, sans-serif; 
                            background-color: #f5f5f5; 
                        }}
                        .board-container {{ 
                            margin: 20px;
                            background-color: #f2d16b;
                            border: 2px solid #8b4513;
                            padding: 10px;
                            border-radius: 8px;
                            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
                        }}
                        .board {{ 
                            width: 500px;
                            height: 550px;
                            margin: 10px;
                        }}
                        .status {{ 
                            font-weight: bold; 
                            font-size: 16px; 
                            margin: 10px;
                            text-align: center;
                        }}
                    </style>
                    <script>
                        // 轮询间隔（毫秒）
                        const POLL_INTERVAL = 1000;
                        // 上次更新时间
                        let lastUpdateTime = 0;

                        // 获取棋盘更新
                        function pollBoardUpdates() {{
                            fetch('/poll?since=' + lastUpdateTime)
                                .then(response => response.json())
                                .then(data => {{
                                    if (data.updated) {{
                                        // 棋盘有更新，更新界面
                                        document.getElementById('board').innerHTML = data.svg;
                                        document.getElementById('status').innerText = data.status;
                                        lastUpdateTime = data.timestamp;
                                    }}
                                    // 继续轮询
                                    setTimeout(pollBoardUpdates, POLL_INTERVAL);
                                }})
                                .catch(error => {{
                                    console.error('轮询出错:', error);
                                    // 发生错误时，延迟后重试
                                    setTimeout(pollBoardUpdates, POLL_INTERVAL * 2);
                                }});
                        }}

                        // 页面加载时启动轮询
                        window.addEventListener('load', () => {{
                            // 立即获取初始棋盘状态
                            fetch('/board')
                                .then(response => response.json())
                                .then(data => {{
                                    document.getElementById('board').innerHTML = data.svg;
                                    document.getElementById('status').innerText = data.status;
                                    lastUpdateTime = data.timestamp;
                                    // 开始轮询更新
                                    pollBoardUpdates();
                                }});
                        }});
                    </script>
                </head>
                <body>
                    <div class="board-container">
                        <div id="board" class="board">{current_svg}</div>
                    </div>
                    <div id="status" class="status">{status_text}</div>
                </body>
                </html>
                """
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", len(html_content.encode("utf-8")))
                self.end_headers()
                self.wfile.write(html_content.encode("utf-8"))

            def send_board_data(self):
                """发送当前棋盘数据"""
                nonlocal current_svg, status_text, last_update_time

                data = {
                    "svg": current_svg,
                    "status": status_text,
                    "timestamp": last_update_time,
                }

                json_data = json.dumps(data, ensure_ascii=False)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", len(json_data.encode("utf-8")))
                self.end_headers()
                self.wfile.write(json_data.encode("utf-8"))

            def handle_polling(self):
                """处理轮询请求，根据时间戳判断是否有更新"""
                nonlocal current_svg, status_text, last_update_time

                # 从URL解析上次更新时间
                query = self.path.split("?", 1)[1] if "?" in self.path else ""
                params = {
                    k: v[0]
                    for k, v in [
                        p.split("=") if "=" in p else (p, [""])
                        for p in query.split("&")
                        if p
                    ]
                }

                client_last_update = float(params.get("since", 0))

                # 如果客户端时间戳小于服务器时间戳，说明有更新
                has_update = client_last_update < last_update_time

                data = {
                    "updated": has_update,
                    "svg": current_svg if has_update else "",
                    "status": status_text if has_update else "",
                    "timestamp": last_update_time,
                }

                json_data = json.dumps(data, ensure_ascii=False)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", len(json_data.encode("utf-8")))
                self.end_headers()
                self.wfile.write(json_data.encode("utf-8"))

            def log_message(self, format, *args):
                """禁止输出HTTP访问日志"""
                pass

        # 棋盘窗口类，提供更新棋盘的API
        class ChessWindow:
            def __init__(self, host, port):
                self.host = host
                self.port = port
                self.server = None
                self.server_thread = None

            def start(self):
                """启动HTTP服务器"""

                def run_server():
                    self.server = ThreadedHTTPServer(
                        (self.host, self.port), ChessHTTPHandler
                    )
                    self.server.serve_forever()

                self.server_thread = threading.Thread(target=run_server, daemon=True)
                self.server_thread.start()

                print(f"HTTP服务已启动在 http://{self.host}:{self.port}/")
                if self.host == "0.0.0.0" or self.host == "127.0.0.1":
                    print(f"本地访问: http://localhost:{self.port}/")
                    print(f"局域网访问请使用：http://<你的IP地址>:{self.port}/")

                return self

            def update_board(self, svg_content, status_text_=""):
                """更新棋盘内容"""
                nonlocal current_svg, status_text, last_update_time

                # 检查并确保SVG内容可以正确嵌入HTML
                if svg_content and not svg_content.strip().startswith("<svg"):
                    # 如果提供的不是SVG内容，可能是一个错误
                    print(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 警告: 提供的内容不是SVG格式"
                    )
                else:
                    # 确保SVG有正确的viewBox属性
                    if svg_content and "viewBox" not in svg_content:
                        svg_content = svg_content.replace(
                            "<svg", '<svg viewBox="-600 -600 1200 1200"'
                        )

                current_svg = svg_content
                status_text = status_text_
                last_update_time = time.time()  # 更新时间戳

            def stop(self):
                """停止HTTP服务器"""
                if self.server:
                    self.server.shutdown()

        # 单例模式
        chess_window = None

        def get_window():
            nonlocal chess_window

            if chess_window is None:
                chess_window = ChessWindow(host, port).start()
                # 自动打开浏览器
                webbrowser.open(f"http://localhost:{port}/")

            return chess_window

        return get_window

    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 错误: {e}")

        # 回退方案
        def dummy_window():
            return None

        return dummy_window


# 创建全局窗口获取函数
get_chess_window = create_window_visualization(port=8000)
