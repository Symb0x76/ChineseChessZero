import threading
import time
import webbrowser
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn


def create_window_visualization(host="127.0.0.1", port=8000):
    """创建基于纯HTTP的棋盘可视化服务器, 避免使用WebSocket"""
    try:
        # 存储当前棋盘状态的全局变量
        current_svg = ""
        status_text = ""
        last_update_time = time.time()

        # 线程化HTTP服务器，支持并发连接
        class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
            """处理请求的线程化版本"""

            daemon_threads = True

            def handle_error(self, request, client_address):
                pass

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
                elif self.path == "/events":
                    # 处理SSE事件流
                    self.send_sse_stream()
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
                        body {{ display: flex; flex-direction: column; align-items: center;
                            font-family: Arial, sans-serif; background-color: #f5f5f5; }}
                        .board-container {{ margin: 20px; background-color: #f2d16b; border: 2px; padding: 10px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3); }}
                        .board {{ width: 500px; display: flex; flex-wrap: wrap; justify-content: center; align-items: center; }}
                        .board img {{ width: 100%; }}
                        .status {{ font-weight: bold; font-size: 16px; margin: 10px; text-align: center; }}
                        .connection-status {{ color: #666; font-size: 12px; margin: 5px; text-align: center; }}
                    </style>
                    <script>
                        // 使用EventSource处理所有更新, 包括初始状态
                        window.addEventListener('load', () => {{
                            setupEventSource();
                        }});
                        function setupEventSource() {{
                            const statusEl = document.getElementById('connection-status');
                            statusEl.textContent = '正在连接...';
                            // 创建EventSource连接
                            const evtSource = new EventSource('/events');
                            // SVG更新事件
                            evtSource.addEventListener('svg', function(e) {{
                                const boardDiv = document.getElementById('board');
                                try {{
                                    const img = boardDiv.querySelector('#boardimg');
                                    img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(e.data);
                                }} catch (error) {{
                                console.error('EventSource error:', error);
                                boardDiv.innerHTML = '<p style="color: red;">SVG数据格式错误</p>'
                                }}
                            }});
                            // 状态更新事件
                            evtSource.addEventListener('status', function(e) {{
                                document.getElementById('status').innerText = e.data;
                            }});
                            // 连接建立
                            evtSource.onopen = function() {{
                                statusEl.textContent = '已连接, 等待棋盘更新...';
                                statusEl.style.color = 'green';
                            }};
                            // 连接错误
                            evtSource.onerror = function() {{
                                statusEl.textContent = '连接断开, 5秒后重连...';
                                statusEl.style.color = 'red';
                                evtSource.close();
                                setTimeout(setupEventSource, 5000);
                            }};
                        }};
                    </script>
                </head>
                <body>
                    <div class="board-container">
                        <div id="board" class="board">
                            <img id="boardimg"/>
                    </div>
                    <div id="status" class="status">{status_text}</div>
                    <div id="connection-status" class="connection-status">准备连接...</div>
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

                json_data = json.dumps(data)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", len(json_data.encode("utf-8")))
                self.end_headers()
                self.wfile.write(json_data.encode("utf-8"))

            def send_sse_stream(self):
                """处理SSE事件流"""
                nonlocal current_svg, status_text, last_update_time

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                # 记录当前客户端连接的初始状态
                client_last_update = 0

                try:
                    # 首次发送当前状态（确保SVG内容格式正确）
                    svg_content = fix_svg(current_svg)
                    self.wfile.write(
                        f"event: update\ndata: {last_update_time}\n\n".encode("utf-8")
                    )
                    self.wfile.write(
                        f"event: svg\ndata: {svg_content}\n\n".encode("utf-8")
                    )
                    self.wfile.write(
                        f"event: status\ndata: {status_text}\n\n".encode("utf-8")
                    )
                    self.wfile.flush()
                    client_last_update = last_update_time

                    # 持续监听更新
                    while True:
                        try:
                            # 如果有更新，发送更新事件
                            if client_last_update < last_update_time:
                                # 确保SVG内容是有效的XML
                                svg_content = fix_svg(current_svg)
                                self.wfile.write(
                                    f"event: update\ndata: {last_update_time}\n\n".encode(
                                        "utf-8"
                                    )
                                )
                                self.wfile.write(
                                    f"event: svg\ndata: {svg_content}\n\n".encode(
                                        "utf-8"
                                    )
                                )
                                self.wfile.write(
                                    f"event: status\ndata: {status_text}\n\n".encode(
                                        "utf-8"
                                    )
                                )
                                self.wfile.flush()
                                client_last_update = last_update_time

                            # 发送心跳保持连接
                            self.wfile.write(f": heartbeat\n\n".encode("utf-8"))
                            self.wfile.flush()

                            # 每5秒检查一次是否有更新
                            time.sleep(5)
                        except (BrokenPipeError, ConnectionResetError) as e:
                            print(f"[{time.strftime('%H:%M:%S')}] 客户端连接中断")
                except (
                    ConnectionResetError,
                    ConnectionAbortedError,
                    BrokenPipeError,
                    TimeoutError,
                    OSError,
                ) as e:
                    # print(f"[{time.strftime('%H:%M:%S')}] 连接已关闭")
                    # 客户端断开连接，结束事件流
                    return

            def log_message(self, format, *args):
                """禁止输出HTTP访问日志"""
                pass

            def log_error(self, format, *args):
                """禁止输出HTTP错误日志"""
                # 可以选择完全禁止，或者只输出简短的信息
                # print(f"[{time.strftime('%H:%M:%S')}] 连接错误 (这是正常的，客户端可能已关闭)")
                pass

        def fix_svg(svg_content):
            """修复SVG内容, 确保正确显示"""
            if not svg_content:
                return ""

            try:
                # 处理可能导致解析问题的特殊字符
                svg_content = (
                    svg_content.replace("&", "&")
                    .replace("<", "<")
                    .replace(">", ">")
                    .replace('"', '"')
                    .replace("'", "'")
                )

                # 还原 SVG 标签和属性
                svg_content = (
                    svg_content.replace("<svg", "<svg")
                    .replace("</svg>", "</svg>")
                    .replace("<g", "<g")
                    .replace("</g>", "</g>")
                    .replace("<path", "<path")
                    .replace("</path>", "</path>")
                    .replace("<circle", "<circle")
                    .replace("</circle>", "</circle>")
                    .replace("<rect", "<rect")
                    .replace("</rect>", "</rect>")
                    .replace("<text", "<text")
                    .replace("</text>", "</text>")
                    .replace("<defs", "<defs")
                    .replace("</defs>", "</defs>")
                    .replace("<use", "<use")
                    .replace("</use>", "</use>")
                )

                # 移除XML声明避免解析出错
                svg_content = svg_content.replace(
                    '<?xml version="1.0" encoding="UTF-8"?>', ""
                )

                # 确保SVG有所有必要的命名空间
                if 'xmlns="http://www.w3.org/2000/svg"' not in svg_content:
                    svg_content = svg_content.replace(
                        "<svg", '<svg xmlns="http://www.w3.org/2000/svg"'
                    )

                # 确保有xlink命名空间
                if 'xmlns:xlink="http://www.w3.org/1999/xlink"' not in svg_content:
                    svg_content = svg_content.replace(
                        "<svg", '<svg xmlns:xlink="http://www.w3.org/1999/xlink"'
                    )

                # 确保有正确的viewBox
                if 'viewBox="' not in svg_content:
                    svg_content = svg_content.replace(
                        "<svg", '<svg viewBox="-600 -600 1200 1200"'
                    )
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] SVG修复出错: {e}")
                return create_error_svg("SVG处理出错")

            return svg_content

        def create_error_svg(message):
            """创建错误提示SVG"""
            return f"""<svg 
                    xmlns="http://www.w3.org/2000/svg" 
                    width="500" 
                    height="550" 
                viewBox="0 0 500 550">
                <rect width="100%" height="100%" fill="#f8d7da"/>
                <text 
                    x="50%" 
                    y="50%" 
                    font-family="Arial" 
                    font-size="20" 
                    fill="#721c24" 
                    text-anchor="middle"> 
                    {message}
                </text>
            </svg>"""

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
                    print(f"局域网访问请使用: http://<你的IP地址>:{self.port}/")

                return self

            def update_board(self, svg_content, status_text_=""):
                """更新棋盘内容"""
                nonlocal current_svg, status_text, last_update_time

                # 从SVG对象中提取字符串内容(如果是SVG对象)
                if hasattr(svg_content, "_repr_svg_"):
                    svg_content = svg_content._repr_svg_()
                elif hasattr(svg_content, "data"):
                    # IPython SVG对象处理
                    svg_content = svg_content.data

                # 确保svg_content是字符串
                if svg_content and not isinstance(svg_content, str):
                    try:
                        svg_content = str(svg_content)
                    except Exception as e:
                        print(
                            f"[{time.strftime('%H:%M:%S')}] 警告: 无法将SVG转换为字符串: {e}"
                        )
                        svg_content = create_error_svg("SVG转换错误")

                # 检查并确保SVG内容可以正确嵌入HTML
                if svg_content and not svg_content.strip().startswith("<svg"):
                    print(f"[{time.strftime('%H:%M:%S')}] 警告: 提供的内容不是SVG格式")
                    svg_content = create_error_svg("内容非SVG格式")
                else:
                    svg_content = fix_svg(svg_content)
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
        print(f"[{time.strftime('%H:%M:%S')}] 错误: {e}")

        # 回退方案
        def dummy_window():
            return None

        return dummy_window


# 创建全局窗口获取函数
get_chess_window = create_window_visualization(port=8000)
