import numpy as np
import cchess
from parameters import C_PUCT, EPS, ALPHA
from tools import is_tie, move_id2move_action, softmax


class Node(object):
    """
    蒙特卡罗树搜索的游戏状态,记录在某一个Node节点下的状态数据,包含当前的游戏得分、当前的游戏round数、从开始到当前的执行记录。
    """

    def __init__(self, parent=None, prob=None):
        self.parent = parent  # 父节点, 只有根节点的parent = None
        self.children = {}  # 子节点词典, 合理动作及其对应的子节点
        self.value = 0  # 当前状态的价值 Q
        self.visits = 0  # 访问次数 N
        self.prob = prob  # 先验概率 P

    def is_leaf(self):
        """
        判断当前节点是否为叶子节点,即是否没有子节点
        """
        return self.children == {}

    def is_root(self):
        """
        判断当前节点是否为根节点
        """
        return self.parent is None

    def expand(self, action_priors):
        """
        通过创建新子节点来展开树

        action_priors: 一个动作及其先验概率的元组列表, 这些先验概率是根据策略函数得出的
        """
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = Node(self, prob)

    def puct_value(self, c_puct=C_PUCT):
        """
        计算PUCT值
        c_puct: PUCT探索常数
        """
        # 计算PUCT值
        if self.visits == 0:
            return float("inf")
        else:
            return self.value + c_puct * self.prob * np.sqrt(self.parent.visits) / (
                1 + self.visits
            )

    def select(self, c_puct):
        """
        在子节点中选择能够提供最大的Q+U的节点
        return: (action, next_node)的二元组
        """
        return max(
            self.children.items(), key=lambda act_node: act_node[1].puct_value(c_puct)
        )

    def update(self, leaf_value):
        """
        从叶节点评估中更新节点值
        leaf_value: 这个子节点的评估值来自当前玩家的视角
        """
        # 统计访问次数
        self.visits += 1
        # 更新Q值，取决于所有访问次数的平均树，使用增量式更新方式
        self.value += 1.0 * (leaf_value - self.value) / self.visits

    def update_recursive(self, leaf_value):
        """就像调用update()一样，但是对所有直系节点进行更新"""
        # 如果它不是根节点，则应首先更新此节点的父节点
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)


class MCTS(object):
    """
    蒙特卡罗树搜索算法
    """

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: 一个函数，接受棋盘状态作为输入，
            并输出一个包含(action, probability)元组的列表以及一个范围在[-1, 1]内的分数
            （即从当前玩家视角出发的终局得分期望值）。
        c_puct: 一个位于(0, inf)区间内的数值，用于控制探索向最大价值策略收敛的速度。
            该值越高，表示越依赖于先验知识。
        """
        self.root = Node(None, 1.0)  # 根节点
        self.policy = policy_value_fn
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.red_history = None
        self.black_history = None

    def playout(self, board, red_states=None, black_states=None):
        """
        从根节点开始进行一次模拟, 直到到达叶子节点, 并返回叶子节点的评估值
        """
        node = self.root
        while 1:
            if node.is_leaf():
                break
            # 贪婪选择下一步
            action, node = node.select(self.c_puct)
            board.push(cchess.Move.from_uci(move_id2move_action[action]))

        # 使用网络评估叶子节点，网络输出（动作，概率）元组p的列表以及当前玩家视角的得分[-1, 1]
        action_probs, leaf_value = self.policy(board, red_states, black_states)
        # 查看游戏是否结束
        end = board.is_game_over()
        if not end and not is_tie(board):
            # 如果游戏没有结束，且不是平局，则展开子节点
            node.expand(action_probs)
        elif end and is_tie(board):
            # 如果游戏结束且平局，则将叶子节点的值设置为0
            leaf_value = 0.0
        else:
            # 如果游戏结束且不是平局，则将叶子节点的值设置为1或-1
            winner = cchess.RED if board.outcome().winner else cchess.BLACK
            leaf_value = 1.0 if winner == board.turn else -1.0
        # 在本次遍历中更新节点的值和访问次数
        # 必须添加符号，因为两个玩家共用一个搜索树
        node.update_recursive(-leaf_value)

    def get_move_probs(
        self,
        board,
        temp=1e-3,
        red_states=None,
        black_states=None,
        on_playout=None,
    ):
        """
        按顺序运行所有搜索并返回可用的动作及其相应的概率

        state: 当前棋盘状态
        temp: 控制动作概率的参数,当temp接近0时,选择概率最高的动作,当temp接近无穷大时,选择概率接近均匀分布的动作
        """
        self.red_history = red_states
        self.black_history = black_states
        for _ in range(self.n_playout):
            board_copy = board.copy()
            self.playout(board_copy, red_states=red_states, black_states=black_states)
            if on_playout is not None:
                try:
                    on_playout(1)
                except Exception:
                    pass

        # 跟据根节点处的访问计数来计算移动概率
        act_visits = [(act, node.visits) for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        """
        更新树以反映当前移动
        """
        if last_move in self.root.children:
            # 如果移动在子节点中，则将其设置为根节点
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            # 如果移动不在子节点中，则创建一个新的根节点
            self.root = Node(None, 1.0)


class MCTS_AI(object):
    """
    基于MCTS的AI玩家

    Args:
        policy_value_fn: 策略价值函数, 输入棋盘状态, 输出动作及其概率的列表和当前玩家视角的得分[-1, 1]
        c_puct: PUCT探索常数
        n_playout: 进行模拟的次数
        is_selfplay: 是否为自我对弈
    """

    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000, is_selfplay=False):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)
        self.is_selfplay = is_selfplay
        self.agent = "AI"

    def set_player_idx(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=False, on_playout=None):
        """
        获取AI的动作

        board: 当前棋盘状态
        temp: 控制动作选择的确定性程度。温度越低(如默认的1e-3),算法越倾向于选择最高概率的动作;温度越高,选择更加随机
        return_prob: 是否返回动作概率
        """
        # 动作空间大小
        move_probs = np.zeros(2086)

        acts, probs = self.mcts.get_move_probs(board, temp, on_playout=on_playout)
        move_probs[list(acts)] = probs
        if self.is_selfplay:
            # 添加Dirichlet Noise进行探索（自我对弈需要）
            move = np.random.choice(
                acts,
                p=(1 - EPS) * probs
                + EPS * np.random.dirichlet(ALPHA * np.ones(len(probs))),
            )
            # 更新根节点并重用搜索树
            self.mcts.update_with_move(move)
        else:
            # 使用默认的temp=1e-3，几乎相当于选择具有最高概率的移动
            move = np.random.choice(acts, p=probs)
            # 重置根节点
            self.mcts.update_with_move(-1)
        if return_prob:
            return move, move_probs
        else:
            return move
