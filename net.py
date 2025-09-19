import torch
import cchess
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.amp import autocast
from tools import move_action2move_id, decode_board

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda") if CUDA else torch.device("cpu")
PIECES = 7  # 每种棋子类型的通道数
PLAYS = 17  # 红方 8 步 + 黑方 8 步 + 1种走子方指示

# 残差块
class ResBlock(nn.Module):
    def __init__(self, num_channels=256):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            num_channels, num_channels, kernel_size=(3, 3), stride=(1, 1), padding=1
        )
        self.conv1_bn = nn.BatchNorm2d(
            num_channels,
        )
        self.conv1_act = nn.ReLU()
        self.conv2 = nn.Conv2d(
            num_channels, num_channels, kernel_size=(3, 3), stride=(1, 1), padding=1
        )
        self.conv2_bn = nn.BatchNorm2d(
            num_channels,
        )
        self.conv2_act = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv1_bn(y)
        y = self.conv1_act(y)
        y = self.conv2(y)
        y = self.conv2_bn(y)
        y = x + y
        y = self.conv2_act(y)
        return y


# 骨干网络
# 输入: N, PLAYS, PIECES, 10, 9 --> N, D, C, H, W
class Net(nn.Module):
    def __init__(
        self, num_channels=256, resblocks_num=40
    ):  # 40 ResBlock为 AlphaZero 论文的数值
        super(Net, self).__init__()
        self.input_channels = PLAYS * PIECES
        # 初始化特征
        self.conv_block = nn.Conv2d(
            self.input_channels,
            num_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
        )
        self.conv_block_bn = nn.BatchNorm2d(
            num_channels,
        )
        self.conv_block_act = nn.ReLU()
        self.res_blocks = nn.ModuleList(
            [ResBlock(num_channels=num_channels) for _ in range(resblocks_num)]
        )
        # 策略头
        self.policy_conv = nn.Conv2d(
            num_channels, PLAYS, kernel_size=(1, 1), stride=(1, 1)
        )
        self.policy_bn = nn.BatchNorm2d(PLAYS)
        self.policy_act = nn.ReLU()
        self.policy_fc = nn.Linear(PLAYS * 10 * 9, 2086)
        # 价值头
        self.value_conv = nn.Conv2d(num_channels, PIECES, kernel_size=(1, 1), stride=(1, 1))
        self.value_bn = nn.BatchNorm2d(PIECES)
        self.value_act1 = nn.ReLU()
        self.value_fc1 = nn.Linear(PIECES * 10 * 9, 256)
        self.value_act2 = nn.ReLU()
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        # 将形状从 [N, PLAYS, PIECES, 10, 9] 重塑为 [N, PLAYS * PIECES, 10, 9]
        x = x.view(batch_size, -1, 10, 9)

        # 公共头
        x = self.conv_block(x)
        x = self.conv_block_bn(x)
        x = self.conv_block_act(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        # 策略头
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = self.policy_act(policy)
        policy = torch.reshape(policy, [-1, PLAYS * 10 * 9])
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        # 价值头
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = self.value_act1(value)
        value = torch.reshape(value, [-1, PIECES * 10 * 9])
        value = self.value_fc1(value)
        value = self.value_act2(value)
        value = self.value_fc2(value)
        value = torch.tanh(value)

        return policy, value


class PolicyValueNet(object):
    def __init__(self, model=None, use_gpu=True):
        self.use_gpu = use_gpu
        self.l2_const = 2e-3
        self.device = (
            torch.device("cuda") if (self.use_gpu and torch.cuda.is_available()) else torch.device("cpu")
        )
        self.policy_value_net = Net().to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.policy_value_net.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.l2_const,
        )
        # 创建CUDA流用于异步操作
        self.stream = (
            torch.cuda.Stream() if self.use_gpu and torch.cuda.is_available() else None
        )
        if model:
            self.policy_value_net.load_state_dict(
                torch.load(model, map_location=self.device)
            )  # 加载模型参数并映射到当前设备

    # 输入一个批次的状态，输出一个批次的动作概率和状态价值
    def policy_value(self, state_batch):
        self.policy_value_net.eval()
        if isinstance(state_batch, torch.Tensor):
            state_batch = state_batch.to(self.device)
        else:
            state_batch = torch.tensor(state_batch, dtype=torch.float).to(self.device)
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(state_batch)
        log_act_probs, value = log_act_probs.cpu(), value.cpu()
        act_probs = np.exp(log_act_probs.detach().numpy())
        return act_probs, value.detach().numpy()

    # 输入棋盘，返回每个合法动作的（动作，概率）元组列表，以及棋盘状态的分数
    def policy_value_fn(self, board, red_states=None, black_states=None):
        self.policy_value_net.eval()
        # 获取合法动作列表
        legal_positions = [
            move_action2move_id[cchess.Move.uci(move)]
            for move in list(board.legal_moves)
        ]

        # 如果没有提供历史状态，则从当前棋盘解码
        if red_states is None or black_states is None:
            red_state, black_state = decode_board(board)
            red_states = [np.zeros((PIECES, 10, 9), dtype=np.float16) for _ in range(PIECES)] + [
                red_state
            ]
            black_states = [
                np.zeros((PIECES, 10, 9), dtype=np.float16) for _ in range(PIECES)
            ] + [black_state]

        # 添加走子方指示层
        if board.turn == cchess.RED:
            current_player = np.ones((1, PIECES, 10, 9), dtype=np.float16)
        else:
            current_player = np.zeros((1, PIECES, 10, 9), dtype=np.float16)
        states = np.concatenate((red_states, black_states, current_player), axis=0)
        current_states = np.ascontiguousarray(states.reshape(-1, PLAYS, PIECES, 10, 9)).astype(
            "float16"
        )
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                current_states = torch.as_tensor(
                    current_states, dtype=torch.float16
                ).to(self.device, non_blocking=True)
                with torch.no_grad():
                    with autocast("cuda"):
                        log_act_probs, value = self.policy_value_net(current_states)
                log_act_probs, value = log_act_probs.to(
                    "cpu", non_blocking=True
                ), value.to("cpu", non_blocking=True)
            torch.cuda.current_stream().wait_stream(self.stream)
        else:
            # 在 CPU 上避免使用 float16；仅在 CUDA 上使用 float16 与 autocast
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            current_states = torch.as_tensor(current_states, dtype=dtype).to(self.device)
            with torch.no_grad():
                if self.device.type == "cuda":
                    with autocast("cuda"):
                        log_act_probs, value = self.policy_value_net(current_states)
                else:
                    log_act_probs, value = self.policy_value_net(current_states)
            log_act_probs, value = log_act_probs.cpu(), value.cpu()
        # 只取出合法动作
        act_probs = np.exp(log_act_probs.detach().numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        # 返回动作概率，以及状态价值
        return act_probs, value.detach().numpy()

    # 保存模型
    def save_model(self, model_file):
        torch.save(self.policy_value_net.state_dict(), model_file)

    # 执行一步训练
    def train_step(self, state_batch, mcts_probs, winner_batch, lr=0.002):
        self.policy_value_net.train()
        if isinstance(state_batch, torch.Tensor):
            state_batch = state_batch.to(self.device)
        else:
            state_batch = torch.as_tensor(state_batch, dtype=torch.float).to(
                self.device
            )
        if isinstance(mcts_probs, torch.Tensor):
            mcts_probs = mcts_probs.to(self.device)
        else:
            mcts_probs = torch.as_tensor(mcts_probs, dtype=torch.float).to(self.device)
        if isinstance(winner_batch, torch.Tensor):
            winner_batch = winner_batch.to(self.device)
        else:
            winner_batch = torch.as_tensor(winner_batch, dtype=torch.float).to(
                self.device
            )
        # 清零梯度
        self.optimizer.zero_grad()
        log_act_probs, value = self.policy_value_net(state_batch)
        value = torch.reshape(value, shape=[-1])
        value_loss = F.mse_loss(input=value, target=winner_batch)
        policy_loss = -torch.mean(
            torch.sum(mcts_probs * log_act_probs, dim=1)
        )
        # 总损失
        # 注意l2惩罚已经包含在优化器内部
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, dim=1)
            )
        return loss.detach().cpu().numpy(), entropy.detach().cpu().numpy()
