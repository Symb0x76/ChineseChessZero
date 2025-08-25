import pickle
from torch.utils.data import Dataset
import time

class PickleDataset(Dataset):
    """基于pickle的象棋数据集类（优化版）"""

    def __init__(self, pkl_path="data_optimized.pkl", max_items=None):
        print(f"[{time.strftime('%H:%M:%S')}] 开始加载 {pkl_path}...")

        # 加载pickle数据
        with open(pkl_path, "rb") as f:
            data_dict = pickle.load(f)

        # 获取数组引用（避免复制）
        self.states = data_dict['states']
        self.mcts_probs = data_dict['mcts_probs']
        self.winners = data_dict['winners']
        self.length = data_dict['total_count']

        # 限制数据量（如果指定）
        if max_items and max_items < self.length:
            self.length = max_items
            print(f"[{time.strftime('%H:%M:%S')}] 限制数据集大小为 {max_items} 条")

        print(f"[{time.strftime('%H:%M:%S')}] 数据加载完成，总共 {self.length} 步棋")
        #print(f"数据格式:")
        #print(f"  - state shape: {self.states.shape}")
        #print(f"  - mcts_prob shape: {self.mcts_probs.shape}")
        #print(f"  - winner shape: {self.winners.shape}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError(f"索引 {idx} 超出范围 (0-{self.length-1})")

        # 直接从numpy数组中获取数据（速度快）
        state = self.states[idx]
        mcts_prob = self.mcts_probs[idx]
        winner = self.winners[idx]
        return (state, mcts_prob, winner)

# 测试代码
if __name__ == "__main__":
    # 测试数据集
    dataset = PickleDataset("data_optimized.pkl")
    print(f"\n数据集长度: {len(dataset)}")

    # 测试第一个数据
    state, mcts_prob, winner = dataset[0]
    print(f"第一个样本:")
    print(f"  - state type: {type(state)}, shape: {state.shape}")
    print(f"  - mcts_prob type: {type(mcts_prob)}, shape: {mcts_prob.shape}")
    print(f"  - winner: {winner}")

    # 测试DataLoader兼容性和速度
    from torch.utils.data import DataLoader

    print(f"\n测试单进程DataLoader速度...")
    start_time = time.time()

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=False)

    batch_count = 0
    for i, (states, mcts_probs, winners) in enumerate(dataloader):
        batch_count += 1
        if i == 0:  # 只打印第一个批次的信息
            print(f"批次 {i}: states={states.shape}, mcts_probs={mcts_probs.shape}, winners={winners.shape}")
        if i >= 100:  # 测试前100个批次
            break

    elapsed_time = time.time() - start_time
    print(f"处理了 {batch_count} 个批次，用时 {elapsed_time:.2f}秒")
    print(f"平均每批次用时: {elapsed_time/batch_count*1000:.2f}ms")

    print("测试完成!")
