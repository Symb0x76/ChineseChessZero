# PUCT探索常数 score = Q(s,a) + C_PUCT * P(s,a) * √(N(s)) / (1 + N(s,a))
# Q(s,a)：状态s下采取行动a的预期价值
# P(s,a)：策略网络给出的先验概率
# N(s)：父节点的访问次数
# N(s,a)：当前节点的访问次数
# 值较大时，算法会更倾向于探索访问次数少的节点
# 值较小时，算法会更倾向于利用已知的高价值路径
C_PUCT = 5
# Dirichlet噪声的ε参数，表示添加噪声的比例或强度
EPS = 0.25
# Dirichlet噪声的α参数，表示添加噪声的分布的形状
ALPHA = 0.2
# 每次移动的模拟次数
PLAYOUT = 1200
# 经验池大小
BUFFER_SIZE = 100000
# 模型地址
MODEL_PATH = "current_policy.pkl"
# 训练数据容器地址
DATA_BUFFER_PATH = "data_buffer.pkl"
# 训练数据批次大小
BATCH_SIZE = 1024
# 训练轮数
EPOCHS = 10
# 训练更新间隔时间
UPDATE_INTERVAL = 3
# kl散度控制
KL_TARG = 0.02
# 训练更新的次数
GAME_BATCH_NUM = 3000
# 保存模型的频率
CHECK_FREQ = 10
