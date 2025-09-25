# PUCT探索常数 score = Q(s,a) + C_PUCT * P(s,a) * √(N(s)) / (1 + N(s,a))
# Q(s,a)：状态s下采取行动a的预期价值
# P(s,a)：策略网络给出的先验概率
# N(s)：父节点的访问次数
# N(s,a)：当前节点的访问次数
# 值较小时，算法会更倾向于探索访问次数少的节点
# 值较大时，算法会更倾向于利用已知的高价值路径
C_PUCT = 5
# Dirichlet噪声的ε参数，表示添加噪声的比例或强度
EPS = 0.25
# Dirichlet噪声的α参数，表示添加噪声的分布的形状
ALPHA = 0.2
# 每次移动的模拟次数
PLAYOUT = 1600
# 数据目录（训练过程中产生的所有数据文件）
DATA_DIR = "data"
# 模型目录（训练产生与加载的模型文件）
MODEL_DIR = "models"
# 训练数据批次大小
BATCH_SIZE = 1024
# 训练轮数
EPOCHS = 10
# kl散度控制
KL_TARG = 0.02
# 保存模型的频率
CHECK_FREQ = 10
# 输出到控制台的日志等级 ({"DEBUG": 1, "INFO": 2, "WARNING": 3, "ERROR": 4, "CRITICAL": 5})
LOG_LEVEL = 3
