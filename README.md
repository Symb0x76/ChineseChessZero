# ChineseChessZero

这是一个基于 AlphaZero 算法的中国象棋 ai，使用标准 uci 协议

![GPL-3.0](https://img.shields.io/github/license/Symb0x76/ChineseChessZero?style=plastic)
![Python 3.13](https://img.shields.io/badge/Python-3.13-blue?style=plastic)

## TODO

-   [ ] 实现网页人机对下互动 GUI
-   [ ] 优化参数

## 使用方法

1. 下载本项目代码

```git
git clone https://github.com/Symb0x76/ChineseChessZero.git
```

2. 按照 [python-chinese-chess](https://github.com/windshadow233/python-chinese-chess) 的说明安装依赖 cchess

3. 按照 [Pytorch](https://pytorch.org) 的说明安装 torch

4. 安装其他依赖库

```pip
pip install -r requirements.txt
```

5. 运行 collect.py

```python
python collect.py
```

可以添加参数` --show`来可视化对弈过程

6. 运行 convert.py 将高压缩的 h5 数据转换为优化的 pickle 数据
```python
python convert.py
```

7. 运行 train.py

```python
python train.py
```

## 参考

-   [python-chinese-chess](https://github.com/windshadow233/python-chinese-chess)
-   [AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku)
-   [aichess](https://github.com/tensorfly-gpu/aichess)
