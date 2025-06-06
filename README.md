# ChineseChessZero

这是一个基于 AlphaZero 算法的中国象棋 ai，使用标准 uci 协议

![GPL-3.0](https://img.shields.io/github/license/Symb0x76/ChineseChessZero?style=plastic)
![Python 3.13](https://img.shields.io/badge/Python-3.13-blue?style=plastic)

## TODO

-   [ ] 实现网页 GUI
-   [ ] 优化参数
-   [ ] 解决 collect.py 多开显卡占用率不高的问题

## 使用方法

1.  下载本项目代码

```git
git clone https://github.com/Symb0x76/ChineseChessZero.git
```

2.  按照[python-chinese-chess](https://github.com/windshadow233/python-chinese-chess)项目的说明安装依赖库 cchess
3.  按照[Pytorch](https://pytorch.org) 官网的说明安装支持 cuda 的 torch

4.  安装其他依赖库

```pip
pip install -r requirements.txt
```

5.  运行 collect.py

```python
python collect.py
```

如果配置比较好可以选择使用解除 gil 锁的 python3.13t 并发运行

但是当前 collect.py 多开会导致 gpu 占用率降到 50%，需要至少 8 并行才能实现正提升

6.  运行 train.py

```python
python train.py
```

仅单开

## 参考

-   [python-chinese-chess](https://github.com/windshadow233/python-chinese-chess)
-   [AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku)
-   [aichess](https://github.com/tensorfly-gpu/aichess)
