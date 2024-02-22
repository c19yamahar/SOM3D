# 3 次元自己組織化マップ

- [日本語](./README.md)
- [English](./README.en.md)

この Python プログラムは、Teuvo Kohonen によって 1980 年代に紹介された概念に基づいて、3 次元自己組織化マップ（SOM）を実装しています。  
SOM は人工ニューラルネットワークの一種で、教師なし学習を使用して訓練され、訓練サンプルの入力空間を低次元空間（通常は二次元）に写像します。  
この実装では、従来の SOM を 3 次元に拡張したものです。

## インストール

Python 3.9 以降をインストールしたのち、以下の手順でインストールしてください：

```bash
$ git clone git@github.com:c19yamahar/SOM3D.git
$ cd SOM3D
$ pip install -r requirements.txt
```

## 使用方法

`SOM3D`クラスを `import` したのち、希望するグリッドの次元、入力次元、学習率、シグマ値で SOM を初期化します。`train` メソッドを呼び出して SOM をデータセットでトレーニングし、`winner`メソッドを使用して与えられた入力ベクトルに対する SOM の勝者ノードを見つけます。`get_vector` メソッドを使用して特定の座標のベクトルを取得できます。

```python
import numpy as np
from som3d import SOM3D

# 10x10x10 グリッドと 3 の入力次元で SOM を初期化

som = SOM3D(x_dim=10, y_dim=10, z_dim=10, input_dim=3)

# ランダムなトレーニングデータを生成

training_data = np.random.rand(100, 3)

# SOM をトレーニング

som.train(training_data, n_epoch=1000)

# 与えられた入力ベクトルのための勝者ノードを見つける

input_vector = np.array([0.5, 0.5, 0.5])
winner_coordinates = som.winner(input_vector)
print(f"勝者ノードの座標: {winner_coordinates}")

# 特定の座標のベクトルを取得

vector = som.get_vector((5, 5, 5))
print(f"座標(5, 5, 5)でのベクトル: {vector}")
```
