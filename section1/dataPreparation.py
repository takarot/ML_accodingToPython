# データの準備

import matplotlib.pyplot as plt
import numpy as np

# 以下　必要なものの定義

# x軸の定義

x_max = 1
x_min = -1

# y軸の定義

y_max = 2
y_min = -1

# スケール，１単位に何店を使うか
SCALE = 50

# train/test で Testデータの割合を指定
TEST_RATE = 0.3


# data create.

data_x = np.arange(x_min, x_max, 1/float(SCALE)).reshape(-1, 1)

data_ty = data_x ** 2
data_vy = data_ty + np.random.randn(len(data_ty), 1) * 0.5  # ノイズ乗っける


# 学習データ/テストデータに分類(分類，回帰問題で使用)

# 学習データ/テストデータに分割
def split_train_test(array):
    length = len(array)
    n_train = int(length * (1 - TEST_RATE))

    indices = list(range(length))
    np.random.shuffle(indices)
    idx_train = indices[:n_train]
    idx_test = indices[n_train:]

    return sorted(array[idx_train]), sorted(array[idx_test])

# インデックスリストを分割

indices = np.arange(len(data_x)) # インデックス値のリスト
idx_train, idx_test = split_train_test(indices)

# 学習データ
x_train = data_x[idx_train]
y_train = data_vy[idx_train]

# テストデータ
x_test = data_x[idx_test]
y_test = data_vy[idx_test]


# グラフ描画

# 分析対象点の散布図
plt.scatter(data_x, data_vy, label='target')

# 元の線を表示
plt.plot(data_x, data_ty, linestyle=':', label='non noise curve')

# x軸 y軸の範囲指定
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# 凡例の位置を指定
plt.legend(loc='best')

plt.show()

