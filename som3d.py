# -----------------------------------------------------------------------------
# Copyright 2019 (C) Nicolas P. Rougier
# Released under a BSD two-clauses license
#
# References: Kohonen, Teuvo. Self-Organization and Associative Memory.
#             Springer, Berlin, 1984.
# -----------------------------------------------------------------------------
# Under the above license, we have made modifications such as adding methods.
# Copyright 2023 (C) H.Yamamoto

import numpy as np
import scipy
import time
from config import norm_min, norm_max
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib import gridspec  # 追加


class SOM3D:
    """3次元の自己組織化マップ"""

    def __init__(self, x_dim, y_dim, z_dim, input_dim, sigma=1.0, learning_rate=0.5):
        """SOMの初期化"""

        # 3Dグリッドを作成
        X, Y, Z = np.meshgrid(
            np.linspace(0, 1, x_dim), np.linspace(0, 1, y_dim), np.linspace(0, 1, z_dim)
        )

        # グリッドを平坦化して3D座標のリストを作成
        self.grid_points = np.c_[X.ravel(), Y.ravel(), Z.ravel()]

        # グリッド内の各点対の距離を計算
        grid_distance = scipy.spatial.distance.cdist(self.grid_points, self.grid_points)

        # コードブック(重みベクトル)を0から1の間のランダムな値で初期化
        self.codebook = np.random.uniform(0, 1, (len(self.grid_points), input_dim))

        self.distance = grid_distance

        # 学習パラメータを設定
        self.sigma = sigma
        self.learning_rate = learning_rate

        # 次元の数
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.input_dim = input_dim

    def normalize(self, data):
        """データを0から1の範囲に正規化する"""
        return (data - norm_min) / (norm_max - norm_min)

    def denormalize(self, normalized_data):
        """正規化されたデータを元に戻す"""
        return normalized_data * (norm_max - norm_min) + norm_min

    def train(self, samples, n_epoch):
        """サンプルの学習"""
        self.codebook = self.codebook.reshape(
            self.x_dim * self.y_dim * self.z_dim, self.input_dim
        )

        # 学習率とシグマの初期化
        t = np.linspace(0, 1, n_epoch)
        lrate = self.learning_rate * (0.01 / self.learning_rate) ** t
        sigma = self.sigma * (0.01 / self.sigma) ** t

        I = np.random.randint(0, len(samples), n_epoch)
        samples = samples[I]

        for i in range(n_epoch):
            epoch_start_time = time.time()
            # ランダムなサンプルを取得
            data = samples[i]

            # 最も近いノード（最小距離）のインデックスを取得
            winner = np.argmin(((self.codebook - data) ** 2).sum(axis=-1))

            # 勝者を中心としたガウス関数
            G = np.exp(-self.distance[winner] ** 2 / sigma[i] ** 2)

            # ガウス関数に従ってノードをサンプルに近づける
            self.codebook -= lrate[i] * G[..., np.newaxis] * (self.codebook - data)
            # エポックの終了時間を記録
            epoch_end_time = time.time()

            # 残り時間を計算
            remaining_time = (epoch_end_time - epoch_start_time) * (n_epoch - i - 1)

            # 残り時間を表示
            hours, rem = divmod(remaining_time, 3600)
            minutes, seconds = divmod(rem, 60)

            # 学習の進捗状況を表示
            percentage = 100 * (i + 1) / n_epoch
            progress_bar = (
                "=" * (int(percentage // 2) - 1)
                + ">"
                + " " * (50 - int(percentage // 2))
            )
            print(
                f"\r[{progress_bar}] {percentage:.2f}% Completed. Remaining time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}",
                end="",
            )

            if (i + 1) % 1000 == 0:  # 1000エポックごとにプロット
                node_vectors = self.codebook.reshape(
                    self.x_dim, self.y_dim, self.z_dim, self.input_dim
                )
                node_vectors_denorm = self.denormalize(node_vectors)
                titles = ["theta1", "theta2", "theta3", "x", "y"]

                # 出力のプロット
                fig = plt.figure(figsize=(15, 3))  # figsizeを適当に設定

                gs = gridspec.GridSpec(
                    1, 10, width_ratios=[1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05]
                )
                ax = []

                for j, title in enumerate(titles):
                    ax.append(fig.add_subplot(gs[0, j * 2], projection="3d"))
                    colors = cm.gnuplot(
                        node_vectors_denorm[:, :, :, j].ravel() / norm_max[j]
                    )

                    X, Y, Z = np.meshgrid(
                        np.linspace(0, self.x_dim - 1, self.x_dim),
                        np.linspace(0, self.y_dim - 1, self.y_dim),
                        np.linspace(0, self.z_dim - 1, self.z_dim),
                    )

                    ax[j].scatter(X.ravel(), Y.ravel(), Z.ravel(), c=colors)
                    ax[j].set_title(f"color={title}")
                    ax[j].set_xlabel("D1")
                    ax[j].set_ylabel("D2")
                    ax[j].set_zlabel("D3")

                    norm = plt.Normalize(norm_min[j], norm_max[j])
                    sm = plt.cm.ScalarMappable(cmap=cm.gnuplot, norm=norm)
                    sm.set_array([])
                    cbar = plt.colorbar(
                        sm,
                        cax=fig.add_subplot(gs[0, j * 2 + 1]),
                        orientation="vertical",
                    )
                    cbar.set_label(f"{title} value")

                plt.savefig(f"Figures/Frames/SOM3D_output_{i+1}.png")
                plt.close(fig)

        # 学習が終わったらコードブックを元の形に戻す
        self.codebook = self.codebook.reshape(
            self.x_dim, self.y_dim, self.z_dim, self.input_dim
        )

    def winner(self, x):
        """入力ベクトルxに最も近いノード（勝者）の座標を返す"""
        distance = (
            (
                self.codebook.reshape(
                    self.x_dim * self.y_dim * self.z_dim, self.input_dim
                )
                - x
            )
            ** 2
        ).sum(axis=-1)
        winner_index = np.argmin(distance)
        x_coord, y_coord, z_coord = np.unravel_index(
            winner_index, (self.x_dim, self.y_dim, self.z_dim)
        )
        return x_coord, y_coord, z_coord

    def get_vector(self, coordinate):
        """指定された座標x,y,zが持つベクトルを返す"""
        return self.codebook[coordinate[0], coordinate[1], coordinate[2]]