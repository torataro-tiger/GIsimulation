from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

class GhostImaging:

    def __init__(self, input_img_path: str, img_width: int, img_height: int):
        # 画像全般に関するパラメータ定義
        self.width  = img_width
        self.height = img_height
        # 入力画像初期処理
        pil_img  = Image.open(input_img_path).convert('L')   # 画像読み込む
        pil_img  = pil_img.resize((self.width, self.height)) # 画像リサイズ
        self.obj = np.array(pil_img, dtype=np.float64) / 255 # numpy配列化 & 1に正規化
        self.obj = np.where(self.obj > 0.5, 1, 0)            # 2値化
        # 計算結果初期化
        self.result_img = np.zeros((self.height, self.width))
        # 評価結果格納用
        self.evals = np.array([])
        # アニメーション用
        self.fig    = plt.figure()
        self.frames = []

    # パターン生成、照射＆光強度から再構成まで
    def simulate_gi(self, pattern_num: int, frame_num: int, use_average: bool = False):
        recon_obj = np.zeros((self.height, self.width))
        if (use_average):
            intensities = np.array([])
            pattern_sum = np.zeros((self.height, self.width))
        for count in range(pattern_num):
            number_of_patterns = count + 1
            # パターン生成
            pattern = np.where(np.random.rand(self.height, self.width) > 0.5, 1, 0)
            # パターンと撮像対象の作用
            mul_mat = self.obj * pattern
            # 1ピクセルとして受光
            intensity = np.sum(np.abs(mul_mat)**2)
            # 取得した光強度から像を再構成
            if (use_average):
                intensities = np.append(intensities, intensity)
                recon_obj, pattern_sum = self.reconstruction_with_ave(
                    intensity, intensities, pattern, pattern_sum, recon_obj
                    )
            else:
                recon_obj = self.reconstruction(intensity, pattern, recon_obj)
            # 足し合わせた結果を評価する
            MSE = self.mean_squared_error()
            self.evals = np.append(self.evals, MSE)

            if number_of_patterns % int(pattern_num / frame_num) == 0:
                print(f'number of patterns: {number_of_patterns}, MSE: {MSE:.4f}')
                frame = plt.imshow(self.result_img, cmap='binary_r', vmin=0, vmax=1)
                text  = plt.text(-0.4, -0.8, f'number of patterns = {number_of_patterns}')
                self.frames.append([frame, text]) # アニメーション用のフレーム追加
    
    # 像再構成用のメソッド
    def reconstruction(self, intensity, pattern, recon_obj):
        recon_obj += intensity * pattern
        max_val = np.amax(recon_obj) # 最大値取得
        min_val = np.amin(recon_obj) # 最小値取得
        self.result_img = (recon_obj - min_val) / (max_val - min_val) # 1に正規化して結果として格納
        return recon_obj
    
    # 像再構成用のメソッド(平均値で差っ引く)
    def reconstruction_with_ave(self, intensity, intensities, pattern, pattern_sum, recon_obj):
        intensity_ave = np.mean(intensities)
        pattern_sum += pattern
        recon_obj += intensity * pattern
        recon_obj_sub_ave = recon_obj - intensity_ave * pattern_sum # ここで差っ引く
        max_val = np.amax(recon_obj_sub_ave) # 最大値取得
        min_val = np.amin(recon_obj_sub_ave) # 最小値取得
        if max_val != min_val:
            self.result_img = (recon_obj_sub_ave - min_val) / (max_val - min_val) # 1に正規化して結果として格納
        else:
            self.result_img = recon_obj_sub_ave
        return recon_obj, pattern_sum

    # 精度評価(MSE)
    def mean_squared_error(self):
        return np.sum((self.obj - self.result_img)**2) / (self.height * self.width)
    
    # 撮像結果表示
    def show_results(self, save_path: str):
        # アニメーション保存
        ani = ArtistAnimation(self.fig, self.frames, interval=10)
        ani.save(save_path + "result_ani.gif", writer='pillow', fps=10)
        plt.close()
        # 画像表示
        plt.subplots()
        plt.imshow(self.obj, cmap='binary_r', vmin=0, vmax=1)
        plt.title('target object')
        plt.colorbar()
        plt.subplots()
        plt.imshow(self.result_img, cmap='binary_r', vmin=0, vmax=1)
        plt.title('reconstructed result')
        plt.colorbar()
        plt.subplots()
        plt.plot(np.arange(len(self.evals)) + 1, self.evals)
        plt.xlabel('number of patterns')
        plt.ylabel('MSE')
        plt.ylim(0, 0.5)
        plt.grid()
        plt.show()

def main():
    '''
    (1) 初期処理
    input_img_path: 撮像対象として入力する画像ファイルのパス
    img_width     : 撮像対象の幅（この大きさにリサイズする）
    img_height    : 撮像対象の高さ（この大きさにリサイズする）
    '''
    gi = GhostImaging(input_img_path = './GI.png', img_width = 16, img_height = 16)

    '''
    (2) ゴーストイメージング開始
    pattern_num: 使うパターンの枚数
    frame_num  : gif化する際のフレームの総数(> 1)
    use_average: 再構成時に全強度の平均値を使うか？(デフォルトはFalseで使わない)
    '''
    gi.simulate_gi(pattern_num = 100000, frame_num = 50, use_average = True)

    '''
    (3) 結果表示
    save_path: gifを保存するパス
    '''
    gi.show_results(save_path = "./")

if __name__ == "__main__":
    main()