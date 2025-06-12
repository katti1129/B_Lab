from ultralytics import YOLO

def main():
    # ダウンロードしたデータセットのパス
    data_path = r'C:\Users\cpsla\PycharmProjects\yolov8\Dataset\final.v4i.yolov8\data.yaml'  # Windowsのパスなので、\を\\またはr'パス'として指定

    # YOLOv8モデルのインスタンス化
    # 'yolov8x.yaml' から新たにトレーニングを開始する場合
    # 事前学習済みモデルを使いたい場合は 'yolov8x.pt' を使用
    model = YOLO('yolov8x.pt')  # 事前学習済みモデルを使用

    # データセットのトレーニング
    model.train(data=data_path, epochs=1000, imgsz=512, batch=16, patience=50, optimizer='Adam',lr0=0.001)

if __name__ == '__main__':
    main()