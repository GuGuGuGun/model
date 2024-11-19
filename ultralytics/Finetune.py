from ultralytics import YOLO


if __name__ == "__main__":
    yolo = YOLO("D:\deeplearning\\ultralytics-8.2.0\\prune.pt") # 加载剪枝后的模型

    yolo.train(data='D:\\deeplearning\\DataSet\\NEU-DET\\little_augmentation\\dataset\\data.yaml', epochs=100, amp=False, workers=2,device=0) # 训练
