from ultralytics import YOLO

# 加载模型
if __name__ == '__main__':
    model = YOLO('填写yaml文件路径')
    model.load('yolo的预训练权重')
    model.train(data='数据集yaml', epochs=100,  device=0,workers=2,
                batch=32,) #其他参数自己调整
