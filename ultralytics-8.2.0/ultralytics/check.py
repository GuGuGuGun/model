from copy import deepcopy
import ultralytics.utils.torch_utils as utils
import torch_pruning
import nn.tasks



def check(cfg, ch=3, nc=None, verbose=True):
    yaml = cfg if isinstance(cfg, dict) else nn.yaml_model_load(cfg,"s")
    print(yaml)
    ch = yaml.get("ch", ch)
    model, savelist = nn.parse_model(deepcopy(yaml), ch=ch, verbose=verbose)
    return model

def check_include_GFLOPs(cfg):
    nn.tasks.DetectionModel(cfg,use_scale="n")

if __name__ == '__main__':
    # model = check("D:\deeplearning/ultralytics-8.2.0/ultralytics\cfg\models/v8/test.yaml")
    Mod = check_include_GFLOPs("D:\deeplearning/ultralytics-8.2.0/ultralytics\cfg\models/v8/test.yaml")
    # Mod = check_include_GFLOPs('D:\deeplearning/ultralytics-8.2.0/ultralytics\cfg\models/v8/yolov8.yaml')
