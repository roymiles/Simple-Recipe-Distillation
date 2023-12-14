from torch.nn import SyncBatchNorm
from torchvision import models
from torchdistill.models.resnet18_wn import resnet18_wn
from torchdistill.models.mobilenet import MobileNetCustom, gen_mbnv2

OFFICIAL_MODEL_DICT = dict()
OFFICIAL_MODEL_DICT.update(models.__dict__)
OFFICIAL_MODEL_DICT.update(models.detection.__dict__)
OFFICIAL_MODEL_DICT.update(models.segmentation.__dict__)

# custom weight normalisation student
# OFFICIAL_MODEL_DICT['resnet18_wn'] = resnet18_wn
# OFFICIAL_MODEL_DICT['mobilenet_v2_custom'] = MobileNetCustom


def get_image_classification_model(model_config, distributed=False):
    model_name = model_config['name']
    quantized = model_config.get('quantized', False)

    # print(models.__dict__.keys())
    # # # exit()
    # # print( models.mobilenet)
    # # exit()
    # # models.resnet18_wn = resnet18_wn
    # # print(models.resnet18_wn)
    # # exit()
    # print(model_name)
    # exit()
    models.mobilenet_v2_custom = gen_mbnv2

    if not quantized and model_name in models.__dict__:
        model = models.__dict__[model_name](**model_config['params'])
    elif quantized and model_name in models.quantization.__dict__:
        model = models.quantization.__dict__[model_name](**model_config['params'])
    else:
        return None

    sync_bn = model_config.get('sync_bn', False)
    if distributed and sync_bn:
        model = SyncBatchNorm.convert_sync_batchnorm(model)
    return model


def get_object_detection_model(model_config):
    model_name = model_config['name']
    if model_name not in models.detection.__dict__:
        return None
    return models.detection.__dict__[model_name](**model_config['params'])


def get_semantic_segmentation_model(model_config):
    model_name = model_config['name']
    if model_name not in models.segmentation.__dict__:
        return None
    return models.segmentation.__dict__[model_name](**model_config['params'])


def get_vision_model(model_config):
    model_name = model_config['name']
    return OFFICIAL_MODEL_DICT[model_name](**model_config['params'])
