from darknet import Darknet
import torch
from utils import get_categories

def _get_model(cfgfile, weightfile, device, input_size, pretrained):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Darknet(cfgfile, device, input_size)
    model.eval()
    if pretrained:
        model.load_weights(weightfile)
    model = model.to(device)
    return model

def yolov2_tiny_voc(device=None, input_size=None, pretrained=True):
    model = _get_model('cfg/yolov2-tiny-voc.cfg', 'weights/yolov2-tiny-voc.weights', device, input_size, pretrained)
    cats = get_categories('classes/voc.names')
    return model, cats
def yolov2_voc(device=None, input_size=None, pretrained=True):
    model = _get_model('cfg/yolov2-voc.cfg', 'weights/yolov2-voc.weights', device, input_size, pretrained)
    cats = get_categories('classes/voc.names')
    return model, cats
def yolov2_coco(device=None, input_size=None, pretrained=True):
    model = _get_model('cfg/yolov2.cfg', 'weights/yolov2.weights', device, input_size, pretrained)
    cats = get_categories('classes/coco.names')
    return model, cats

