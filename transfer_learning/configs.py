from pathlib import Path

import torch

from models.yolo import Model
from utils.general_methods import set_logging
from utils.google_utils import attempt_download

dependencies = ['torch', 'yaml']
set_logging()


def create(name, pretrained, channels, classes):
    config = Path(__file__).parent / 'models' / f'{name}.yaml'
    try:
        model = Model(config, channels, classes)
        if pretrained:
            fname = f'{name}.pt'
            attempt_download(fname)
            ckpt = torch.load(fname, map_location=torch.device('cpu'))
            state_dict = ckpt['model'].float().state_dict()
            state_dict = {k: v for k, v in state_dict.items() if model.state_dict()[k].shape == v.shape}
            model.load_state_dict(state_dict, strict=False)
            if len(ckpt['model'].names) == classes:
                model.names = ckpt['model'].names
        return model

    except Exception as e:
        help_url = 'https://github.com/ultralytics/yolov5/issues/36'
        s = 'Cache maybe be out of date, try force_reload=True. See %s for help.' % help_url
        raise Exception(s) from e


def yolov3(pretrained=False, channels=3, classes=80):
    return create('yolov3', pretrained, channels, classes)


def yolov3_spp(pretrained=False, channels=3, classes=80):
    return create('yolov3-spp', pretrained, channels, classes)


def yolov3_tiny(pretrained=False, channels=3, classes=80):
    return create('yolov3-tiny', pretrained, channels, classes)


def custom(path_or_model='path/to/model.pt'):
    model = torch.load(path_or_model) if isinstance(path_or_model, str) else path_or_model
    if isinstance(model, dict):
        model = model['model']  # load model

    hub_model = Model(model.yaml).to(next(model.parameters()).device)
    hub_model.load_state_dict(model.float().state_dict())
    hub_model.names = model.names
    return hub_model


if __name__ == '__main__':
    model = create(name='yolov3', pretrained=True, channels=3, classes=80)
    model = model.autoshape()

    from PIL import Image

    imgs = [Image.open(x) for x in Path('data/images').glob('*.jpg')]
    results = model(imgs)
    results.show()
    results.print()
