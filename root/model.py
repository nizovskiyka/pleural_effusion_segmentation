import torch
import segmentation_models_pytorch as smp

def build_model(cfg: object) -> object:
    '''Create model'''
    model = smp.Unet(
        encoder_name=cfg.backbone,
        encoder_weights="imagenet",
        in_channels=1,
        classes=cfg.num_classes,
        activation=None,
    )
    model.to(cfg.device)
    return model

def load_model(model_path: str, cfg: object) -> object:
    '''Load model from checkpoint'''
    model = build_model(cfg)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
