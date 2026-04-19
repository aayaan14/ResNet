from .resnet import resnet18, resnet50

MODEL_REGISTRY = {
    "resnet18": resnet18,
    "resnet50": resnet50,
}


def build_model(arch: str, num_classes: int = 10):
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture '{arch}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[arch](num_classes=num_classes)