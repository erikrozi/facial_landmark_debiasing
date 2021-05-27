from typing import Dict, Iterable, Callable
import torch
from torch import nn, Tensor

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = list(self.pretrained._modules.keys())
        self._features = {layer: torch.empty(0) for layer in self.layers}

        for layer_id in self.layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.forward_hook(layer_id))

    def forward_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._features
