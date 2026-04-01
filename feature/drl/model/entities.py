from dataclasses import dataclass

@dataclass(frozen=True)
class LayerSpec:
    input_dim: int
    output_dim: int