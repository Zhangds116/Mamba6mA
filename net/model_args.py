from __future__ import annotations
import math

class ModelArgs:
    d_model: int = 64
    n_layer: int = 2
    vocab_size: int = 4
    d_state: int = 16
    expand: int = 4
    dt_rank: str = 'auto'
    d_conv: int = 4
    pad_vocab_size_multiple: int = 4
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)
