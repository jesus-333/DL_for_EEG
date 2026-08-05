"""

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>

"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import torch

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class channel_encoder(torch.nn.Module) :
    def __init__(self) :
        super().__init__()
