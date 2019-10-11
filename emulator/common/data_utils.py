__all__ = [
    'load_weights',
]

import pickle
from pathlib import Path
from typing import Union

from emulator.common.data_types import TWeightsCfg


def load_weights(weights_path: Union[str, Path]) -> TWeightsCfg:
    weights_path = Path(weights_path)

    with weights_path.open('rb') as file:
        return pickle.load(file)
