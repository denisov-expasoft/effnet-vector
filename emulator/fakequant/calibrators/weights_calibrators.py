from typing import Tuple

from emulator.fakequant.calibrators.base import ChannelMinMaxCalibrator
from emulator.fakequant.calibrators.base import MinMaxCalibrator
from emulator.fakequant.calibrators.base import TAxis
from emulator.fakequant.calibrators.base import WEIGHTS_CALIBRATORS_REGISTRY

WEIGHTS_CALIBRATORS_REGISTRY['min_max'] = MinMaxCalibrator
WEIGHTS_CALIBRATORS_REGISTRY['min_max_depthwise'] = MinMaxCalibrator
WEIGHTS_CALIBRATORS_REGISTRY['min_max_filterwise'] = ChannelMinMaxCalibrator


@WEIGHTS_CALIBRATORS_REGISTRY.add_item_decorator('min_max_filter_depthwise')
class DepthwiseMinMaxWeightsBiasCalibrator(MinMaxCalibrator):

    def _get_axis(self, data_shape: Tuple[int, ...]) -> TAxis:
        number_of_dims = len(data_shape)
        channels_dim = number_of_dims - 2

        return tuple(
            dim
            for dim in range(number_of_dims)
            if dim != channels_dim
        )
