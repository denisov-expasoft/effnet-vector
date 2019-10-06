__all__ = ['SupportedLayerTypes']

from enum import Enum
from enum import unique


@unique
class SupportedLayerTypes(Enum):
    LAYER_UNKNOWN = "unknown"

    LAYER_INPUT = 'input_node'
    LAYER_OUTPUT = 'output_node'

    LAYER_ADD = 'add_layer'
    LAYER_CONV2D = 'conv2d_layer'
    LAYER_CONV2D_DEPTHWISE = 'conv2d_depthwise_layer'
    LAYER_FC = 'fully_connected_layer'
    LAYER_REDUCE_MEAN = 'reduce_mean_layer'

    @classmethod
    def _missing_(cls, _):
        return SupportedLayerTypes.LAYER_UNKNOWN
