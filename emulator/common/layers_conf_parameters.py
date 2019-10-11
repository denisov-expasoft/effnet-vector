__all__ = [
    'LayerConfigurationParameters',
]

from enum import Enum
from enum import unique


@unique
class LayerConfigurationParameters(Enum):
    QUANT_ACTIVATIONS_BITS = 'activations_bits'
    QUANT_WEIGHTS_BITS = 'weights_bits'
    QUANT_WEIGHTS_THRESHOLDS = 'weights_thresholds'
    QUANT_WEIGHTS_NARROW_RANGE = 'weights_narrow_range'

    ATTR_COMMON_TYPE = 'type'
    ATTR_COMMON_BOTTOM = 'bottom'

    ATTR_IO_SHAPE = 'shape'
    ATTR_AXIS = 'axis'
    ATTR_KERNEL_SIZE = 'ksize'
    ATTR_WEIGHTS_SHAPE = 'weights_shape'
    ATTR_NN_ACTIVATION_TYPE = 'activation'
    ATTR_NN_PADDING_STRATEGY = 'padding'
    ATTR_NN_STRIDES = 'strides'
    ATTR_NN_DILATIONS = 'dilations'
    ATTR_PADDING_VALUES = 'paddings'
    ATTR_REDUCTION_KEEPDIMS = 'keepdims'

    ATTR_INPUT_SHAPE = 'input_shape'
    ATTR_OUTPUT_SHAPE = 'output_shape'

    ATTR_IO_GRAPH_ANCHOR = 'anchor'
