__all__ = [
    'efficientnet_preprocess_function',
]

_EFFNET_MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
_EFFNET_STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]

def efficientnet_preprocess_function(image):
    image = image - _EFFNET_MEAN_RGB
    image = image / _EFFNET_STDDEV_RGB
    return image
