class EmulatorError(Exception):
    """A common type of errors which can occur"""


class EmulatorSettingsError(EmulatorError):
    """Raised whenever there is a problem with the module settings"""


class GraphError(EmulatorError):
    pass


class ModelBuildingError(EmulatorError):
    pass


class GraphConfigError(GraphError):
    pass


class GraphLayerError(GraphError):
    pass


class GraphLayerCfgError(GraphLayerError):
    pass


class EmulatorRegistryKeyError(EmulatorError):
    pass


class DatasetError(EmulatorError):
    pass
