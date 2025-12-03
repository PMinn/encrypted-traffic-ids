class FlowMeterException(Exception):
    pass

class NoPacketsException(FlowMeterException):
    pass

class NoIPPacketsException(FlowMeterException):
    pass