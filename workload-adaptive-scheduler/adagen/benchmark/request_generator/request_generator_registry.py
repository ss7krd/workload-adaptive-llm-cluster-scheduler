from adagen.benchmark.request_generator.synthetic_request_generator import (
    SyntheticRequestGenerator,
)
from adagen.benchmark.request_generator.trace_request_generator import (
    TraceRequestGenerator,
)
from adagen.types import RequestGeneratorType
from adagen.utils.base_registry import BaseRegistry


class RequestGeneratorRegistry(BaseRegistry):
    pass


RequestGeneratorRegistry.register(
    RequestGeneratorType.SYNTHETIC, SyntheticRequestGenerator
)
RequestGeneratorRegistry.register(RequestGeneratorType.TRACE, TraceRequestGenerator)
