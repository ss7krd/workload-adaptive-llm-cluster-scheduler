import datetime
from dataclasses import dataclass, field
from typing import Optional

from sarathi.config import BaseEndpointConfig
from sarathi.config.base_poly_config import BasePolyConfig
from sarathi.config.flat_dataclass import create_flat_dataclass
from sarathi.logger import init_logger
from sarathi.types import (
    ReplicaResourceMapping,
    RequestGeneratorType,
    RequestIntervalGeneratorType,
    RequestLengthGeneratorType,
)

logger = init_logger(__name__)


@dataclass
class BaseRequestIntervalGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42, metadata={"help": "Random seed for the request interval generator."}
    )


@dataclass
class BaseRequestLengthGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42, metadata={"help": "Random seed for the request length generator."}
    )


@dataclass
class TraceRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/AzureFunctionsInvocationTraceForTwoWeeksJan2021Processed.csv",
        metadata={"help": "Path to the trace file for request intervals."},
    )
    start_time: str = field(
        default="1970-01-04 12:00:00", metadata={"help": "Start time for the trace."}
    )
    end_time: str = field(
        default="1970-01-04 15:00:00", metadata={"help": "End time for the trace."}
    )
    time_scale_factor: float = field(
        default=0.3,
        metadata={"help": "Factor to scale the time intervals in the trace."},
    )

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.TRACE


@dataclass
class PoissonRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    qps: float = field(
        default=1.0,
        metadata={"help": "Queries per second for the Poisson distribution."},
    )

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.POISSON


@dataclass
class GammaRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    qps: float = field(
        default=1.0, metadata={"help": "Queries per second for the Gamma distribution."}
    )
    cv: float = field(
        default=0.5,
        metadata={"help": "Coefficient of variation for the Gamma distribution."},
    )

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.GAMMA


@dataclass
class StaticRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.STATIC


@dataclass
class TraceRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    trace_file: str = field(
        default="dataFolder/burstGPT.csv",
        metadata={"help": "Path to the trace file for request lengths."},
    )
    prefill_scale_factor: float = field(
        default=1, metadata={"help": "Scale factor for prefill tokens."}
    )
    decode_scale_factor: float = field(
        default=1, metadata={"help": "Scale factor for decode tokens."}
    )
    max_tokens: int = field(
        default=4096, metadata={"help": "Maximum number of tokens allowed."}
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.TRACE


@dataclass
class ZipfRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    theta: float = field(
        default=0.6, metadata={"help": "Theta parameter for the Zipf distribution."}
    )
    scramble: bool = field(
        default=False, metadata={"help": "Whether to scramble the Zipf distribution."}
    )
    min_tokens: int = field(
        default=1024, metadata={"help": "Minimum number of tokens."}
    )
    max_tokens: int = field(
        default=4096, metadata={"help": "Maximum number of tokens."}
    )
    prefill_to_decode_ratio: float = field(
        default=20.0, metadata={"help": "Ratio of prefill tokens to decode tokens."}
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.ZIPF


@dataclass
class UniformRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    min_tokens: int = field(
        default=1024, metadata={"help": "Minimum number of tokens."}
    )
    max_tokens: int = field(
        default=4096, metadata={"help": "Maximum number of tokens."}
    )
    prefill_to_decode_ratio: float = field(
        default=20.0, metadata={"help": "Ratio of prefill tokens to decode tokens."}
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.UNIFORM


@dataclass
class FixedRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    prefill_tokens: int = field(
        default=4096, metadata={"help": "Number of prefill tokens."}
    )
    decode_tokens: int = field(
        default=512, metadata={"help": "Number of decode tokens."}
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.FIXED


@dataclass
class BaseRequestGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42, metadata={"help": "Random seed for the request generator."}
    )


@dataclass
class SyntheticRequestGeneratorConfig(BaseRequestGeneratorConfig):
    length_generator_config: BaseRequestLengthGeneratorConfig = field(
        default_factory=FixedRequestLengthGeneratorConfig
    )
    interval_generator_config: BaseRequestIntervalGeneratorConfig = field(
        default_factory=PoissonRequestIntervalGeneratorConfig
    )
    num_requests: int = field(
        default=64, metadata={"help": "Number of requests to generate."}
    )
    duration: float = field(
        default=None, metadata={"help": "Duration of the synthetic request generation."}
    )

    @staticmethod
    def get_type():
        return RequestGeneratorType.SYNTHETIC


@dataclass
class TraceRequestGeneratorConfig(BaseRequestGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/sydney_enterprise.csv",
        metadata={"help": "Path to the trace file for request generation."},
    )
    date: str = field(
        default="2023-08-21", metadata={"help": "Date for the trace data."}
    )
    prefill_scale_factor: float = field(
        default=0.3, metadata={"help": "Scale factor for prefill tokens."}
    )
    decode_scale_factor: float = field(
        default=1, metadata={"help": "Scale factor for decode tokens."}
    )
    time_scale_factor: float = field(
        default=0.04, metadata={"help": "Scale factor for time intervals."}
    )
    max_tokens: int = field(
        default=4096, metadata={"help": "Maximum number of tokens allowed."}
    )

    @staticmethod
    def get_type():
        return RequestGeneratorType.TRACE


@dataclass
class BenchmarkConfig(BaseEndpointConfig):
    seed: int = field(default=42, metadata={"help": "Random seed for the benchmark."})
    output_dir: str = field(
        default="benchmark_output",
        metadata={"help": "Directory to store benchmark output."},
    )
    num_replicas: int = field(
        default=2, metadata={"help": "Number of replicas to use."}
    )
    num_requests_limit: int = field(
        default=10, metadata={"help": "Number of maximum requests to use."}
    )
    model_name: str = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",#mistralai/Mixtral-8x7B-Instruct-v0.1
        metadata={"help": "Model to use"}
    )
    modelName: str = field(
        default="Llama3-8B",
        metadata={"help": "Model to use-final"}
    )
    write_json_trace: bool = field(
        default=True, metadata={"help": "Whether to write JSON trace output."}
    )
    enable_profiling: bool = field(
        default=False, metadata={"help": "Whether to enable profiling."}
    )
    time_limit: Optional[int] = field(
        default=None, metadata={"help": "Time limit for the benchmark in seconds."}
    )
    model_list: list[str] = field(
        default_factory=lambda: ["Llama3-8B", "Mixtral", "Llama3-70B"]
    )
    request_rate_list: list[int] = field(
        default_factory=lambda: [20, 30, 40, 50, 60]
    )
    slo_scale_list: list[float] = field(
        default_factory=lambda: [0.5, 0.75, 1, 1.25, 1.5]
    )
    scheduler_name: str = field(
        default="adagen",
        metadata={"help": "Dispatching scheduler to use: adagen, llumnix [+SD (optional)], round-robin [+SD (optional)], exhaustive (please run from inside adagen folder),"}
    )
    figure_name: str = field(
        default="12a",
        metadata={"help": "figure number"}
    )
    instance_count: int = field(
        default=8,
        metadata={"help": "number of instances"}
    )
    KV_cache_size: float = field(
        default=200
    )
    st_size: int = field(
        default = 60
    )
    replica_resource_mapping: Optional[ReplicaResourceMapping] = field(
        default=None, metadata={"help": "Mapping of replicas to resources."}
    )
    request_generator_config: BaseRequestGeneratorConfig = field(
        default_factory=SyntheticRequestGeneratorConfig
    )

    def __post_init__(self):
        super().__post_init__()

        if not self.time_limit:
            self.time_limit = float("inf")
