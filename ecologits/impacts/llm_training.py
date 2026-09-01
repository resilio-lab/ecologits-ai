"""Environmental impact estimates for the training of an LLM.

The training estimate is allocated to an inference request by its output token
count.  The inputs describing model publication and fleet compute are kept
explicit because they are not part of the provider response schema.
"""

import datetime
from typing import Any

from ecologits.impacts.constants import (
    FLOPS_PER_GPU,
    FLOPS_PER_WATT,
    GPU_EMBODIED_IMPACT_ADPE,
    GPU_EMBODIED_IMPACT_GWP,
    GPU_EMBODIED_IMPACT_PE,
    GPU_EMBODIED_IMPACT_WCF,
    GPU_UTILIZATION_RATE,
    HARDWARE_LIFESPAN,
    INFERENCE_COMPUTE_SHARE,
    MODEL_LIFESPAN,
    NETWORK_EMBODIED_IMPACT_ADPE,
    NETWORK_EMBODIED_IMPACT_GWP,
    NETWORK_EMBODIED_IMPACT_PE,
    NETWORK_EMBODIED_IMPACT_WCF,
    NETWORK_LIFESPAN,
    SERVER_EMBODIED_IMPACT_ADPE,
    SERVER_EMBODIED_IMPACT_GWP,
    SERVER_EMBODIED_IMPACT_PE,
    SERVER_EMBODIED_IMPACT_WCF,
    SERVER_GPU_NETWORK_POWER,
    SERVER_GPUS,
)
from ecologits.impacts.modeling import GWP, PE, WCF, ADPe, Embodied, Energy, Impacts, Usage
from ecologits.utils.range_value import RangeValue, ValueOrRange


def value_bounds(value: ValueOrRange) -> tuple[float, float]:
    """Return the ``(min, max)`` bounds of a value or range."""
    if isinstance(value, RangeValue):
        return float(value.min), float(value.max)
    return float(value), float(value)


def inference_compute_capacity_per_model(
    publication_date: datetime.datetime,
    compute_capacity: dict[str, float],
    number_of_active_models: dict[str, float],
    inference_compute_share: float = INFERENCE_COMPUTE_SHARE,
) -> float:
    """Return yearly inference capacity for one model, in watts."""
    year = str(publication_date.year)
    return compute_capacity.get(year, 1.0) * 1e9 * inference_compute_share / number_of_active_models.get(year, 1.0)


def total_output_tokens(
    inference_compute_capacity_per_model: float,
    flops_per_watt: float,
    gpu_utilization_rate: float,
    model_lifespan: float,
    model_active_parameter_count: ValueOrRange,
) -> ValueOrRange:
    """Estimate total inference output tokens over the model lifespan."""
    minimum, maximum = value_bounds(model_active_parameter_count)
    coefficient = inference_compute_capacity_per_model * flops_per_watt * gpu_utilization_rate * model_lifespan / 2e9
    if minimum == maximum:
        return coefficient / minimum
    return RangeValue(min=coefficient / maximum, max=coefficient / minimum)


def training_flops(
    publication_date: datetime.datetime,
    model_total_parameter_count: ValueOrRange,
) -> ValueOrRange:
    """Estimate training FLOPs from publication date and total parameters."""
    days = (publication_date - datetime.datetime(2020, 1, 1)).days
    coefficient = 10 ** (0.0006 * days + 17.1510)
    minimum, maximum = value_bounds(model_total_parameter_count)
    values = [coefficient * (parameters * 1e9) ** 0.5410 for parameters in (minimum, maximum)]
    return values[0] if values[0] == values[1] else RangeValue(min=values[0], max=values[1])


def server_hours_training(
    training_flops: ValueOrRange,
    flops_per_gpu: float = FLOPS_PER_GPU,
    gpu_utilization_rate: float = GPU_UTILIZATION_RATE,
    server_gpu_count: float = SERVER_GPUS,
) -> ValueOrRange:
    """Convert training FLOPs to server-hours."""
    return training_flops / (flops_per_gpu * gpu_utilization_rate * server_gpu_count) / 3600


def total_training_energy(
    server_hours: ValueOrRange,
    total_power: float = SERVER_GPU_NETWORK_POWER,
    datacenter_pue: ValueOrRange = 1.0,
) -> ValueOrRange:
    """Return total training energy in kWh."""
    return server_hours * total_power * datacenter_pue


def allocated_per_request(total: ValueOrRange, tokens: ValueOrRange, output_token_count: float) -> ValueOrRange:
    """Allocate a share of a total to a single inference request.

    Args:
        total: Total value over the whole allocation basis (e.g. the model lifespan).
        tokens: Total number of output tokens over the same basis.
        output_token_count: Number of output tokens of the request.

    Returns:
        The share of the total allocated to the request.
    """
    total_min, total_max = value_bounds(total)
    token_min, token_max = value_bounds(tokens)
    values = [total_min / token_max * output_token_count, total_max / token_min * output_token_count]
    return values[0] if values[0] == values[1] else RangeValue(min=values[0], max=values[1])


def compute_llm_train_impacts(
    publication_date: datetime.datetime,
    compute_capacity: dict[str, float],
    number_of_active_models: dict[str, float],
    model_active_parameter_count: ValueOrRange,
    model_total_parameter_count: ValueOrRange,
    output_token_count: float,
    if_electricity_mix_adpe: float,
    if_electricity_mix_pe: float,
    if_electricity_mix_gwp: float,
    if_electricity_mix_wue: float,
    datacenter_pue: ValueOrRange,
    datacenter_wue: ValueOrRange,
    **kwargs: Any,
) -> Impacts:
    """Estimate training impacts allocated to one inference request.

    Optional keyword arguments customize the hardware and regression defaults.
    Network equipment is included in the training energy (server power) and in
    the embodied impacts. Training-data-storage impacts are estimated separately
    by ``compute_llm_train_data_storage_impacts``.
    """
    capacity = inference_compute_capacity_per_model(
        publication_date, compute_capacity, number_of_active_models,
        kwargs.get("inference_compute_share", INFERENCE_COMPUTE_SHARE),
    )
    tokens = total_output_tokens(
        capacity, kwargs.get("flops_per_watt", FLOPS_PER_WATT),
        kwargs.get("gpu_utilization_rate", GPU_UTILIZATION_RATE),
        kwargs.get("model_lifespan", MODEL_LIFESPAN), model_active_parameter_count,
    )
    flops = training_flops(publication_date, model_total_parameter_count)
    hours = server_hours_training(
        flops, kwargs.get("flops_per_gpu", FLOPS_PER_GPU),
        kwargs.get("gpu_utilization_rate", GPU_UTILIZATION_RATE),
        kwargs.get("server_gpu_count", SERVER_GPUS),
    )
    energy_value = allocated_per_request(total_training_energy(
        hours, kwargs.get("total_power", SERVER_GPU_NETWORK_POWER), datacenter_pue,
    ), tokens, output_token_count)
    usage_energy = Energy(value=energy_value)
    usage_gwp = GWP(value=usage_energy.value * if_electricity_mix_gwp)
    usage_adpe = ADPe(value=usage_energy.value * if_electricity_mix_adpe)
    usage_pe = PE(value=usage_energy.value * if_electricity_mix_pe)
    usage_wcf = WCF(value=usage_energy.value * (datacenter_wue + datacenter_pue * if_electricity_mix_wue))

    server_gpu_count = kwargs.get("server_gpu_count", SERVER_GPUS)
    embodied_hours = hours
    embodied_values = {}
    for name, server, gpu in (
        (
            "gwp", kwargs.get("server_embodied_gwp", SERVER_EMBODIED_IMPACT_GWP),
            kwargs.get("gpu_embodied_gwp", GPU_EMBODIED_IMPACT_GWP),
        ),
        (
            "adpe", kwargs.get("server_embodied_adpe", SERVER_EMBODIED_IMPACT_ADPE),
            kwargs.get("gpu_embodied_adpe", GPU_EMBODIED_IMPACT_ADPE),
        ),
        (
            "pe", kwargs.get("server_embodied_pe", SERVER_EMBODIED_IMPACT_PE),
            kwargs.get("gpu_embodied_pe", GPU_EMBODIED_IMPACT_PE),
        ),
        (
            "wcf", kwargs.get("server_embodied_wcf", SERVER_EMBODIED_IMPACT_WCF),
            kwargs.get("gpu_embodied_wcf", GPU_EMBODIED_IMPACT_WCF),
        ),
    ):
        embodied_values[name] = allocated_per_request(
            embodied_hours * (server + gpu * server_gpu_count) / (
                kwargs.get("server_lifetime", HARDWARE_LIFESPAN) / 3600
            ),
            tokens, output_token_count,
        )
    for name, impact in (("gwp", NETWORK_EMBODIED_IMPACT_GWP), ("adpe", NETWORK_EMBODIED_IMPACT_ADPE),
                         ("pe", NETWORK_EMBODIED_IMPACT_PE), ("wcf", NETWORK_EMBODIED_IMPACT_WCF)):
        embodied_values[name] = allocated_per_request(
            embodied_hours * impact / (NETWORK_LIFESPAN / 3600), tokens, output_token_count,
        ) + embodied_values.get(name, 0)
    embodied_gwp = GWP(value=embodied_values["gwp"])
    embodied_adpe = ADPe(value=embodied_values["adpe"])
    embodied_pe = PE(value=embodied_values["pe"])
    embodied_wcf = WCF(value=embodied_values["wcf"])
    return Impacts(
        energy=usage_energy,
        gwp=usage_gwp + embodied_gwp,
        adpe=usage_adpe + embodied_adpe,
        pe=usage_pe + embodied_pe,
        wcf=usage_wcf + embodied_wcf,
        usage=Usage(energy=usage_energy, gwp=usage_gwp, adpe=usage_adpe, pe=usage_pe, wcf=usage_wcf),
        embodied=Embodied(gwp=embodied_gwp, adpe=embodied_adpe, pe=embodied_pe, wcf=embodied_wcf),
    )


__all__ = [
    "allocated_per_request",
    "compute_llm_train_impacts",
    "inference_compute_capacity_per_model",
    "server_hours_training",
    "total_output_tokens",
    "total_training_energy",
    "training_flops",
    "value_bounds",
]
