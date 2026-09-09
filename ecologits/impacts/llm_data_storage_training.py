"""Environmental impacts of storing an LLM's training data."""

import datetime
from typing import Any

from ecologits.impacts.llm_training import (
    FLOPS_PER_WATT,
    GPU_UTILIZATION_RATE,
    INFERENCE_COMPUTE_SHARE,
    MODEL_LIFESPAN,
    allocated_per_request,
    inference_compute_capacity_per_model,
    total_output_tokens,
    training_flops,
    value_bounds,
)
from ecologits.impacts.modeling import GWP, PE, WCF, ADPe, Embodied, Energy, Impacts, Usage
from ecologits.utils.range_value import RangeValue, ValueOrRange

STORAGE_DURATION = 100 * 24
HDD_VOLUME = 30
HDD_POWER = 0.0095
HDD_USAGE_RATIO = 0.2
HDD_EMBODIED_IMPACT_GWP = 640.5
HDD_EMBODIED_IMPACT_ADPE = 0.0
HDD_EMBODIED_IMPACT_PE = 0.0
HDD_EMBODIED_IMPACT_WCF = 163480
HDD_LIFESPAN = 5 * 365 * 24 * 60 * 60


def training_tokens(
    training_flops_value: ValueOrRange,
    model_active_parameter_count: ValueOrRange,
) -> ValueOrRange:
    """
    Estimate the number of tokens used to train a model.

    Args:
        training_flops_value: Number of floating point operations used for training.
        model_active_parameter_count: Number of active parameters of the model (in billion).

    Returns:
        The estimated number of training tokens.
    """
    flops_min, flops_max = value_bounds(training_flops_value)
    params_min, params_max = value_bounds(model_active_parameter_count)
    values = (flops_min / (6 * params_max * 1e9), flops_max / (6 * params_min * 1e9))
    return values[0] if values[0] == values[1] else RangeValue(min=values[0], max=values[1])


def training_data_volume(training_tokens_value: ValueOrRange) -> ValueOrRange:
    """
    Estimate training-data volume in TB.

    Args:
        training_tokens_value: Number of tokens used to train the model.

    Returns:
        The training-data volume in TB, assuming four bytes per token.
    """
    return training_tokens_value * 4 / 1000**4


def hdd_required_count(training_data_volume_value: ValueOrRange, hdd_volume: float = HDD_VOLUME) -> ValueOrRange:
    """
    Compute the number of required HDDs to store the training data.

    Args:
        training_data_volume_value: Training-data volume in TB.
        hdd_volume: Storage capacity of a single HDD in TB.

    Returns:
        The number of required HDDs.
    """
    return training_data_volume_value / hdd_volume


def hdd_energy_training(
    hdd_count: ValueOrRange,
    hdd_power: float = HDD_POWER,
    hdd_usage_ratio: float = HDD_USAGE_RATIO,
    storage_duration: float = STORAGE_DURATION,
) -> ValueOrRange:
    """
    Compute the energy consumption of the HDDs during storage.

    Args:
        hdd_count: Number of HDDs used to store the training data.
        hdd_power: Power consumption of a single HDD in kW.
        hdd_usage_ratio: Usage ratio of the HDDs.
        storage_duration: Storage duration in hours.

    Returns:
        The energy consumption of the HDDs in kWh.
    """
    return hdd_count * hdd_power * hdd_usage_ratio * storage_duration


def compute_llm_train_data_storage_impacts(
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
    """
    Estimate training-data storage impacts allocated to one request.

    Args:
        publication_date: Publication date of the model.
        compute_capacity: Total fleet compute capacity in GW per year.
        number_of_active_models: Number of active models sharing the capacity per year.
        model_active_parameter_count: Number of active parameters of the model (in billion).
        model_total_parameter_count: Number of total parameters of the model (in billion).
        output_token_count: Number of generated tokens.
        if_electricity_mix_adpe: ADPe impact factor of electricity consumption in kgSbeq / kWh.
        if_electricity_mix_pe: PE impact factor of electricity consumption in MJ / kWh.
        if_electricity_mix_gwp: GWP impact factor of electricity consumption in kgCO2eq / kWh.
        if_electricity_mix_wue: WUE impact factor of electricity consumption in L / kWh.
        datacenter_pue: Power Usage Effectiveness of the data center.
        datacenter_wue: Water Usage Effectiveness of the data center.

    Returns:
        The training-data storage environmental impacts allocated to the request.
    """
    capacity = inference_compute_capacity_per_model(
        publication_date, compute_capacity, number_of_active_models,
        kwargs.get("inference_compute_share", INFERENCE_COMPUTE_SHARE),
    )
    total_tokens = total_output_tokens(
        capacity, kwargs.get("flops_per_watt", FLOPS_PER_WATT),
        kwargs.get("gpu_utilization_rate", GPU_UTILIZATION_RATE),
        kwargs.get("model_lifespan", MODEL_LIFESPAN), model_active_parameter_count,
    )
    train_data = training_data_volume(
        training_tokens(
            training_flops(publication_date, model_total_parameter_count),
            model_active_parameter_count,
        )
    )
    hdds = hdd_required_count(train_data, kwargs.get("hdd_volume", HDD_VOLUME))
    energy = allocated_per_request(
        hdd_energy_training(
            hdds,
            kwargs.get("hdd_power", HDD_POWER),
            kwargs.get("hdd_usage_ratio", HDD_USAGE_RATIO),
            kwargs.get("storage_duration", STORAGE_DURATION),
        ),
        total_tokens, output_token_count,
    ) * datacenter_pue
    usage_energy = Energy(value=energy)
    usage_gwp = GWP(value=usage_energy.value * if_electricity_mix_gwp)
    usage_adpe = ADPe(value=usage_energy.value * if_electricity_mix_adpe)
    usage_pe = PE(value=usage_energy.value * if_electricity_mix_pe)
    usage_wcf = WCF(value=usage_energy.value * (datacenter_wue + datacenter_pue * if_electricity_mix_wue))
    storage_hours = kwargs.get("storage_duration", STORAGE_DURATION)
    lifetime = kwargs.get("hdd_lifetime", HDD_LIFESPAN)
    # storage_duration is in hours while HDD_LIFESPAN is in seconds: convert the
    # lifetime to hours so the ratio matches the training module's allocation.
    embodied = allocated_per_request(
        hdds * storage_hours / (lifetime / 3600), total_tokens, output_token_count
    )
    embodied_gwp = GWP(value=embodied * kwargs.get("hdd_embodied_gwp", HDD_EMBODIED_IMPACT_GWP))
    embodied_adpe = ADPe(value=embodied * kwargs.get("hdd_embodied_adpe", HDD_EMBODIED_IMPACT_ADPE))
    embodied_pe = PE(value=embodied * kwargs.get("hdd_embodied_pe", HDD_EMBODIED_IMPACT_PE))
    embodied_wcf = WCF(value=embodied * kwargs.get("hdd_embodied_wcf", HDD_EMBODIED_IMPACT_WCF))
    return Impacts(
        energy=usage_energy,
        gwp=usage_gwp + embodied_gwp,
        adpe=usage_adpe + embodied_adpe,
        pe=usage_pe + embodied_pe,
        wcf=usage_wcf + embodied_wcf,
        usage=Usage(energy=usage_energy, gwp=usage_gwp, adpe=usage_adpe, pe=usage_pe, wcf=usage_wcf),
        embodied=Embodied(gwp=embodied_gwp, adpe=embodied_adpe, pe=embodied_pe, wcf=embodied_wcf),
    )
