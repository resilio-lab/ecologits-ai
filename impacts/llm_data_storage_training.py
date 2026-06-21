import datetime
from typing import Any, Optional, Union, cast

from impacts.dag import DAG
from impacts.constants import (
    INFERENCE_COMPUTE_SHARE,
    GPU_UTILIZATION_RATE,
    MODEL_LIFESPAN,
    FLOPS_PER_WATT,
    STORAGE_DURATION,
    HDD_VOLUME,
    HDD_POWER,
    HDD_USAGE_RATIO,
    HDD_EMBODIED_IMPACT_GWP,
    HDD_EMBODIED_IMPACT_WCF,
    HDD_EMBODIED_IMPACT_ADPE,
    HDD_EMBODIED_IMPACT_PE,
    HDD_LIFESPAN,
)
from impacts.modeling import (
    GWP,
    PE,
    WCF,
    ADPe,
    Embodied,
    Energy,
    Impacts,
    Usage,
)
from range_value import RangeValue, ValueOrRange


dag = DAG()


@dag.asset
def inference_compute_capacity_per_model(
    publication_date: datetime.datetime,
    compute_capacity: dict,
    number_of_active_models: dict,
    inference_compute_share: float,
) -> float:
    """
    Compute the yearly inference compute capacity allocated to a single model.

    Args:
        publication_date: Publication date of the model.
        compute_capacity: A dictionary containing the company's total yearly compute capacity (in GigaWatts).
        number_of_active_models: A dictionary containing the company's total number of active AI models in each year.
        inference_compute_share: The share of compute capacity allocated to inference.

    Returns:
        The yearly inference compute capacity allocated to a single model in Watts.
    """
    publication_year = str(publication_date.year)
    compute_capacity_tmp = compute_capacity.get(publication_year, 1)  # gigaWatts
    compute_capacity_tmp *= 10**9  # convert from GigaWatts to Watts
    # TODO: modify the default number for compute_capacity & number_of_active_models
    return (
        compute_capacity_tmp
        * inference_compute_share
        / number_of_active_models.get(publication_year, 1)
    )


@dag.asset
def total_output_tokens(
    inference_compute_capacity_per_model: float,
    flops_per_watt: float,
    gpu_utilization_rate: float,
    model_lifespan: float,  # in seconds
    model_active_parameter_count: ValueOrRange,
) -> ValueOrRange:
    """
    Total output token count during the lifespan of the model.

    Args:
        inference_compute_capacity_per_model: The yearly inference compute capacity allocated to a single model in Watts.
        flops_per_watt: The number of FLOP that can be performed in one secondper Watt.
        gpu_utilization_rate: The share of time that the GPU is actively performing.
        model_lifespan: The time span during which the model is actively used after its publication
        model_active_parameter_count: Number of active parameters of the model (in billion).

    Returns:
        The total output token count during the lifespan of the model.
    """
    model_active_parameter_count *= 10**9  # convert from billion to actual number
    # seconds_per_year = 365 * 24 * 60 * 60
    return (
        inference_compute_capacity_per_model
        * flops_per_watt
        * gpu_utilization_rate
        * model_lifespan
        / (2 * model_active_parameter_count)
    )


@dag.asset
def training_flops(
    publication_date: datetime.datetime,
    model_total_parameter_count: ValueOrRange,
) -> ValueOrRange:
    """
    Total training FLOPs during the training phase of the model.

    Args:
        publication_date: Publication date of the model.
        model_total_parameter_count: Number of total parameters of the model (in billion).

    Returns:
        The total training FLOPs during the training phase of the model.
    """
    publication_days_since_2020 = (
        publication_date - datetime.datetime(2020, 1, 1)
    ).days
    model_total_parameter_count *= 10**9  # convert from billion to actual number
    return (10 ** (0.0006 * publication_days_since_2020 + 17.1510)) * (
        model_total_parameter_count**0.5410
    )


@dag.asset
def training_tokens(
    training_flops: ValueOrRange,
    model_active_parameter_count: ValueOrRange,
) -> ValueOrRange:
    """
    Training tokens used in the training phase of the model.

    Args:
        training_flops: Total training FLOPs during the training phase of the model.
        model_active_parameter_count: Number of active parameters of the model (in billion).

    Returns:
        The total training tokens used in the training phase of the model.
    """
    model_active_parameter_count *= 10**9  # convert from billion to actual number
    return training_flops / (6 * model_active_parameter_count)


@dag.asset
def training_data_volume(
    training_tokens: ValueOrRange,
) -> ValueOrRange:
    """
    Training data volume stored in the training phase of the model.

    Args:
        training_tokens: Total training tokens used in the training phase of the model.

    Returns:
        The total training data volume stored in the training phase of the model (in TB).
    """
    return training_tokens * 4 / (1000**4)  # convert from tokens to bytes to TB


# TODO: check if it's 1000 or 1024 for the conversion from bytes to TB.


@dag.asset
def hdd_required_count(
    training_data_volume: ValueOrRange,
    hdd_volume: float,
) -> ValueOrRange:
    """
    HDD required count to store the training data volume.

    Args:
        training_data_volume: Total training data volume stored in the training phase of the model (in TB).
        hdd_volume: Volume of a single HDD (in TB).

    Returns:
        The total number of HDDs required to store the training data volume.
    """
    return training_data_volume / hdd_volume


@dag.asset
def hdd_energy_training(
    hdd_required_count: ValueOrRange,
    hdd_power: float,
    hdd_usage_ratio: float,
    storage_duration: float,  # in hours
) -> ValueOrRange:
    """
    Energy consumption of HDDs used to store the training data volume.

    Args:
        hdd_required_count: Total number of HDDs required to store the training data volume.
        hdd_power: Power consumption of a single HDD (in kW).
        hdd_usage_ratio: The share of time that the HDD is actively performing.
        storage_duration: The duration for which the training data is stored (in hours).

    Returns:
        The total energy consumption of HDDs used to store the training data volume (in kWh).
    """
    return hdd_required_count * hdd_power * hdd_usage_ratio * storage_duration


@dag.asset
def request_hdd_energy_training(
    hdd_energy_training: ValueOrRange,
    total_output_tokens: ValueOrRange,
    output_token_count: float,
    datacenter_pue: ValueOrRange,
) -> ValueOrRange:
    """
    Energy consumption of HDDs used to store the training data volume during the lifespan of the model.

    Args:
        hdd_energy_training: Total energy consumption of HDDs used to store the training data volume (in kWh).
        total_output_tokens: Total number of output tokens generated by the model.
        output_token_count: Number of tokens in each output.
        datacenter_pue: Power Usage Effectiveness of the datacenter.

    Returns:
        The total energy consumption of HDDs used to store the training data volume during the lifespan of the model (in kWh).
    """
    return (hdd_energy_training * output_token_count / total_output_tokens) * datacenter_pue


@dag.asset
def request_hdd_usage_gwp_training(
    request_hdd_energy_training: ValueOrRange,
    if_electricity_mix_gwp: float
) -> ValueOrRange:
    """
    Usage GWP impact of HDDs used to store the training data volume during the lifespan of the model.

    Args:
        request_hdd_energy_training: Total energy consumption of HDDs used to store the training data volume during the lifespan of the model (in kWh).
        if_electricity_mix_gwp: GWP impact of the electricity mix (in kgCO2eq/kWh).

    Returns:
        The total usage GWP impact of HDDs used to store the training data volume during the lifespan of the model (in kgCO2eq).
    """
    return request_hdd_energy_training * if_electricity_mix_gwp


@dag.asset
def request_hdd_usage_adpe_training(
    request_hdd_energy_training: ValueOrRange,
    if_electricity_mix_adpe: float
) -> ValueOrRange:
    """
    Usage ADPe impact of HDDs used to store the training data volume during the lifespan of the model.

    Args:
        request_hdd_energy_training: Total energy consumption of HDDs used to store the training data volume during the lifespan of the model (in kWh).
        if_electricity_mix_adpe: ADPe impact of the electricity mix (in kgSbeq/kWh).

    Returns:
        The total usage ADPe impact of HDDs used to store the training data volume during the lifespan of the model (in kgSbeq).
    """
    return request_hdd_energy_training * if_electricity_mix_adpe


@dag.asset
def request_hdd_usage_pe_training(
    request_hdd_energy_training: ValueOrRange,
    if_electricity_mix_pe: float
) -> ValueOrRange:
    """
    Usage PE impact of HDDs used to store the training data volume during the lifespan of the model.

    Args:
        request_hdd_energy_training: Total energy consumption of HDDs used to store the training data volume during the lifespan of the model (in kWh).
        if_electricity_mix_pe: PE impact of the electricity mix (in MJ/kWh).

    Returns:
        The total usage PE impact of HDDs used to store the training data volume during the lifespan of the model (in MJ).
    """
    return request_hdd_energy_training * if_electricity_mix_pe


@dag.asset
def request_hdd_usage_wcf_training(
    request_hdd_energy_training: ValueOrRange,  # in kiloWatt-hour
    if_electricity_mix_wue: float,  # in L / kWh
    datacenter_wue: float,
    datacenter_pue: float,
) -> ValueOrRange:
    """
    The WCF usage impact of the training phase allocated to each request.

    Args:
        request_hdd_energy_training: The energy consumption of HDDs used to store the training data volume during the lifespan of the model in kiloWatt-hour.
        if_electricity_mix_wue: The WCF impact factor of electricity consumption in L / kWh.
        datacenter_wue: Water Usage Effectiveness of the data center in L/kWh.
        datacenter_pue: Power Usage Effectiveness of the data center.

    Returns:
        The WCF usage impact of the training phase allocated to each request in Liter.
    """
    return request_hdd_energy_training * (
        datacenter_wue + datacenter_pue * if_electricity_mix_wue
    )
# TODO: check the formula for water usage


@dag.asset
def hdd_embodied_gwp_training(
    hdd_required_count: ValueOrRange,
    hdd_embodied_impact_gwp: float,
    storage_duration: float,  # in hours
    hdd_lifetime: float,  # in seconds
) -> ValueOrRange:
    """
    Embodied GWP impact of HDDs used to store the training data volume.

    Args:
        hdd_required_count: Total number of HDDs required to store the training data volume.
        hdd_embodied_impact_gwp: Embodied GWP impact of a single HDD (in kgCO2eq).
        storage_duration: Duration for which the training data is stored (in hours).
        hdd_lifetime: Lifetime of a single HDD (in seconds).

    Returns:
        The total embodied GWP impact of HDDs used to store the training data volume (in kgCO2eq).
    """
    return hdd_required_count * hdd_embodied_impact_gwp * (storage_duration / hdd_lifetime)


@dag.asset
def hdd_embodied_adpe_training(
    hdd_required_count: ValueOrRange,
    hdd_embodied_impact_adpe: float,
    storage_duration: float,  # in hours
    hdd_lifetime: float,  # in seconds
) -> ValueOrRange:
    """
    Embodied ADPe impact of HDDs used to store the training data volume.

    Args:
        hdd_required_count: Total number of HDDs required to store the training data volume.
        hdd_embodied_impact_adpe: Embodied ADPe impact of a single HDD (in kgSbeq).
        storage_duration: Duration for which the training data is stored (in hours).
        hdd_lifetime: Lifetime of a single HDD (in seconds).

    Returns:
        The total embodied ADPe impact of HDDs used to store the training data volume (in kgSbeq).
    """
    return hdd_required_count * hdd_embodied_impact_adpe * (storage_duration / hdd_lifetime)


@dag.asset
def hdd_embodied_pe_training(
    hdd_required_count: ValueOrRange,
    hdd_embodied_impact_pe: float,
    storage_duration: float,  # in hours
    hdd_lifetime: float,  # in seconds
) -> ValueOrRange:
    """
    Embodied PE impact of HDDs used to store the training data volume.

    Args:
        hdd_required_count: Total number of HDDs required to store the training data volume.
        hdd_embodied_impact_pe: Embodied PE impact of a single HDD (in MJ).
        storage_duration: Duration for which the training data is stored (in hours).
        hdd_lifetime: Lifetime of a single HDD (in seconds).

    Returns:
        The total embodied PE impact of HDDs used to store the training data volume (in MJ).
    """
    return hdd_required_count * hdd_embodied_impact_pe * (storage_duration / hdd_lifetime)


@dag.asset
def hdd_embodied_wcf_training(
    hdd_required_count: ValueOrRange,
    hdd_embodied_impact_wcf: float,
    storage_duration: float,  # in hours
    hdd_lifetime: float,  # in seconds
) -> ValueOrRange:
    """
    Embodied WCF impact of HDDs used to store the training data volume.

    Args:
        hdd_required_count: Total number of HDDs required to store the training data volume.
        hdd_embodied_impact_wcf: Embodied WCF impact of a single HDD (in L).
        storage_duration: Duration for which the training data is stored (in hours).
        hdd_lifetime: Lifetime of a single HDD (in seconds).

    Returns:
        The total embodied WCF impact of HDDs used to store the training data volume (in L).
    """
    return hdd_required_count * hdd_embodied_impact_wcf * (storage_duration / hdd_lifetime)


@dag.asset
def request_hdd_embodied_gwp_training(
    hdd_embodied_gwp_training: ValueOrRange,
    total_output_tokens: ValueOrRange,
    output_token_count: float,
) -> ValueOrRange:
    """
    Embodied GWP impact of HDDs used to store the training data volume allocated to each request.

    Args:
        hdd_embodied_gwp_training: Total embodied GWP impact of HDDs used to store the training data volume (in kgCO2eq).
        total_output_tokens: Total number of output tokens generated by the model.
        output_token_count: Number of tokens in each output.

    Returns:
        The total embodied GWP impact of HDDs used to store the training data volume allocated to each request (in kgCO2eq).
    """
    return hdd_embodied_gwp_training * output_token_count / total_output_tokens


@dag.asset
def request_hdd_embodied_adpe_training(
    hdd_embodied_adpe_training: ValueOrRange,
    total_output_tokens: ValueOrRange,
    output_token_count: float,
) -> ValueOrRange:
    """
    Embodied ADPe impact of HDDs used to store the training data volume allocated to each request.

    Args:
        hdd_embodied_adpe_training: Total embodied ADPe impact of HDDs used to store the training data volume (in kgSbeq).
        total_output_tokens: Total number of output tokens generated by the model.
        output_token_count: Number of tokens in each output.

    Returns:
        The total embodied ADPe impact of HDDs used to store the training data volume allocated to each request (in kgSbeq).
    """
    return hdd_embodied_adpe_training * output_token_count / total_output_tokens


@dag.asset
def request_hdd_embodied_pe_training(
    hdd_embodied_pe_training: ValueOrRange,
    total_output_tokens: ValueOrRange,
    output_token_count: float,
) -> ValueOrRange:
    """
    Embodied PE impact of HDDs used to store the training data volume allocated to each request.

    Args:
        hdd_embodied_pe_training: Total embodied PE impact of HDDs used to store the training data volume (in MJ).
        total_output_tokens: Total number of output tokens generated by the model.
        output_token_count: Number of tokens in each output.

    Returns:
        The total embodied PE impact of HDDs used to store the training data volume allocated to each request (in MJ).
    """
    return hdd_embodied_pe_training * output_token_count / total_output_tokens  


@dag.asset
def request_hdd_embodied_wcf_training(
    hdd_embodied_wcf_training: ValueOrRange,
    total_output_tokens: ValueOrRange,
    output_token_count: float,
) -> ValueOrRange:
    """
    Embodied WCF impact of HDDs used to store the training data volume allocated to each request.

    Args:
        hdd_embodied_wcf_training: Total embodied WCF impact of HDDs used to store the training data volume (in L).
        total_output_tokens: Total number of output tokens generated by the model.
        output_token_count: Number of tokens in each output.

    Returns:
        The total embodied WCF impact of HDDs used to store the training data volume allocated to each request (in L).
    """
    return hdd_embodied_wcf_training * output_token_count / total_output_tokens


def compute_llm_train_data_storage_impacts_dag(
    publication_date: datetime.datetime,
    compute_capacity: dict,
    number_of_active_models: dict,
    model_active_parameter_count: ValueOrRange,
    model_total_parameter_count: ValueOrRange,
    output_token_count: float,
    if_electricity_mix_adpe: float,
    if_electricity_mix_pe: float,
    if_electricity_mix_gwp: float,
    if_electricity_mix_wue: float,
    datacenter_pue: ValueOrRange,
    datacenter_wue: ValueOrRange,
    inference_compute_share: Optional[float] = INFERENCE_COMPUTE_SHARE,
    gpu_utilization_rate: Optional[float] = GPU_UTILIZATION_RATE,
    model_lifespan: Optional[float] = MODEL_LIFESPAN,
    flops_per_watt: Optional[float] = FLOPS_PER_WATT,
    #############
    storage_duration: Optional[float] = STORAGE_DURATION,
    hdd_volume: Optional[float] = HDD_VOLUME,
    hdd_power: Optional[float] = HDD_POWER,
    hdd_usage_ratio: Optional[float] = HDD_USAGE_RATIO,
    hdd_embodied_gwp: Optional[float] = HDD_EMBODIED_IMPACT_GWP,
    hdd_embodied_adpe: Optional[float] = HDD_EMBODIED_IMPACT_ADPE,
    hdd_embodied_pe: Optional[float] = HDD_EMBODIED_IMPACT_PE,
    hdd_embodied_wcf: Optional[float] = HDD_EMBODIED_IMPACT_WCF,
    hdd_lifetime: Optional[float] = HDD_LIFESPAN,
) -> dict[str, ValueOrRange]:
    """
    Compute the impacts dag of an LLM generation request.

    Args:
        publication_date: Publication date of the model.
        compute_capacity: A dictionary containing the company's total yearly compute capacity (in GigaWatts).
        number_of_active_models: A dictionary containing the company's total number of active AI models in each year.
        model_active_parameter_count: Number of active parameters of the model (in billion).
        model_total_parameter_count: Number of parameters of the model (in billion).
        output_token_count: Number of generated tokens.
        if_electricity_mix_adpe: ADPe impact factor of electricity consumption in kgSbeq / kWh (Antimony).
        if_electricity_mix_pe: PE impact factor of electricity consumption in MJ / kWh.
        if_electricity_mix_gwp: GWP impact factor of electricity consumption in kgCO2eq / kWh.
        if_electricity_mix_wue: WCF impact factor of electricity consumption in L / kWh.
        datacenter_wue: Water Usage Effectiveness of the data center in L/kWh.
        datacenter_pue: Power Usage Effectiveness of the data center.
        inference_compute_share: The share of compute capacity allocated to inference, used to estimate the inference compute capacity from the total compute capacity.
        gpu_utilization_rate: The share of time that the GPU is actively performing
        model_lifespan: The time span during which the model is actively used after its publication
        flops_per_watt: The number of FLOP that can be performed in one second per Watt.
        storage_duration: The duration for which the training data is stored (in hours).
        hdd_volume: Volume of a single HDD (in TB).
        hdd_power: Power consumption of a single HDD (in kW).     
        hdd_usage_ratio: The share of time that the HDD is actively performing.
        hdd_embodied_gwp: Embodied GWP impact of a single HDD (in kgCO2eq).
        hdd_embodied_adpe: Embodied ADPe impact of a single HDD (in kgSbeq).
        hdd_embodied_pe: Embodied PE impact of a single HDD (in MJ).
        hdd_embodied_wcf: Embodied WCF impact of a single HDD (in L).
        hdd_lifetime: Lifetime of a single HDD (in seconds).

    Returns:
        The environmental impacts dag for training - data storage phase with all intermediate states.
    """
    results = dag.execute(
        publication_date=publication_date,
        compute_capacity=compute_capacity,
        number_of_active_models=number_of_active_models,
        model_active_parameter_count=model_active_parameter_count,
        model_total_parameter_count=model_total_parameter_count,
        output_token_count=output_token_count,
        if_electricity_mix_gwp=if_electricity_mix_gwp,
        if_electricity_mix_adpe=if_electricity_mix_adpe,
        if_electricity_mix_pe=if_electricity_mix_pe,
        if_electricity_mix_wue=if_electricity_mix_wue,
        datacenter_wue=datacenter_wue,
        datacenter_pue=datacenter_pue,
        inference_compute_share=inference_compute_share,
        gpu_utilization_rate=gpu_utilization_rate,
        model_lifespan=model_lifespan,
        flops_per_watt=flops_per_watt,
        storage_duration=storage_duration,
        hdd_volume=hdd_volume,
        hdd_power=hdd_power,
        hdd_usage_ratio=hdd_usage_ratio,
        hdd_embodied_impact_gwp=hdd_embodied_gwp,       
        hdd_embodied_impact_adpe=hdd_embodied_adpe,
        hdd_embodied_impact_pe=hdd_embodied_pe,
        hdd_embodied_impact_wcf=hdd_embodied_wcf,
        hdd_lifetime=hdd_lifetime
    )
    return results


def compute_llm_train_data_storage_impacts(
    publication_date: datetime.datetime,
    compute_capacity: dict,
    number_of_active_models: dict,
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
    Compute the training - data storage impacts of an LLM generation request.

    Args:
        publication_date: Publication date of the model.
        compute_capacity: A dictionary containing the company's total yearly compute capacity (in GigaWatts).
        number_of_active_models: A dictionary containing the company's total number of active AI models in each year.
        model_active_parameter_count: Number of active parameters of the model (in billion).
        model_total_parameter_count: Number of parameters of the model (in billion).
        output_token_count: Number of generated tokens.
        if_electricity_mix_adpe: ADPe impact factor of electricity consumption in kgSbeq / kWh (Antimony).
        if_electricity_mix_pe: PE impact factor of electricity consumption in MJ / kWh.
        if_electricity_mix_gwp: GWP impact factor of electricity consumption in kgCO2eq / kWh.
        if_electricity_mix_wue: WCF impact factor of electricity consumption in L / kWh.
        datacenter_wue: Water Usage Effectiveness of the data center in L/kWh.
        datacenter_pue: Power Usage Effectiveness of the data center.
        **kwargs: Any other optional parameter.
    Returns:
        The training - data storage impacts of an LLM generation request.
    """

    active_params = [model_active_parameter_count]
    total_params = [model_total_parameter_count]

    if isinstance(model_active_parameter_count, RangeValue) or isinstance(
        model_total_parameter_count, RangeValue
    ):
        if isinstance(model_active_parameter_count, RangeValue):
            active_params = [
                model_active_parameter_count.min,
                model_active_parameter_count.max,
            ]
        else:
            active_params = [model_active_parameter_count, model_active_parameter_count]
        if isinstance(model_total_parameter_count, RangeValue):
            total_params = [
                model_total_parameter_count.min,
                model_total_parameter_count.max,
            ]
        else:
            total_params = [model_total_parameter_count, model_total_parameter_count]

    results: dict[str, Union[RangeValue, float, int]] = {}
    fields = [
        "request_hdd_energy_training",
        "request_hdd_usage_gwp_training",
        "request_hdd_usage_adpe_training",
        "request_hdd_usage_pe_training",
        "request_hdd_usage_wcf_training",
        "request_hdd_embodied_gwp_training",
        "request_hdd_embodied_adpe_training",
        "request_hdd_embodied_pe_training",
        "request_hdd_embodied_wcf_training"
    ]
    for act_param, tot_param in zip(active_params, total_params):
        res = compute_llm_train_data_storage_impacts_dag(
            publication_date=publication_date,
            compute_capacity=compute_capacity,
            number_of_active_models=number_of_active_models,
            model_active_parameter_count=act_param,
            model_total_parameter_count=tot_param,
            output_token_count=output_token_count,
            if_electricity_mix_adpe=if_electricity_mix_adpe,
            if_electricity_mix_pe=if_electricity_mix_pe,
            if_electricity_mix_gwp=if_electricity_mix_gwp,
            if_electricity_mix_wue=if_electricity_mix_wue,
            datacenter_pue=datacenter_pue,
            datacenter_wue=datacenter_wue,
            **kwargs,
        )
        for field in fields:
            if field in results:
                min_result = results[field]
                max_result = res[field]
                if isinstance(min_result, RangeValue):
                    min_result = cast(Union[float, int], min_result.min)
                if isinstance(max_result, RangeValue):
                    max_result = cast(Union[float, int], max_result.max)
                results[field] = RangeValue(min=min_result, max=max_result)
            else:
                results[field] = res[field]

    energy = Energy(value=results["request_hdd_energy_training"])
    gwp_usage = GWP(value=results["request_hdd_usage_gwp_training"])
    adpe_usage = ADPe(value=results["request_hdd_usage_adpe_training"])
    pe_usage = PE(value=results["request_hdd_usage_pe_training"])
    wcf_usage = WCF(value=results["request_hdd_usage_wcf_training"])
    gwp_embodied = GWP(value=results["request_hdd_embodied_gwp_training"])
    adpe_embodied = ADPe(value=results["request_hdd_embodied_adpe_training"])
    pe_embodied = PE(value=results["request_hdd_embodied_pe_training"])
    wcf_embodied = WCF(value=results["request_hdd_embodied_wcf_training"])

    # print(results["request_hdd_energy_training"])

    return Impacts(
        energy=energy,
        gwp=gwp_usage + gwp_embodied,
        adpe=adpe_usage + adpe_embodied,
        pe=pe_usage + pe_embodied,
        wcf=wcf_usage + wcf_embodied,
        usage=Usage(
            energy=energy, gwp=gwp_usage, adpe=adpe_usage, pe=pe_usage, wcf=wcf_usage
        ),
        embodied=Embodied(
            gwp=gwp_embodied, adpe=adpe_embodied, pe=pe_embodied, wcf=wcf_embodied
        ),
    )

