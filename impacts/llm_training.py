import datetime
from typing import Any, Optional, Union, cast

from impacts.dag import DAG
from impacts.constants import (
    INFERENCE_COMPUTE_SHARE,
    GPU_UTILIZATION_RATE,
    MODEL_LIFESPAN, 
    FLOPS_PER_WATT,
    ENERGY_PER_FLOPS,
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
    model_lifespan: float,  # in years
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
    seconds_per_year = 365 * 24 * 60 * 60
    return (
        inference_compute_capacity_per_model
        * flops_per_watt
        * gpu_utilization_rate
        * model_lifespan
        * seconds_per_year
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
def total_training_energy(
    training_flops: ValueOrRange,  # in FLOP
    energy_per_flops: float,  # in Watt / (FLOP/s)
    datacenter_pue: ValueOrRange,
) -> ValueOrRange:
    """
    Total training energy during the training phase of the model.

    Args:
        training_flops: The total training FLOPs during the training phase of the model.
        energy_per_flops: The energy consumption per FLOP/s.
        datacenter_pue: The power usage effectiveness of the datacenter.

    Returns:
        The total training energy during the training phase of the model in kiloWatt-hour.
    """

    return (
        training_flops * energy_per_flops / (3600 * 1000)
    ) * datacenter_pue  # convert from Watt-second to kiloWatt-hour


@dag.asset
def training_energy_per_request(
    total_training_energy: ValueOrRange,  # in kiloWatt-hour
    total_output_tokens: ValueOrRange,
    output_token_count: float,
) -> ValueOrRange:
    """
    The training energy consumption allocated to each request.

    Args:
        total_training_energy: The total training energy during the training phase of the model in kiloWatt-hour.
        total_output_tokens: The total output token count during the lifespan of the model.
        output_token_count: The number of tokens generated in the request.
    Returns:
        The training energy consumption allocated to each request in kiloWatt-hour.
    """
    return (total_training_energy / total_output_tokens) * output_token_count


@dag.asset
def training_usage_gwp_per_request(
    training_energy_per_request: ValueOrRange,  # in kiloWatt-hour
    if_electricity_mix_gwp: float,  # in kgCO2eq / kWh
) -> ValueOrRange:
    """
    The GWP usage impact of the training phase allocated to each request.

    Args:
        training_energy_per_request: The training energy consumption allocated to each request in kiloWatt-hour.
        if_electricity_mix_gwp: The GWP impact factor of electricity consumption in kgCO2eq / kWh.

    Returns:
        The GWP usage impact of the training phase allocated to each request in kgCO2eq.
    """
    return training_energy_per_request * if_electricity_mix_gwp


@dag.asset
def training_usage_adpe_per_request(
    training_energy_per_request: ValueOrRange,  # in kiloWatt-hour
    if_electricity_mix_adpe: float,  # in kgSbeq / kWh
) -> ValueOrRange:
    """
    The ADPe usage impact of the training phase allocated to each request.

    Args:
        training_energy_per_request: The training energy consumption allocated to each request in kiloWatt-hour.
        if_electricity_mix_adpe: The ADPe impact factor of electricity consumption in kgSbeq / kWh.

    Returns:
        The ADPe usage impact of the training phase allocated to each request in kgSbeq.
    """
    return training_energy_per_request * if_electricity_mix_adpe


@dag.asset
def training_usage_pe_per_request(
    training_energy_per_request: ValueOrRange,  # in kiloWatt-hour
    if_electricity_mix_pe: float,  # in MJ / kWh
) -> ValueOrRange:
    """
    The PE usage impact of the training phase allocated to each request.

    Args:
        training_energy_per_request: The training energy consumption allocated to each request in kiloWatt-hour.
        if_electricity_mix_pe: The PE impact factor of electricity consumption in MJ / kWh.

    Returns:
        The PE usage impact of the training phase allocated to each request in MJ.
    """
    return training_energy_per_request * if_electricity_mix_pe


@dag.asset
def training_usage_wcf_per_request(
    training_energy_per_request: ValueOrRange,  # in kiloWatt-hour
    if_electricity_mix_wue: float,  # in L / kWh
    datacenter_wue: float,
    datacenter_pue: float,
) -> ValueOrRange:
    """
    The WCF usage impact of the training phase allocated to each request.

    Args:
        training_energy_per_request: The training energy consumption allocated to each request in kiloWatt-hour.
        if_electricity_mix_wue: The WCF impact factor of electricity consumption in L / kWh.
        datacenter_wue: Water Usage Effectiveness of the data center in L/kWh.
        datacenter_pue: Power Usage Effectiveness of the data center.

    Returns:
        The WCF usage impact of the training phase allocated to each request in Liter.
    """
    return training_energy_per_request * (
        datacenter_wue + datacenter_pue * if_electricity_mix_wue
    )


def compute_llm_train_impacts_dag(
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
    energy_per_flops: Optional[float] = ENERGY_PER_FLOPS,
    # gpu_embodied_gwp: Optional[float] = GPU_EMBODIED_IMPACT_GWP,
    # gpu_embodied_adpe: Optional[float] = GPU_EMBODIED_IMPACT_ADPE,
    # gpu_embodied_pe: Optional[float] = GPU_EMBODIED_IMPACT_PE,
    # server_gpu_count: Optional[int] = SERVER_GPUS,
    # server_power: Optional[float] = SERVER_POWER,
    # server_embodied_gwp: Optional[float] = SERVER_EMBODIED_IMPACT_GWP,
    # server_embodied_adpe: Optional[float] = SERVER_EMBODIED_IMPACT_ADPE,
    # server_embodied_pe: Optional[float] = SERVER_EMBODIED_IMPACT_PE,
    # server_lifetime: Optional[float] = HARDWARE_LIFESPAN,
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
        energy_per_flops: The energy consumption per FLOP/s.

    Returns:
        The environmental impacts dag for training phase with all intermediate states.
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
        energy_per_flops=energy_per_flops,
    )
    return results


def compute_llm_train_impacts(
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
    Compute the training impacts of an LLM generation request.

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
        The training impacts of an LLM generation request.
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
        "training_energy_per_request",
        "training_usage_gwp_per_request",
        "training_usage_adpe_per_request",
        "training_usage_pe_per_request",
        "training_usage_wcf_per_request",
    ]
    for act_param, tot_param in zip(active_params, total_params):
        res = compute_llm_train_impacts_dag(
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

    energy = Energy(value=results["training_energy_per_request"])
    gwp_usage = GWP(value=results["training_usage_gwp_per_request"])
    adpe_usage = ADPe(value=results["training_usage_adpe_per_request"])
    pe_usage = PE(value=results["training_usage_pe_per_request"])
    wcf_usage = WCF(value=results["training_usage_wcf_per_request"])
    gwp_embodied = GWP(value=0)
    adpe_embodied = ADPe(value=0)
    pe_embodied = PE(value=0)
    # TODO: add embodied impacts for training

    # print(results["training_energy_per_request"])

    return Impacts(
        energy=energy,
        gwp=gwp_usage + gwp_embodied,
        adpe=adpe_usage + adpe_embodied,
        pe=pe_usage + pe_embodied,
        wcf=wcf_usage,
        usage=Usage(
            energy=energy,
            gwp=gwp_usage,
            adpe=adpe_usage,
            pe=pe_usage,
            wcf=wcf_usage
        ),
        embodied=Embodied(
            gwp=gwp_embodied,
            adpe=adpe_embodied,
            pe=pe_embodied
        )
    )