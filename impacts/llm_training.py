import datetime
from typing import Any, Optional, Union, cast

from impacts.dag import DAG
from impacts.constants import (
    GPU_EMBODIED_IMPACT_ADPE,
    GPU_EMBODIED_IMPACT_GWP,
    GPU_EMBODIED_IMPACT_PE,
    GPU_EMBODIED_IMPACT_WCF,
    HARDWARE_LIFESPAN,
    SERVER_EMBODIED_IMPACT_ADPE,
    SERVER_EMBODIED_IMPACT_GWP,
    SERVER_EMBODIED_IMPACT_PE,
    SERVER_EMBODIED_IMPACT_WCF,
    SERVER_GPUS,
    NETWORK_EMBODIED_IMPACT_GWP,
    NETWORK_EMBODIED_IMPACT_ADPE,
    NETWORK_EMBODIED_IMPACT_PE,
    NETWORK_EMBODIED_IMPACT_WCF,
    NETWORK_LIFESPAN,
    #################
    FLOPS_PER_GPU,
    INFERENCE_COMPUTE_SHARE,
    GPU_UTILIZATION_RATE,
    MODEL_LIFESPAN,
    FLOPS_PER_WATT,
    # ENERGY_PER_FLOPS,
    SERVER_GPU_NETWORK_POWER,
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
def server_hours_training(
    training_flops: ValueOrRange,  # in FLOP
    flops_per_GPU: float,  # in FLOP/s / GPU
    gpu_utilization_rate: float,
    server_gpu_count: float,
) -> ValueOrRange:
    """
    Total server hours during the training phase of the model.

    Args:
        training_flops: The total training FLOPs during the training phase of the model.
        flops_per_GPU: The number of FLOPs that can be performed in one second per GPU.
        gpu_utilization_rate: The share of time that the GPU is actively performing.
        server_gpu_count: The number of GPUs in the server.

    Returns:
        The total server hours during the training phase of the model.
    """
    return (
        training_flops
        / (flops_per_GPU * gpu_utilization_rate * server_gpu_count)
        / 3600
    )


@dag.asset
def total_training_energy(
    server_hours_training: ValueOrRange,  # in server-hours
    total_power: float,  # kW
    datacenter_pue: ValueOrRange,
) -> ValueOrRange:
    """
    Total training energy during the training phase of the model.

    Args:
        server_hours_training: The total server hours during the training phase of the model.
        total_power: The total power consumption of the server in kW.
        datacenter_pue: The power usage effectiveness of the datacenter.

    Returns:
        The total training energy during the training phase of the model in kiloWatt-hour.
    """
    return (total_power * server_hours_training) * datacenter_pue


###########################################
# # OLD version:
# @dag.asset
# def total_training_energy(
#     training_flops: ValueOrRange,  # in FLOP
#     energy_per_flops: float,  # in Watt / (FLOP/s)
#     datacenter_pue: ValueOrRange,
# ) -> ValueOrRange:
#     """
#     Total training energy during the training phase of the model.

#     Args:
#         training_flops: The total training FLOPs during the training phase of the model.
#         energy_per_flops: The energy consumption per FLOP/s.
#         datacenter_pue: The power usage effectiveness of the datacenter.

#     Returns:
#         The total training energy during the training phase of the model in kiloWatt-hour.
#     """

#     return (
#         training_flops * energy_per_flops / (3600 * 1000)
#     ) * datacenter_pue  # convert from Watt-second to kiloWatt-hour
# # TODO: add new ratio to take into account the energy consumption outside of the GPU
###########################################


@dag.asset
def request_energy_training(
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
def request_usage_gwp_training(
    request_energy_training: ValueOrRange,  # in kiloWatt-hour
    if_electricity_mix_gwp: float,  # in kgCO2eq / kWh
) -> ValueOrRange:
    """
    The GWP usage impact of the training phase allocated to each request.

    Args:
        request_energy_training: The training energy consumption allocated to each request in kiloWatt-hour.
        if_electricity_mix_gwp: The GWP impact factor of electricity consumption in kgCO2eq / kWh.

    Returns:
        The GWP usage impact of the training phase allocated to each request in kgCO2eq.
    """
    return request_energy_training * if_electricity_mix_gwp


@dag.asset
def request_usage_adpe_training(
    request_energy_training: ValueOrRange,  # in kiloWatt-hour
    if_electricity_mix_adpe: float,  # in kgSbeq / kWh
) -> ValueOrRange:
    """
    The ADPe usage impact of the training phase allocated to each request.

    Args:
        request_energy_training: The training energy consumption allocated to each request in kiloWatt-hour.
        if_electricity_mix_adpe: The ADPe impact factor of electricity consumption in kgSbeq / kWh.

    Returns:
        The ADPe usage impact of the training phase allocated to each request in kgSbeq.
    """
    return request_energy_training * if_electricity_mix_adpe


@dag.asset
def request_usage_pe_training(
    request_energy_training: ValueOrRange,  # in kiloWatt-hour
    if_electricity_mix_pe: float,  # in MJ / kWh
) -> ValueOrRange:
    """
    The PE usage impact of the training phase allocated to each request.

    Args:
        request_energy_training: The training energy consumption allocated to each request in kiloWatt-hour.
        if_electricity_mix_pe: The PE impact factor of electricity consumption in MJ / kWh.

    Returns:
        The PE usage impact of the training phase allocated to each request in MJ.
    """
    return request_energy_training * if_electricity_mix_pe


@dag.asset
def request_usage_wcf_training(
    request_energy_training: ValueOrRange,  # in kiloWatt-hour
    if_electricity_mix_wue: float,  # in L / kWh
    datacenter_wue: float,
    datacenter_pue: float,
) -> ValueOrRange:
    """
    The WCF usage impact of the training phase allocated to each request.

    Args:
        request_energy_training: The training energy consumption allocated to each request in kiloWatt-hour.
        if_electricity_mix_wue: The WCF impact factor of electricity consumption in L / kWh.
        datacenter_wue: Water Usage Effectiveness of the data center in L/kWh.
        datacenter_pue: Power Usage Effectiveness of the data center.

    Returns:
        The WCF usage impact of the training phase allocated to each request in Liter.
    """
    return request_energy_training * (
        datacenter_wue + datacenter_pue * if_electricity_mix_wue
    )
# TODO: check the formula for water usage


@dag.asset
def total_embodied_gwp_training(
    server_hours_training: ValueOrRange,
    server_embodied_gwp: float,
    gpu_embodied_gwp: float,
    server_gpu_count: float,
    network_embodied_gwp: float,
    server_lifetime: float,
    network_lifetime: float,
) -> ValueOrRange:
    """
    The total GWP embodied impact during the training phase of the model.

    Args:
        server_hours_training: The total server hours during the training phase of the model.
        server_embodied_gwp: The GWP embodied impact of the server in kgCO2eq.
        gpu_embodied_gwp: The GWP embodied impact of the GPU in kgCO2eq.
        network_embodied_gwp: The GWP embodied impact of the network equipment in kgCO2eq.
        server_lifetime: Lifetime duration of the server in seconds.
        network_lifetime: Lifetime duration of the network equipment in seconds.

    Returns:
        The total GWP embodied impact during the training phase of the model in kgCO2eq.
    """
    return server_hours_training * (
        server_embodied_gwp + gpu_embodied_gwp * server_gpu_count
    ) / (server_lifetime / 3600) + server_hours_training * network_embodied_gwp / (
        network_lifetime / 3600
    )


@dag.asset
def total_embodied_adpe_training(
    server_hours_training: ValueOrRange,
    server_embodied_adpe: float,
    gpu_embodied_adpe: float,
    server_gpu_count: float,
    network_embodied_adpe: float,
    server_lifetime: float,
    network_lifetime: float,
) -> ValueOrRange:
    """
    The total ADPe embodied impact during the training phase of the model.

    Args:
        server_hours_training: The total server hours during the training phase of the model.
        server_embodied_adpe: The ADPe embodied impact of the server in kgSbeq.
        gpu_embodied_adpe: The ADPe embodied impact of the GPU in kgSbeq.
        network_embodied_adpe: The ADPe embodied impact of the network equipment in kgSbeq.
        server_lifetime: Lifetime duration of the server in seconds.
        network_lifetime: Lifetime duration of the network equipment in seconds.

    Returns:
        The total ADPe embodied impact during the training phase of the model in kgSbeq.
    """
    return server_hours_training * (
        server_embodied_adpe + gpu_embodied_adpe * server_gpu_count
    ) / (server_lifetime / 3600) + server_hours_training * network_embodied_adpe / (
        network_lifetime / 3600
    )


@dag.asset
def total_embodied_pe_training(
    server_hours_training: ValueOrRange,
    server_embodied_pe: float,              
    gpu_embodied_pe: float,
    server_gpu_count: float,
    network_embodied_pe: float,
    server_lifetime: float,
    network_lifetime: float,
) -> ValueOrRange:
    """
    The total PE embodied impact during the training phase of the model.

    Args:
        server_hours_training: The total server hours during the training phase of the model.
        server_embodied_pe: The PE embodied impact of the server in MJ.
        gpu_embodied_pe: The PE embodied impact of the GPU in MJ.
        network_embodied_pe: The PE embodied impact of the network equipment in MJ.
        server_lifetime: Lifetime duration of the server in seconds.
        network_lifetime: Lifetime duration of the network equipment in seconds.

    Returns:
        The total PE embodied impact during the training phase of the model in MJ.
    """
    return server_hours_training * (
        server_embodied_pe + gpu_embodied_pe * server_gpu_count
    ) / (server_lifetime / 3600) + server_hours_training * network_embodied_pe / (
        network_lifetime / 3600
    )


@dag.asset
def total_embodied_wcf_training(
    server_hours_training: ValueOrRange,
    server_embodied_wcf: float,         
    gpu_embodied_wcf: float,
    server_gpu_count: float,
    network_embodied_wcf: float,
    server_lifetime: float,
    network_lifetime: float,
) -> ValueOrRange:
    """
    The total WCF embodied impact during the training phase of the model.

    Args:
        server_hours_training: The total server hours during the training phase of the model.
        server_embodied_wcf: The WCF embodied impact of the server in L.
        gpu_embodied_wcf: The WCF embodied impact of the GPU in L.
        network_embodied_wcf: The WCF embodied impact of the network equipment in L.
        server_lifetime: Lifetime duration of the server in seconds.
        network_lifetime: Lifetime duration of the network equipment in seconds.

    Returns:
        The total WCF embodied impact during the training phase of the model in L.
    """
    return server_hours_training * (
        server_embodied_wcf + gpu_embodied_wcf * server_gpu_count
    ) / (server_lifetime / 3600) + server_hours_training * network_embodied_wcf / (
        network_lifetime / 3600
    )


@dag.asset
def request_embodied_gwp_training(
    total_embodied_gwp_training: ValueOrRange,
    total_output_tokens: ValueOrRange,
    output_token_count: float,
) -> ValueOrRange:
    """
    The GWP embodied impact of the training phase allocated to each request.

    Args:
        total_embodied_gwp_training: The total GWP embodied impact during the training phase of the model in kgCO2eq.
        total_output_tokens: The total output token count during the lifespan of the model.
        output_token_count: The number of tokens generated in the request.

    Returns:
        The GWP embodied impact of the training phase allocated to each request in kgCO2eq.
    """
    return (total_embodied_gwp_training / total_output_tokens) * output_token_count


@dag.asset
def request_embodied_adpe_training(
    total_embodied_adpe_training: ValueOrRange,
    total_output_tokens: ValueOrRange,
    output_token_count: float,
) -> ValueOrRange:
    """
    The ADPe embodied impact of the training phase allocated to each request.

    Args:
        total_embodied_adpe_training: The total ADPe embodied impact during the training phase of the model in kgSbeq.
        total_output_tokens: The total output token count during the lifespan of the model.
        output_token_count: The number of tokens generated in the request.

    Returns:
        The ADPe embodied impact of the training phase allocated to each request in kgSbeq.
    """
    return (total_embodied_adpe_training / total_output_tokens) * output_token_count


@dag.asset
def request_embodied_pe_training(
    total_embodied_pe_training: ValueOrRange,
    total_output_tokens: ValueOrRange,
    output_token_count: float,
) -> ValueOrRange:
    """
    The PE embodied impact of the training phase allocated to each request.

    Args:
        total_embodied_pe_training: The total PE embodied impact during the training phase of the model in MJ.
        total_output_tokens: The total output token count during the lifespan of the model.
        output_token_count: The number of tokens generated in the request.
    
    Returns:
        The PE embodied impact of the training phase allocated to each request in MJ.
    """
    return (total_embodied_pe_training / total_output_tokens) * output_token_count


@dag.asset
def request_embodied_wcf_training(
    total_embodied_wcf_training: ValueOrRange,
    total_output_tokens: ValueOrRange,
    output_token_count: float,) -> ValueOrRange:
    """
    The WCF embodied impact of the training phase allocated to each request.    
    
    Args:        
        total_embodied_wcf_training: The total WCF embodied impact during the training phase of the model in L.
        total_output_tokens: The total output token count during the lifespan of the model.
        output_token_count: The number of tokens generated in the request.

    Returns:
        The WCF embodied impact of the training phase allocated to each request in L.
    """
    return (total_embodied_wcf_training / total_output_tokens) * output_token_count


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
    # energy_per_flops: Optional[float] = ENERGY_PER_FLOPS,
    flops_per_GPU: Optional[float] = FLOPS_PER_GPU,
    total_power: Optional[float] = SERVER_GPU_NETWORK_POWER,
    gpu_embodied_gwp: Optional[float] = GPU_EMBODIED_IMPACT_GWP,
    gpu_embodied_adpe: Optional[float] = GPU_EMBODIED_IMPACT_ADPE,
    gpu_embodied_pe: Optional[float] = GPU_EMBODIED_IMPACT_PE,
    gpu_embodied_wcf: Optional[float] = GPU_EMBODIED_IMPACT_WCF,
    server_gpu_count: Optional[int] = SERVER_GPUS,
    server_embodied_gwp: Optional[float] = SERVER_EMBODIED_IMPACT_GWP,
    server_embodied_adpe: Optional[float] = SERVER_EMBODIED_IMPACT_ADPE,
    server_embodied_pe: Optional[float] = SERVER_EMBODIED_IMPACT_PE,
    server_embodied_wcf: Optional[float] = SERVER_EMBODIED_IMPACT_WCF,
    server_lifetime: Optional[float] = HARDWARE_LIFESPAN,
    network_embodied_gwp: Optional[float] = NETWORK_EMBODIED_IMPACT_GWP,
    network_embodied_adpe: Optional[float] = NETWORK_EMBODIED_IMPACT_ADPE,
    network_embodied_pe: Optional[float] = NETWORK_EMBODIED_IMPACT_PE,
    network_embodied_wcf: Optional[float] = NETWORK_EMBODIED_IMPACT_WCF,
    network_lifetime: Optional[float] = NETWORK_LIFESPAN,
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
        flops_per_GPU: The number of FLOPs that can be performed in one second per GPU.
        server_gpu_count: The number of GPUs in the server.
        total_power: The total power consumption of the server (including sever, gpu and network) in kW.

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
        # energy_per_flops=energy_per_flops,
        flops_per_GPU=flops_per_GPU,
        server_gpu_count=server_gpu_count,
        total_power=total_power,
        gpu_embodied_gwp=gpu_embodied_gwp,
        gpu_embodied_adpe=gpu_embodied_adpe,
        gpu_embodied_pe=gpu_embodied_pe,
        gpu_embodied_wcf=gpu_embodied_wcf,          
        server_embodied_gwp=server_embodied_gwp,
        server_embodied_adpe=server_embodied_adpe,
        server_embodied_pe=server_embodied_pe,
        server_embodied_wcf=server_embodied_wcf,
        server_lifetime=server_lifetime,
        network_embodied_gwp=network_embodied_gwp,
        network_embodied_adpe=network_embodied_adpe,
        network_embodied_pe=network_embodied_pe,
        network_embodied_wcf=network_embodied_wcf,
        network_lifetime=network_lifetime,
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
        "request_energy_training",
        "request_usage_gwp_training",
        "request_usage_adpe_training",
        "request_usage_pe_training",
        "request_usage_wcf_training",
        "request_embodied_gwp_training",
        "request_embodied_adpe_training",
        "request_embodied_pe_training",
        "request_embodied_wcf_training"
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

    energy = Energy(value=results["request_energy_training"])
    gwp_usage = GWP(value=results["request_usage_gwp_training"])
    adpe_usage = ADPe(value=results["request_usage_adpe_training"])
    pe_usage = PE(value=results["request_usage_pe_training"])
    wcf_usage = WCF(value=results["request_usage_wcf_training"])
    gwp_embodied = GWP(value=results["request_embodied_gwp_training"])
    adpe_embodied = ADPe(value=results["request_embodied_adpe_training"])
    pe_embodied = PE(value=results["request_embodied_pe_training"])
    wcf_embodied = WCF(value=results["request_embodied_wcf_training"])
    # TODO: add embodied impacts for training

    # print(results["request_energy_training"])

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
