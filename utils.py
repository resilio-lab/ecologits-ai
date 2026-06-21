from __future__ import annotations
from typing import Optional, Union

from pydantic import BaseModel

from data_repository.electricity_mix_repository import electricity_mixes
from data_repository.model_repository import ParametersMoE, models
from data_repository.provider_repository import providers

from impacts.llm_inference import compute_llm_infer_impacts
from impacts.llm_training import compute_llm_train_impacts
from impacts.llm_data_storage_training import compute_llm_train_data_storage_impacts
from impacts.modeling import GWP, PE, WCF, ADPe, Embodied, Energy, Usage
from log import logger
from status_messages import ErrorMessage, ModelNotRegisteredError, WarningMessage, ZoneNotRegisteredError
from range_value import RangeValue


class ImpactsOutput(BaseModel):
    """
    Impacts output data model.

    Attributes:
        energy: Total energy consumption
        gwp: Total Global Warming Potential (GWP) impact
        adpe: Total Abiotic Depletion Potential for Elements (ADPe) impact
        pe: Total Primary Energy (PE) impact
        wcf: Total Water Consumption Footprint (WCF) impact
        usage: Impacts for the usage phase
        embodied: Impacts for the embodied phase
        warnings: List of warnings
        errors: List of errors
    """
    energy: Energy | None = None
    gwp: GWP | None = None
    adpe: ADPe | None = None
    pe: PE | None = None
    wcf: WCF | None = None
    usage: Usage | None = None
    embodied: Embodied | None = None
    warnings: list[WarningMessage] | None = None
    errors: list[ErrorMessage] | None = None

    @property
    def has_warnings(self) -> bool:
        return isinstance(self.warnings, list) and len(self.warnings) > 0

    @property
    def has_errors(self) -> bool:
        return isinstance(self.errors, list) and len(self.errors) > 0

    def add_warning(self, warning: WarningMessage) -> None:
        if self.warnings is None:
            self.warnings = []
        self.warnings.append(warning)

    def add_errors(self, error: ErrorMessage) -> None:
        if self.errors is None:
            self.errors = []
        self.errors.append(error)


def llm_infer_impacts(
    provider: str,
    model_name: str,
    output_token_count: int,
    request_latency: Optional[float] = None,
    electricity_mix_zone: str | None  = None,
) -> ImpactsOutput:
    """
    High-level function to compute the inferenceimpacts of an LLM generation request.

    Args:
        provider: Name of the provider.
        model_name: Name of the LLM used.
        output_token_count: Number of generated tokens.
        request_latency: Measured request latency in seconds.
        electricity_mix_zone: ISO 3166-1 alpha-3 code of the electricity mix zone (WOR by default).

    Returns:
        The inferenceimpacts of an LLM generation request.
    """

    model = models.find_model(provider=provider, model_name=model_name)
    if model is None:
        error = ModelNotRegisteredError(message=f"Could not find model `{model_name}` for {provider} provider.")
        logger.warning_once(str(error))
        return ImpactsOutput(errors=[error])

    if isinstance(model.architecture.parameters, ParametersMoE):
        model_total_params = model.architecture.parameters.total
        model_active_params = model.architecture.parameters.active
    else:
        model_total_params = model.architecture.parameters
        model_active_params = model.architecture.parameters

    provider_config = providers.find_provider(provider)
    datacenter_location = provider_config.datacenter_location
    datacenter_pue = provider_config.datacenter_pue
    datacenter_wue = provider_config.datacenter_wue

    if electricity_mix_zone is None:
        electricity_mix_zone = datacenter_location
    if electricity_mix_zone is None:
        electricity_mix_zone = "WOR"
    if_electricity_mix = electricity_mixes.find_electricity_mix(zone=electricity_mix_zone)
    if if_electricity_mix is None:
        error = ZoneNotRegisteredError(message=f"Could not find electricity mix for `{electricity_mix_zone}` zone.")
        logger.warning_once(str(error))
        return ImpactsOutput(errors=[error])

    impacts = compute_llm_infer_impacts(
        model_active_parameter_count=model_active_params,
        model_total_parameter_count=model_total_params,
        output_token_count=output_token_count,
        if_electricity_mix_adpe=if_electricity_mix.adpe,
        if_electricity_mix_pe=if_electricity_mix.pe,
        if_electricity_mix_gwp=if_electricity_mix.gwp,
        if_electricity_mix_wue=if_electricity_mix.wue,
        datacenter_pue=datacenter_pue,
        datacenter_wue=datacenter_wue,
        request_latency=request_latency,
        tps=model.deployment.tps if model.deployment else None,
        ttft=model.deployment.ttft if model.deployment else None,
    )
    impacts = ImpactsOutput.model_validate(impacts.model_dump())

    if model.has_warnings:
        for w in model.warnings:
            logger.warning_once(str(w))
            impacts.add_warning(w)

    return impacts


def llm_train_impacts(
    provider: str,
    model_name: str,
    output_token_count: int,
    electricity_mix_zone: str | None  = None,
) -> ImpactsOutput:
    """
    High-level function to compute the training impacts of an LLM training request.

    Args:
        provider: Name of the provider.
        model_name: Name of the LLM used.
        output_token_count: Number of generated tokens.
        electricity_mix_zone: ISO 3166-1 alpha-3 code of the electricity mix zone (WOR by default).

    Returns:
        The training impacts of an LLM training request.
    """

    model = models.find_model(provider=provider, model_name=model_name)
    if model is None:
        error = ModelNotRegisteredError(message=f"Could not find model `{model_name}` for {provider} provider.")
        logger.warning_once(str(error))
        return ImpactsOutput(errors=[error])

    if isinstance(model.architecture.parameters, ParametersMoE):
        model_total_params = model.architecture.parameters.total
        model_active_params = model.architecture.parameters.active
    else:
        model_total_params = model.architecture.parameters
        model_active_params = model.architecture.parameters

    provider_config = providers.find_provider(provider)
    datacenter_location = provider_config.datacenter_location
    datacenter_pue = provider_config.datacenter_pue
    datacenter_wue = provider_config.datacenter_wue
    
    if electricity_mix_zone is None:
        electricity_mix_zone = datacenter_location
    if electricity_mix_zone is None:
        electricity_mix_zone = "WOR"
    if_electricity_mix = electricity_mixes.find_electricity_mix(zone=electricity_mix_zone)
    if if_electricity_mix is None:
        error = ZoneNotRegisteredError(message=f"Could not find electricity mix for `{electricity_mix_zone}` zone.")
        logger.warning_once(str(error))
        return ImpactsOutput(errors=[error])

    impacts = compute_llm_train_impacts(
        publication_date=model.publication_date,
        compute_capacity=provider_config.compute_capacity,
        number_of_active_models=provider_config.number_of_active_models,
        model_active_parameter_count=model_active_params,
        model_total_parameter_count=model_total_params,
        output_token_count=output_token_count,
        if_electricity_mix_adpe=if_electricity_mix.adpe,
        if_electricity_mix_pe=if_electricity_mix.pe,
        if_electricity_mix_gwp=if_electricity_mix.gwp,
        if_electricity_mix_wue=if_electricity_mix.wue,
        datacenter_pue=datacenter_pue,
        datacenter_wue=datacenter_wue,
    )
    impacts = ImpactsOutput.model_validate(impacts.model_dump())

    if model.has_warnings:
        for w in model.warnings:
            logger.warning_once(str(w))
            impacts.add_warning(w)

    return impacts


def llm_train_data_storage_impacts(
    provider: str,
    model_name: str,
    output_token_count: int,
    electricity_mix_zone: str | None  = None,
) -> ImpactsOutput:
    """
    High-level function to compute the training - data storage impacts of an LLM training request.

    Args:
        provider: Name of the provider.
        model_name: Name of the LLM used.
        output_token_count: Number of generated tokens.
        electricity_mix_zone: ISO 3166-1 alpha-3 code of the electricity mix zone (WOR by default).

    Returns:
        The training - data storage impacts of an LLM training request.
    """

    model = models.find_model(provider=provider, model_name=model_name)
    if model is None:
        error = ModelNotRegisteredError(message=f"Could not find model `{model_name}` for {provider} provider.")
        logger.warning_once(str(error))
        return ImpactsOutput(errors=[error])

    if isinstance(model.architecture.parameters, ParametersMoE):
        model_total_params = model.architecture.parameters.total
        model_active_params = model.architecture.parameters.active
    else:
        model_total_params = model.architecture.parameters
        model_active_params = model.architecture.parameters

    provider_config = providers.find_provider(provider)
    datacenter_location = provider_config.datacenter_location
    datacenter_pue = provider_config.datacenter_pue
    datacenter_wue = provider_config.datacenter_wue
    
    if electricity_mix_zone is None:
        electricity_mix_zone = datacenter_location
    if electricity_mix_zone is None:
        electricity_mix_zone = "WOR"
    if_electricity_mix = electricity_mixes.find_electricity_mix(zone=electricity_mix_zone)
    if if_electricity_mix is None:
        error = ZoneNotRegisteredError(message=f"Could not find electricity mix for `{electricity_mix_zone}` zone.")
        logger.warning_once(str(error))
        return ImpactsOutput(errors=[error])

    impacts = compute_llm_train_data_storage_impacts(
        publication_date=model.publication_date,
        compute_capacity=provider_config.compute_capacity,
        number_of_active_models=provider_config.number_of_active_models,
        model_active_parameter_count=model_active_params,
        model_total_parameter_count=model_total_params,
        output_token_count=output_token_count,
        if_electricity_mix_adpe=if_electricity_mix.adpe,
        if_electricity_mix_pe=if_electricity_mix.pe,
        if_electricity_mix_gwp=if_electricity_mix.gwp,
        if_electricity_mix_wue=if_electricity_mix.wue,
        datacenter_pue=datacenter_pue,
        datacenter_wue=datacenter_wue,
    )
    impacts = ImpactsOutput.model_validate(impacts.model_dump())

    if model.has_warnings:
        for w in model.warnings:
            logger.warning_once(str(w))
            impacts.add_warning(w)

    return impacts






