import datetime

from ecologits.impacts.llm_data_storage_training import compute_llm_train_data_storage_impacts
from ecologits.utils.range_value import RangeValue


def _to_float(value: float | RangeValue) -> float:
    return float(value.min if isinstance(value, RangeValue) else value)


def _storage_kwargs(**overrides):  # type: ignore[no-untyped-def]
    kwargs = {
        "publication_date": datetime.datetime(2024, 1, 1),
        "compute_capacity": {"2024": 1},
        "number_of_active_models": {"2024": 10},
        "model_active_parameter_count": 7,
        "model_total_parameter_count": 7,
        "output_token_count": 100,
        "if_electricity_mix_adpe": 0.001,
        "if_electricity_mix_pe": 10,
        "if_electricity_mix_gwp": 0.5,
        "if_electricity_mix_wue": 5,
        "datacenter_pue": 1.2,
        "datacenter_wue": 0.3,
    }
    kwargs.update(overrides)
    return kwargs


def test_compute_llm_train_data_storage_impacts_includes_embodied_wcf() -> None:
    impacts = compute_llm_train_data_storage_impacts(**_storage_kwargs())

    assert impacts.energy.value > 0
    assert impacts.wcf.value > impacts.usage.wcf.value
    assert impacts.embodied.wcf is not None


def test_compute_llm_train_data_storage_impacts_monotonicity() -> None:
    prev_energy = 0.0
    for total_params in [7, 13, 70]:
        impacts = compute_llm_train_data_storage_impacts(**_storage_kwargs(
            model_active_parameter_count=total_params,
            model_total_parameter_count=total_params,
        ))
        energy = _to_float(impacts.energy.value)
        assert energy > prev_energy
        prev_energy = energy

    prev_energy = 0.0
    for output_tokens in [10, 100, 1000]:
        impacts = compute_llm_train_data_storage_impacts(**_storage_kwargs(output_token_count=output_tokens))
        energy = _to_float(impacts.energy.value)
        assert energy > prev_energy
        prev_energy = energy


def test_compute_llm_train_data_storage_impacts_embodied_and_ranges() -> None:
    impacts = compute_llm_train_data_storage_impacts(**_storage_kwargs(
        model_active_parameter_count=RangeValue(min=7, max=70),
        model_total_parameter_count=RangeValue(min=7, max=70),
    ))

    assert _to_float(impacts.embodied.gwp.value) > 0
    assert _to_float(impacts.embodied.adpe.value) >= 0
    assert _to_float(impacts.embodied.pe.value) >= 0
    assert _to_float(impacts.embodied.wcf.value) > 0
    assert isinstance(impacts.energy.value, RangeValue)
    assert impacts.energy.value.min < impacts.energy.value.max
