import datetime

from ecologits.impacts.llm_training import compute_llm_train_impacts, total_output_tokens, training_flops
from ecologits.utils.range_value import RangeValue


def test_training_low_level_ranges_are_ordered() -> None:
    flops = training_flops(datetime.datetime(2024, 1, 1), RangeValue(min=7, max=70))
    tokens = total_output_tokens(1e9, 1e12, 0.7, 100, RangeValue(min=7, max=70))

    assert isinstance(flops, RangeValue)
    assert isinstance(tokens, RangeValue)
    assert flops.min < flops.max
    assert tokens.min < tokens.max


def test_compute_llm_train_impacts_returns_compatible_impacts() -> None:
    impacts = compute_llm_train_impacts(
        publication_date=datetime.datetime(2024, 1, 1),
        compute_capacity={"2024": 1},
        number_of_active_models={"2024": 10},
        model_active_parameter_count=7,
        model_total_parameter_count=7,
        output_token_count=100,
        if_electricity_mix_adpe=0.001,
        if_electricity_mix_pe=10,
        if_electricity_mix_gwp=0.5,
        if_electricity_mix_wue=5,
        datacenter_pue=1.2,
        datacenter_wue=0.3,
    )

    assert impacts.energy.value > 0
    assert impacts.gwp.value > impacts.usage.gwp.value
    assert impacts.wcf.value == impacts.usage.wcf.value
