import pytest

from ecologits.tracers.utils import llm_impacts, llm_train_data_storage_impacts, llm_train_impacts
from ecologits.utils.range_value import RangeValue


def to_float(value: float | RangeValue) -> float:
    return float(value.min if isinstance(value, RangeValue) else value)


def test_llm_impacts_golden_values() -> None:
    # Golden values pinning the end-to-end inference pipeline (model catalog,
    # provider config, and impact formulas). They must be updated whenever the
    # catalog, the constants, or the methodology intentionally change.
    impacts = llm_impacts(provider="openai", model_name="gpt-4o-mini", output_token_count=100,
                          request_latency=2.0)

    assert impacts.energy is not None
    assert impacts.gwp is not None
    assert to_float(impacts.energy.value) == pytest.approx(7.1823932951712785e-06)
    assert to_float(impacts.gwp.value) == pytest.approx(3.1027854576210317e-06)
    assert to_float(impacts.adpe.value) == pytest.approx(1.8940955249903765e-11)
    assert to_float(impacts.pe.value) == pytest.approx(7.370519936582833e-05)
    assert to_float(impacts.wcf.value) == pytest.approx(1.283599675312459e-04)
    assert impacts.embodied is not None
    assert impacts.embodied.wcf is not None
    assert to_float(impacts.embodied.wcf.value) > 0


def test_llm_train_impacts_golden_values() -> None:
    impacts = llm_train_impacts(provider="openai", model_name="gpt-4o-mini", output_token_count=100)

    assert impacts.energy is not None
    assert impacts.gwp is not None
    assert impacts.wcf is not None
    assert to_float(impacts.energy.value) == pytest.approx(2.0092552362976884e-11)
    assert to_float(impacts.gwp.value) == pytest.approx(8.481430148213055e-12)
    assert to_float(impacts.wcf.value) == pytest.approx(3.1407694860625405e-10)


def test_llm_train_data_storage_impacts_golden_values() -> None:
    impacts = llm_train_data_storage_impacts(provider="openai", model_name="gpt-4o-mini", output_token_count=100)

    assert impacts.energy is not None
    assert impacts.embodied.gwp is not None
    assert to_float(impacts.energy.value) == pytest.approx(4.1676283022914094e-16)
    assert to_float(impacts.embodied.gwp.value) == pytest.approx(2.673001209262244e-15)
