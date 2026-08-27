import datetime

from ecologits.impacts.llm_data_storage_training import compute_llm_train_data_storage_impacts


def test_compute_llm_train_data_storage_impacts_includes_embodied_wcf() -> None:
    impacts = compute_llm_train_data_storage_impacts(
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
    assert impacts.wcf.value > impacts.usage.wcf.value
    assert impacts.embodied.wcf is not None
