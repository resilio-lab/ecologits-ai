import json
import os
from dataclasses import dataclass
from typing import Any, Optional

from range_value import RangeValue


@dataclass
class ProviderConfig:
    """
    Provider configuration containing datacenter metadata
    
    Attributes:
        name: Provider name (e.g. "openai", "anthropic")
        datacenter_location: ISO 3166-1 alpha-3 code of datacenter location
        datacenter_pue: Power Usage Effectiveness (ratio, can be scalar or range)
        datacenter_wue: Water Usage Effectiveness (can be scalar or range)
        compute_capacity: Yearly compute capacity values
        number_of_active_models: Yearly number of active models values
    """
    name: str
    datacenter_location: str
    datacenter_pue: float | RangeValue
    datacenter_wue: float | RangeValue
    compute_capacity: dict[str, int | float | None]
    number_of_active_models: dict[str, int | float | None]


class ProviderRepository:
    """
    Repository of provider configurations
    """

    def __init__(self, providers: list[ProviderConfig]) -> None:
        self.__providers: dict[str, ProviderConfig] = {}
        for provider in providers:
            self.__providers[provider.name] = provider

    def find_provider(self, name: str) -> Optional[ProviderConfig]:
        """Find a provider configuration by name"""
        return self.__providers.get(name)

    def list_providers(self) -> list[ProviderConfig]:
        """List all provider configurations"""
        return list(self.__providers.values())

    @classmethod
    def from_json(cls, filepath: Optional[str] = None) -> "ProviderRepository":
        """
        Load provider configurations from JSON file
        
        JSON format:
        {
          "providers": [
            {
              "name": "anthropic",
              "datacenter_location": "USA",
              "datacenter_pue": {"type": "range", "min": 1.09, "max": 1.14},
              "datacenter_wue": {"type": "scalar", "value": 0.999}
              “
            },
            ...
          ]
        }
        """
        if filepath is None:
            filepath = os.path.join(
                os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data", "providers.json"
            )
        
        providers = []
        with open(filepath) as fd:
            data = json.load(fd)
            
            if "providers" in data:
                for provider_data in data["providers"]:
                    # Parse PUE value
                    pue_data = provider_data["datacenter_pue"]
                    if pue_data.get("type") == "range":
                        datacenter_pue = RangeValue(min=pue_data["min"], max=pue_data["max"])
                    else:
                        datacenter_pue = pue_data["value"]
                    
                    # Parse WUE value
                    wue_data = provider_data["datacenter_wue"]
                    if wue_data.get("type") == "range":
                        datacenter_wue = RangeValue(min=wue_data["min"], max=wue_data["max"])
                    else:
                        datacenter_wue = wue_data["value"]
                    
                    config = ProviderConfig(
                        name=provider_data["name"],
                        datacenter_location=provider_data["datacenter_location"],
                        datacenter_pue=datacenter_pue,
                        datacenter_wue=datacenter_wue,
                        compute_capacity=provider_data.get("compute_capacity", {}),
                        number_of_active_models=provider_data.get("number_of_active_models", {}),
                    )
                    providers.append(config)
            
        return cls(providers)


providers = ProviderRepository.from_json()

