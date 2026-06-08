"""Shared defaults and coefficients for calculations."""


###############################
# defaults provided by Ecologits
MODEL_QUANTIZATION_BITS = 16

GPU_ENERGY_ALPHA = 1.1665273170451914e-06
GPU_ENERGY_BETA = -0.011205921025579175
GPU_ENERGY_GAMMA = 4.052928146734005e-05

LATENCY_ALPHA = 0.0006785088094353663
LATENCY_BETA = 0.0003119310311688259
LATENCY_GAMMA = 0.019473717579473387 # latency coefficient has been deprecated in the latest version of the modelling

GPU_MEMORY = 80  # GB
GPU_EMBODIED_IMPACT_GWP = 273
GPU_EMBODIED_IMPACT_ADPE = 0.00895
GPU_EMBODIED_IMPACT_PE = 3721

SERVER_GPUS = 8
SERVER_POWER = 1.2  # kW
SERVER_EMBODIED_IMPACT_GWP = 5700
SERVER_EMBODIED_IMPACT_ADPE = 0.37
SERVER_EMBODIED_IMPACT_PE = 70000

HARDWARE_LIFESPAN = 3 * 365 * 24 * 60 * 60

BATCH_SIZE = 64
###############################


###############################
# defaults for training phase
FLOPS_PER_WATT = 1.4e12  #  FLOP/s / Watt  #  take NVIDIA H100 as reference (https://epoch.ai/data-insights/ml-hardware-energy-efficiency)
ENERGY_PER_FLOPS = 1 / FLOPS_PER_WATT  # Watt / (FLOP/s)
GPU_UTILIZATION_RATE = 0.7  # the share of time that the GPU is actively performing
INFERENCE_COMPUTE_SHARE = 0.8 # the share of inference compute capacity in the total company's yearly compute capacity, used to estimate the inference compute capacity from the total compute capacity
MODEL_LIFESPAN = 2  # in years, the time span during which the model is actively used after its publication







##############################

__all__ = [
    "MODEL_QUANTIZATION_BITS",
    "GPU_ENERGY_ALPHA",
    "GPU_ENERGY_BETA",
    "GPU_ENERGY_GAMMA",
    "LATENCY_ALPHA",
    "LATENCY_BETA",
    "LATENCY_GAMMA",
    "GPU_MEMORY",
    "GPU_EMBODIED_IMPACT_GWP",
    "GPU_EMBODIED_IMPACT_ADPE",
    "GPU_EMBODIED_IMPACT_PE",
    "SERVER_GPUS",
    "SERVER_POWER",
    "SERVER_EMBODIED_IMPACT_GWP",
    "SERVER_EMBODIED_IMPACT_ADPE",
    "SERVER_EMBODIED_IMPACT_PE",
    "HARDWARE_LIFESPAN",
    "BATCH_SIZE",
    # above is for inference impacts modelling (eologits), below is for training impacts modelling
    "FLOPS_PER_WATT",
    "ENERGY_PER_FLOPS",
    "GPU_UTILIZATION_RATE",
    "INFERENCE_COMPUTE_SHARE",
    "MODEL_LIFESPAN",
]