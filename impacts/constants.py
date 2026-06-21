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
GPU_EMBODIED_IMPACT_WCF = 101500  # L eq, ResilioDB
# TODO: check the unit of WUE - if it is in L eq.

SERVER_GPUS = 8
SERVER_POWER = 1.2  # kW
SERVER_EMBODIED_IMPACT_GWP = 5700
SERVER_EMBODIED_IMPACT_ADPE = 0.37
SERVER_EMBODIED_IMPACT_PE = 70000
# above is for SERVER without GPU
SERVER_EMBODIED_IMPACT_WCF = 1555315  # L eq, ResilioDB
# TODO: check if it's for Server\GPU or Server+GPU.

HARDWARE_LIFESPAN = 3 * 365 * 24 * 60 * 60

BATCH_SIZE = 64

SERVER_GPU_POWER = 6.8 # kW, the total power consumption of a server with 8 GPUs (1.2 + 8 * 0.7)
###############################


###############################
# defaults for network equipment
FIREWALL_POWER = 0.09  # kW, ADEME
FIREWALL_USAGE_RATIO = 0.0358  # ADEME-ARCEP
ROUTER_POWER = 0.09  # kW, ADEME
ROUTER_USAGE_RATIO = 0.286  # ADEME-ARCEP
SWITCH_POWER = 0.09  # kW, ADEME
SWITCH_USAGE_RATIO = 1.468  # ADEME-ARCEP

NETWORK_POWER = FIREWALL_POWER * FIREWALL_USAGE_RATIO + ROUTER_POWER * ROUTER_USAGE_RATIO + SWITCH_POWER * SWITCH_USAGE_RATIO

FIREWALL_EMBODIED_IMPACT_GWP = 333.8  # kgCO2eq, ResilioDB
ROUTER_EMBODIED_IMPACT_GWP = 403  # kgCO2eq, ResilioDB
SWITCH_EMBODIED_IMPACT_GWP = 363  # kgCO2eq, ResilioDB

NETWORK_EMBODIED_IMPACT_GWP = FIREWALL_EMBODIED_IMPACT_GWP * FIREWALL_USAGE_RATIO + ROUTER_EMBODIED_IMPACT_GWP * ROUTER_USAGE_RATIO + SWITCH_EMBODIED_IMPACT_GWP * SWITCH_USAGE_RATIO

FIREWALL_EMBODIED_IMPACT_WCF = 94100  # L eq, ResilioDB
ROUTER_EMBODIED_IMPACT_WCF = 115765  # L eq, ResilioDB
SWITCH_EMBODIED_IMPACT_WCF = 104795  # L eq, ResilioDB

NETWORK_EMBODIED_IMPACT_WCF = FIREWALL_EMBODIED_IMPACT_WCF * FIREWALL_USAGE_RATIO + ROUTER_EMBODIED_IMPACT_WCF * ROUTER_USAGE_RATIO + SWITCH_EMBODIED_IMPACT_WCF * SWITCH_USAGE_RATIO

FIREWALL_EMBODIED_IMPACT_ADPE = 0  # fake value
ROUTER_EMBODIED_IMPACT_ADPE = 0  # fake value
SWITCH_EMBODIED_IMPACT_ADPE = 0  # fake value

NETWORK_EMBODIED_IMPACT_ADPE = FIREWALL_EMBODIED_IMPACT_ADPE * FIREWALL_USAGE_RATIO + ROUTER_EMBODIED_IMPACT_ADPE * ROUTER_USAGE_RATIO + SWITCH_EMBODIED_IMPACT_ADPE * SWITCH_USAGE_RATIO

FIREWALL_EMBODIED_IMPACT_PE = 0  # fake value
ROUTER_EMBODIED_IMPACT_PE = 0  # fake value
SWITCH_EMBODIED_IMPACT_PE = 0  # fake value

NETWORK_EMBODIED_IMPACT_PE = FIREWALL_EMBODIED_IMPACT_PE * FIREWALL_USAGE_RATIO + ROUTER_EMBODIED_IMPACT_PE * ROUTER_USAGE_RATIO + SWITCH_EMBODIED_IMPACT_PE * SWITCH_USAGE_RATIO

NETWORK_LIFESPAN = 5 * 365 * 24 * 60 * 60  # seconds, ADEME

SERVER_GPU_NETWORK_POWER = SERVER_GPU_POWER + NETWORK_POWER  # kW
###############################


###############################
# defaults for training phase
FLOPS_PER_WATT = 1.4e12  #  FLOP/s / Watt  #  take NVIDIA H100 as reference (https://epoch.ai/data-insights/ml-hardware-energy-efficiency)
FLOPS_PER_GPU = 1.979e15  #  FLOP/s / GPU  # take NVIDIA H100 SXM BFLOAT16 as reference (https://resources.nvidia.com/en-us-hopper-architecture/nvidia-tensor-core-gpu-datasheet?ncid=no-ncid)
ENERGY_PER_FLOPS = 1 / FLOPS_PER_WATT  # Watt / (FLOP/s)

GPU_UTILIZATION_RATE = 0.7  # the share of time that the GPU is actively performing
# TODO: check the GPU utilization rate. In SNCF excel, 0.85 * 0.85 is used.
INFERENCE_COMPUTE_SHARE = 0.8 # the share of inference compute capacity in the total company's yearly compute capacity, used to estimate the inference compute capacity from the total compute capacity
MODEL_LIFESPAN = 2 * 365 * 24 * 60 * 60  # in seconds, the time span during which the model is actively used after its publication
##############################


###############################
# defaults for training data storage
STORAGE_DURATION = 100 * 24 # in hours, 100 days, Epoch AI
HDD_VOLUME = 30 # TB
HDD_POWER = 0.0095 # kW
HDD_USAGE_RATIO = 0.2  # Epoch AI

HDD_EMBODIED_IMPACT_GWP = 640.5 # kgCO2eq, ResilioDB
HDD_EMBODIED_IMPACT_WCF = 163480 # L eq, ResilioDB
HDD_EMBODIED_IMPACT_ADPE = 0 # fake value
HDD_EMBODIED_IMPACT_PE = 0 # fake value

HDD_LIFESPAN = 5 * 365 * 24 * 60 * 60 # seconds

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
    "GPU_EMBODIED_IMPACT_WCF",
    "SERVER_GPUS",
    "SERVER_POWER",
    "SERVER_EMBODIED_IMPACT_GWP",
    "SERVER_EMBODIED_IMPACT_ADPE",
    "SERVER_EMBODIED_IMPACT_PE",
    "SERVER_EMBODIED_IMPACT_WCF",
    "HARDWARE_LIFESPAN",
    "BATCH_SIZE",
    "SERVER_GPU_POWER",
    ############################
    "NETWORK_POWER",
    "NETWORK_EMBODIED_IMPACT_GWP",
    "NETWORK_EMBODIED_IMPACT_ADPE",
    "NETWORK_EMBODIED_IMPACT_PE",
    "NETWORK_EMBODIED_IMPACT_WCF",
    "NETWORK_LIFESPAN",
    "SERVER_GPU_NETWORK_POWER",
    ############################
    # above is for inference impacts modelling, below is for training impacts modelling
    "FLOPS_PER_WATT",
    "FLOPS_PER_GPU",
    "ENERGY_PER_FLOPS",
    "GPU_UTILIZATION_RATE",
    "INFERENCE_COMPUTE_SHARE",
    "MODEL_LIFESPAN",
    ############################
    "STORAGE_DURATION",
    "HDD_VOLUME",
    "HDD_POWER",
    "HDD_USAGE_RATIO",
    "HDD_EMBODIED_IMPACT_GWP",
    "HDD_EMBODIED_IMPACT_WCF",
    "HDD_EMBODIED_IMPACT_ADPE",
    "HDD_EMBODIED_IMPACT_PE",
    "HDD_LIFESPAN",
]