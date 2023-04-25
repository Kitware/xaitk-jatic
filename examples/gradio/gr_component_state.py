# Choice of model
model_name = []

# Choice of saliency algorithm
saliency_algo_name = []

# Number of threads to utilize when generating masks
threads_state = []

# Window_size for SlidingWindowStack algorithm
window_size_state = []

# Stride for SlidingWindowStack algorithm
stride_state = []

# Number of random masks for RISEStack algorithm
num_masks_state = []

# Spatial resolution of masking grid for RISEStack algorithm
spatial_res_state = []

# Probability of the grid cell being set to 1 (otherwise 0)
p1_state = []

# Random seed to allow for reproducibility
seed_state = []

# Debiased option for RISEStack saliency algorithm
debiased_state = []


def select_model(model_choice):
    model_name.append(model_choice)
    return model_choice


def select_saliency_algo(sal_choice):
    saliency_algo_name.append(sal_choice)
    return sal_choice


def select_threads(threads):
    threads_state.append(threads)
    return threads


def enter_window_size(val):
    window_size_state.append(val)
    return val


def enter_stride(val):
    stride_state.append(val)
    return val


def enter_num_masks(val):
    num_masks_state.append(val)
    return val


def enter_spatial_res(val):
    spatial_res_state.append(val)
    return val


def select_p1(prob):
    p1_state.append(prob)
    return prob


def enter_seed(seed):
    seed_state.append(seed)
    return seed


def check_debiased(debiased):
    debiased_state.append(debiased)
    return debiased
