# Choice of image classification model
img_cls_model_name = []

# Choice of object detection model
obj_det_model_name = []

# Choice of image classification saliency algorithm
img_cls_saliency_algo_name = []

# Choice of object detection saliency algorithm
obj_det_saliency_algo_name = []

# Number of threads to utilize when generating masks
threads_state = [4]

# Window_size for SlidingWindowStack algorithm
window_size_state = ['(50,50)']

# Stride for SlidingWindowStack algorithm
stride_state = ['(20,20)']

# Number of random masks for RISEStack/DRISEStack algorithm
num_masks_state = ['200']

# Spatial resolution of masking grid for RISEStack/DRISEStack algorithm
spatial_res_state = ['8']

# Probability of the grid cell being set to 1 (otherwise 0)
p1_state = [0.5]

# Random seed to allow for reproducibility
seed_state = [0]

# Debiased option for RISEStack/DRISEStack saliency algorithm
debiased_state = [False]

# Occlusion grid cell size in pixels for RandomGridStack algorithm
occlusion_grid_state = ['(128,128)']


def select_img_cls_model(model_choice):
    img_cls_model_name.append(model_choice)
    return model_choice

def select_obj_det_model(model_choice):
    obj_det_model_name.append(model_choice)
    return model_choice


def select_img_cls_saliency_algo(sal_choice):
    img_cls_saliency_algo_name.append(sal_choice)
    return sal_choice


def select_obj_det_saliency_algo(sal_choice):
    obj_det_saliency_algo_name.append(sal_choice)
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


def enter_occlusion_grid_size(val):
    occlusion_grid_state.append(val)
    return val