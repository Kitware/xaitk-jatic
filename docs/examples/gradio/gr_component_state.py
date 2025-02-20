"""
Module: saliency_configuration

This module provides utility functions to configure and manage parameters for
image classification and object detection saliency algorithms. The parameters
are dynamically stored in lists to enable flexible and modular setup.

Functions:
----------
1. Model Selection:
   - select_img_cls_model: Select the image classification model.
   - select_obj_det_model: Select the object detection model.

2. Saliency Algorithm Selection:
   - select_img_cls_saliency_algo: Select the saliency algorithm for image classification.
   - select_obj_det_saliency_algo: Select the saliency algorithm for object detection.

3. Saliency Algorithm Parameters:
   - select_threads: Configure the number of threads.
   - enter_window_size: Configure the window size.
   - enter_stride: Configure the stride value.
   - enter_num_masks: Configure the number of masks.
   - enter_spatial_res: Configure the spatial resolution.
   - select_p1: Set the probability threshold.
   - enter_seed: Set the random seed for reproducibility.
   - check_debiased: Set the debiased mode flag.
   - enter_occlusion_grid_size: Configure the occlusion grid size.

Usage:
------
This module is designed to support dynamic configuration of saliency-based
interpretability algorithms for image classification and object detection
tasks. It is typically used in visualization pipelines for deep learning
models to generate interpretable results.

Example:
--------
>>> from saliency_configuration import select_img_cls_model, enter_window_size
>>> model = select_img_cls_model("ResNet-50")
>>> window_size = enter_window_size("(3, 3)")
>>> print(model, window_size)
'ResNet-50' '(3, 3)'

Notes:
------
- The state of parameters is stored in global lists, making it easier to
  track and modify configurations dynamically.
- The functions support integration with saliency libraries like RISE
  or Sliding Window-based techniques.
"""

# Choice of image classification model
img_cls_model_name = ["ResNet-50"]

# Choice of object detection model
obj_det_model_name = ["Faster-RCNN"]

# Choice of image classification saliency algorithm
img_cls_saliency_algo_name = ["RISE"]

# Choice of object detection saliency algorithm
obj_det_saliency_algo_name = ["DRISE"]

# Number of threads to utilize when generating masks
threads_state = [4]

# Window_size for SlidingWindowStack algorithm
window_size_state = ["(50,50)"]

# Stride for SlidingWindowStack algorithm
stride_state = ["(20,20)"]

# Number of random masks for RISEStack/DRISEStack algorithm
num_masks_state = [200]

# Spatial resolution of masking grid for RISEStack/DRISEStack algorithm
spatial_res_state = [8]

# Probability of the grid cell being set to 1 (otherwise 0)
p1_state = [0.5]

# Random seed to allow for reproducibility
seed_state = [0]

# Debiased option for RISEStack/DRISEStack saliency algorithm
debiased_state = [True]

# Occlusion grid cell size in pixels for RandomGridStack algorithm
occlusion_grid_state = ["(128,128)"]


def select_img_cls_model(model_choice: str) -> str:
    """
    Select and store the image classification model choice.

    Args:
        model_choice (str): The name of the image classification model.

    Returns:
        str: The selected model choice.
    """
    img_cls_model_name.append(model_choice)
    return model_choice


def select_obj_det_model(model_choice: str) -> str:
    """
    Select and store the object detection model choice.

    Args:
        model_choice (str): The name of the object detection model.

    Returns:
        str: The selected model choice.
    """
    obj_det_model_name.append(model_choice)
    return model_choice


def select_img_cls_saliency_algo(sal_choice: str) -> str:
    """
    Select and store the image classification saliency algorithm.

    Args:
        sal_choice (str): The name of the saliency algorithm for image classification.

    Returns:
        str: The selected saliency algorithm.
    """
    img_cls_saliency_algo_name.append(sal_choice)
    return sal_choice


def select_obj_det_saliency_algo(sal_choice: str) -> str:
    """
    Select and store the object detection saliency algorithm.

    Args:
        sal_choice (str): The name of the saliency algorithm for object detection.

    Returns:
        str: The selected saliency algorithm.
    """
    obj_det_saliency_algo_name.append(sal_choice)
    return sal_choice


def select_threads(threads: int) -> int:
    """
    Select and store the number of threads to use.

    Args:
        threads (int): The number of threads.

    Returns:
        int: The selected number of threads.
    """
    threads_state.append(threads)
    return threads


def enter_window_size(val: str) -> str:
    """
    Enter and store the window size for saliency algorithms.

    Args:
        val (str): The window size value as a string.

    Returns:
        str: The entered window size.
    """
    window_size_state.append(val)
    return val


def enter_stride(val: str) -> str:
    """
    Enter and store the stride value for saliency algorithms.

    Args:
        val (str): The stride value as a string.

    Returns:
        str: The entered stride value.
    """
    stride_state.append(val)
    return val


def enter_num_masks(val: int) -> int:
    """
    Enter and store the number of masks for saliency algorithms.

    Args:
        val (int): The number of masks.

    Returns:
        int: The entered number of masks.
    """
    num_masks_state.append(val)
    return val


def enter_spatial_res(val: int) -> int:
    """
    Enter and store the spatial resolution for saliency algorithms.

    Args:
        val (int): The spatial resolution value.

    Returns:
        int: The entered spatial resolution.
    """
    spatial_res_state.append(val)
    return val


def select_p1(prob: float) -> float:
    """
    Select and store the probability threshold (p1) for saliency algorithms.

    Args:
        prob (float): The probability threshold.

    Returns:
        float: The selected probability threshold.
    """
    p1_state.append(prob)
    return prob


def enter_seed(seed: int) -> int:
    """
    Enter and store the random seed value.

    Args:
        seed (int): The seed value for randomness.

    Returns:
        int: The entered seed value.
    """
    seed_state.append(seed)
    return seed


def check_debiased(debiased: bool) -> bool:
    """
    Check and store the debiased state for saliency algorithms.

    Args:
        debiased (bool): Whether the debiased mode is enabled.

    Returns:
        bool: The entered debiased state.
    """
    debiased_state.append(debiased)
    return debiased


def enter_occlusion_grid_size(val: str) -> str:
    """
    Enter and store the occlusion grid size for saliency algorithms.

    Args:
        val (str): The occlusion grid size value as a string.

    Returns:
        str: The entered occlusion grid size.
    """
    occlusion_grid_state.append(val)
    return val
