"""
evaluating the model on test data
1. load test data
2. apply preprocessing steps to the characters
3. segment characters
4. classify characters one by one
5. concatenate characters to form the word (as a list)
6. compare the predicted word with the ground truth word (as a list)
7. compare the lists to calculate the accuracy
"""

from preprocessing import load_img, reduce_noise, sharpen, increase_contrast, adjust_skew, extract_baseline, compress_thresholding, segment_characters
import torch.nn as nn

def evaluate_model(model: nn.Module, test_data: list[str]) -> float:
    """
    evaluate the model on test data
    """
    return 0.0