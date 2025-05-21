"""Utility functions for supporting binary quantization

From https://docs.vespa.ai/en/binarizing-vectors.html#appendix-conversion-to-int8
"""

import numpy as np


def binarize_tensor(tensor: list[float]) -> list[int]:
    """
    Binarize a floating-point list by thresholding at zero
    and packing the bits into bytes.
    """
    tensor = np.array(tensor)
    return np.packbits(np.where(tensor > 0, 1, 0), axis=0).astype(np.int8).tolist()
