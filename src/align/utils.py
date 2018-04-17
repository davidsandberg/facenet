# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

def crop(img, bb, margin):
    """
    img = image from misc.imread, which should be in (H, W, C) format
    bb = pixel coordinates of bounding box: (x0, y0, x1, y1)
    margin = float from 0 to 1 for the amount of margin to add, relative to the
        bounding box dimensions (half margin added to each side)
    """
    
    if margin < 0:
        raise ValueError("the margin must be a value between 0 and 1")
    if margin > 1:
        raise ValueError("the margin must be a value between 0 and 1 - this is a change from the existing API")
    
    img_height = img.shape[0]
    img_width = img.shape[1]
    x0, y0, x1, y1 = bb[:4]
    margin_height = (y1 - y0) * margin / 2
    margin_width = (x1 - x0) * margin / 2
    x0 = int(np.maximum(x0 - margin_width, 0))
    y0 = int(np.maximum(y0 - margin_height, 0))
    x1 = int(np.minimum(x1 + margin_width, img_width))
    y1 = int(np.minimum(y1 + margin_height, img_height))
    return  img[y0:y1,x0:x1,:], (x0, y0, x1, y1)