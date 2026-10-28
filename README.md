# QTR-PAR: Quaternion Tensor Ring with Periodic Anchoring Regularization

This repository provides the official MATLAB implementation of the paper:

**QTR-PAR: Quaternion Tensor Ring with Periodic Anchoring Regularization for Multi-Dimensional Data Recovery**
Shan Gai, Chenglong Lv, and Chengjie Xie
*Manuscript under review at IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2026.*

## Requirements

- MATLAB R2021a or later (tested on R2023a).
- (Optional) Parallel Computing Toolbox for accelerated block matching. The code gracefully degrades to sequential execution if not installed.
- No external toolboxes are strictly required. A built-in FSIM implementation is provided in the demo script. If the official `Img_QA` toolbox is available, place it in the `lib/` folder for exact FSIM calculation.

## Quick Start

1. Clone or download this repository to your local machine.
2. Place your input video frames (`.png` or `.jpg`) into the `data/` folder. Sample frames are provided for a quick test.
3. Open MATLAB, navigate to the root directory, and run the main demo script:
   ```matlab
   Demo_Video_TC