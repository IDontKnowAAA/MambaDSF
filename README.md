# MambaDSF

**Hybrid Mamba-Transformer with Multi-Scale Dilated Attention for Small Target Detection in Sonar Images**

<p align="center">
  <img src="assets/MambaDSF_Architecture.png" width="100%" alt="MambaDSF Architecture">
</p>

## Overview

MambaDSF is a hybrid framework that leverages selective state space models (SSMs) for linear-complexity global context modeling in sonar imagery. The framework addresses three key challenges in small target detection:

- **Scarce discriminative features**: Small targets span limited pixels and may reduce to one or two feature-map cells after downsampling.
- **Low signal-to-noise ratio**: Target responses are comparable in intensity to reverberation and speckle noise.
- **Scale ambiguity**: Apparent target size varies with imaging range and resolution.

## Architecture

MambaDSF consists of three main components:

1. **MambaFPN Backbone**: Couples MambaVision with a bidirectional feature pyramid (FPN + PANet) for multi-scale extraction and long-range dependency modeling.
2. **DFMamba Encoder**: Performs intra-scale dilated attention at four receptive-field scales and cross-scale SSM fusion for semantic alignment.
3. **RT-DETR Decoder**: Generates predictions via deformable cross-attention over 300 learnable queries.

### Qualitative Comparison

<p align="center">
  <img src="assets/Compare120dpi.png" width="100%" alt="Detection Comparison">
</p>

*Qualitative comparison on representative UATD test samples across eight detection methods.*

## Code Release Status

> **Note**: The codebase is currently undergoing preparation for public release.

We appreciate your patience and interest in our work. The complete implementation will be made available soon. In the meantime, feel free to reach out to the authors for any technical inquiries.

## Acknowledgements

This work was supported in part by the National Natural Science Foundation of China under Grant 62001443, and in part by the Natural Science Foundation of Shandong Province under Grant ZR2020QE294.

## Contact

- Hui Lin: harrylin929@gmail.com
- Jiayi Li: leanolee58@gmail.com
- Shenghui Rong (Corresponding): rsh@ouc.edu.cn
