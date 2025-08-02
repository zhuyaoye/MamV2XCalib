# MamV2XCalib

## Overview
This repository provides a PyTorch implementation and checkpoint for **MamV2XCalib** (ICCV 2025). It leverages vehicle-to-infrastructure (V2X) collaboration methods to calibrate roadside cameras without targets.  
For details, see the paper: [MamV2XCalib: V2X-based Target-less Infrastructure Camera Calibration with State Space Model](https://arxiv.org/abs/2507.23595).

---

## Installation

```bash
cd MamV2XCalib

conda create -n mamcalib python=3.10
conda activate mamcalib

pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt   # Some packages may require source installation

cd VideoMamba
pip install -e causal-conv1d
pip install -e mamba
```

---

## Checkpoint

You can download pretrained checkpoints from the following Google Drive folder:  
👉 [Google Drive - MamV2XCalib Checkpoints](https://drive.google.com/drive/folders/1go9sdBDD6sQvSsapzB3Z3F7ud-PfoTHF)

*Note: to be released*

---

## Quick Start

### Dataset Download
You may refer to the following datasets for download and preparation:

- **[DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X)**: A large-scale dataset for vehicle-infrastructure cooperation perception tasks.  
- **[V2X-Seq](https://github.com/AIR-THU/DAIR-V2X-Seq)**: Sequence-level V2X dataset for spatiotemporal modeling.  
- **[TUMTraf Dataset](https://innovation-mobility.com/en/project-providentia/a9-dataset/)**: Real-world traffic scene dataset.

### Evaluation
- Modify the config and dataset path in `evaluate_calib_mam.py` according to your dataset.  
- Run the evaluation script:  
```bash
python evaluate_calib_mam.py
```

### Training
- Modify the config, hyperparameters, and dataset paths in `train_half.py` according to your dataset.  
- Run the first stage of training:  
```bash
python train_half.py
```
- After obtaining the checkpoint, add it to `train_mam.py`.  
- Run the second stage of training:  
```bash
python train_mam.py
```

---

## Citation
If you find this work useful, please cite:

@inproceedings{MamV2XCalib,  
    title={MamV2XCalib: V2X-based Target-less Infrastructure Camera Calibration with State Space Model},  
    author={Yaoye Zhu, Zhe Wang, Yan Wang},  
    year={2025},  
    eprint={2507.23595},  
    archivePrefix={arXiv},  
    primaryClass={cs.CV},  
    url={https://arxiv.org/abs/2507.23595},  
}

---

## Acknowledgement
- We appreciate help from:  Public codes such as [LCCNet](https://github.com/IIPCVLAB/LCCNet), [RAFT](https://github.com/princeton-vl/RAFT), [VideoMamba](https://github.com/OpenGVLab/VideoMamba), [tum-traffic-dataset-dev-kit](https://github.com/tum-traffic-dataset/tum-traffic-dataset-dev-kit) etc.
- This work is supported by National Science and Technology Major Project (2022ZD0115502), and Wuxi Research Institute of Applied Technologies, Tsinghua University under Grant 20242001120.