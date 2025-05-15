# EMMOE: A Comprehensive Benchmark for Embodied Mobile Manipulation in Open Environments

Autonomous household robots driven by user instructions.

[![Project](https://img.shields.io/badge/Project-blue)](https://silence143.github.io/EMMOE)
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b)](https://arxiv.org/abs/2503.08604)
[![PDF](https://img.shields.io/badge/Paper-lightgrey)](assets/EMMOE.pdf)
[![Model](https://img.shields.io/badge/Model-yellow?logo=huggingface)](https://huggingface.co/collections/Dongping-Li/emmoe-dataset-and-model-67c6b04da2b83b08ec273ef2)
[![Dataset](https://img.shields.io/badge/Dataset-yellow?logo=huggingface)](https://huggingface.co/datasets/Dongping-Li/EMMOE-100)

<!-- [![Demo](https://img.youtube.com/vi/wYnjsRY2SXs/0.jpg)](https://www.youtube.com/watch?v=wYnjsRY2SXs) -->


[![Demo](https://github.com/silence143/EMMOE/blob/main/assets/page.png)](https://github.com/silence143/EMMOE/blob/main/assets/paper_demonstration.mp4)


## Quick Start

### Installation

Our codes are based on [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA/) and [Hab-M3](https://github.com/Jiayuan-Gu/hab-mobile-manipulation), and include two separate environments. We have already modified the original source codes, so **the source codes are incompatible with our project**.


Environment 1: 
```
git clone https://github.com/silence143/EMMOE.git
cd EMMOE
conda create -n homiebot python=3.10 -y
conda activate homiebot
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
pip install decord opencv-python git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d
```

Environment 2: 
You can follow the installation steps in [Hab-M3](https://github.com/Jiayuan-Gu/hab-mobile-manipulation), and replace ``git submodule update --init --recursive`` with following commands:
```
cd /hab-mobile-manipulation
git clone https://github.com/facebookresearch/habitat-lab.git && cd habitat-lab
git checkout 2ec4f6832422faebf20ca413b1ebf78547a4855d && cd ..
git clone https://github.com/facebookresearch/habitat-sim.git && cd habitat-sim
git checkout ccbfa32e0af6d2bfb5a3131f28f69b72f184e638 && cd ..
```
You can also refer to their repositories if you meet installation problems.


### Interactive Play

You can download the original [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA/) models and use `scripts/train/finetune_lora.sh` and `scripts/train/dpo.py` to train, or directly download our models [here](https://huggingface.co/collections/Dongping-Li/emmoe-dataset-and-model-67c6b04da2b83b08ec273ef2). Then save models into `checkpoints/` folder.

To run a EMMOE task, you need to prepare two terminals: one for the high-level planner and one for the low-level executor. For **Terminal 1**, run the following command first:
```
conda activate homiebot
python infer.py
```

When prompted with `Input your task:`, you can input a household task (e.g., "find a short can and put it into the kitchen drawer"). Once you see `Server is listening...`, open **Terminal 2**, and run:
```
conda activate hab-mm
cd /hab-mobile-manipulation
python /mobile_manipulation/infer.py
```
Then the system will then automatically execute your task until it's completed, all results will be saved in the `infer/` directory.

## Evaluations

### Experiments

To reproduce the results in our [paper](https://arxiv.org/abs/2503.08604), you need to first download EMMOE-100 dataset [here](https://huggingface.co/datasets/Dongping-Li/EMMOE-100), and still prepare two terminals.

In **Terminal 1**, run: 
```
conda activate homiebot
python /scripts/eval/eval_HomieBot.py
```
You can also replace the script with other scripts in the same folder to eval other models.


when you see `Server is listening...`, open **Terminal 2**, and run:
```
conda activate hab-mm
cd /hab-mobile-manipulation
python /mobile_manipulation/eval_M3.py
```
The results will be save in the corresponding sub-folder under `/exp` folder. 

Note: 
- We use a single NVIDIA-40 for evaluations, and it would take totally 70 hours around for each model's evaluation. If you want to use multi-GPUs or eval multiple models simultaneously to speed up, remember to edit communication port in both scripts, as well the temporary filename `input_path` at `eval_M3.py` (line 609). Also remember to keep the consistency of `exp_name` and index range.

- You might meet memory leak during experiments, which is an innate problem in [Hab-M3](https://github.com/Jiayuan-Gu/hab-mobile-manipulation). You can refer to their guidances, but can't guarantee to totally solve this problem. We recommend you can manually restart the experiments from the breakpoints, and edit the index range in both evaluation files.

### Metrics

After getting the experimental results, you can use the scripts in `scripts/metrics`, `count_TP.ipynb` will together calculate SR and TP, `srr.ipynb` for SRR, `ser.ipynb` for SER, `plw_sr.ipynb` for PLWSR, `analysis.ipynb` for low-level action's SR and detailed error statistics.


## Citation

```BibTeX
@article{li2025emmoe,
  title={EMMOE: A Comprehensive Benchmark for Embodied Mobile Manipulation in Open Environments},
  author={Li, Dongping and Cai, Tielong and Tang, Tianci and Chai, Wenhao and Driggs-Campbell, Katherine Rose and Wang, Gaoang},
  journal={arXiv preprint arXiv:2503.08604},
  year={2025}
}
```

## License

This project is released under the [Apache License 2.0](https://github.com/silence143/EMMOE/blob/main/LICENSE). Please also adhere to the Licenses of models and datasets being used.