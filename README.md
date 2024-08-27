# [ICML-24] Counterfactual Reasoning for Multi-Label Image Classification via Patching-Based Training

The implementation for the paper [Counterfactual Reasoning for Multi-Label Image Classification via Patching-Based Training](https://arxiv.org/pdf/2404.06287) (ICML 2024). 

## Preparing Data 

See the `README.md` file in the `data` directory for instructions on downloading and preparing the datasets.

## Training Model

See `run.sh` .

## Pretrained Models

| Model | Pretrained | mAP   | Link                                                         |
| :----------------- | ---------- | ----- | ------------------------------------------------------------ |
|   ResNet-101   | in1k   | 85.1 | [Log](https://drive.google.com/drive/folders/1GZQoBBgYCUTpeCP4RvDdDLydNtgJZKCm)  |
|   TResNet-L    | in1k   | 88.7 | [Log](https://drive.google.com/drive/folders/1xGmZrrRDCEm8YQEOdYm_zxjQ8Hz8B4C5)  |
|   TResNet-L    | in21k  | 90.5 | [Log](https://drive.google.com/drive/folders/1OOC-BJjQksFkMVKvVlIf771sZB7exvw0)  |
| Q2L-TResNet-L  | in21k  | 91.0 | [Log](https://drive.google.com/drive/folders/1N30PfY77XtnSsMAB6rL9xfMv3JNUEptI)  |
| Q2L-TResNet-L  | oi     | 91.7 | [Model&Log](https://drive.google.com/drive/folders/1LYhVADz1O8BOLNBkZsBJ37noEERjiBZL)  |

