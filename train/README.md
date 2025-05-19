# Training Pipeline of T2I-ConBench

<table align="center">
  <tr>
    <td align="center"> 
      <img src="assets/cl_method.png" alt="Teaser" style="width: 1000px;"/> 
      <br>
      <em style="font-size: 18px;"><strong style="font-size: 18px;">Overview of the continual post‑training baselines evaluated in this work, encompassing rehearsal‑based, regularization‑based, and parameter‑isolation methods (sparse fine‑tuning and low‑rank adaptation).</em>
    </td>
  </tr>
</table>

This document outlines the step-by-step process for training and generating images in T2I-ConBench.

## 🔧 Dependencies and Installation
```bash
conda create -n t2i-conbench python=3.9
conda activate t2i-conbench
pip install -r requirements.txt
```

## 📂 Dataset Preparation

1. Process DreamBooth Dataset
   ```bash
   bash run/process_dreambooth.sh
   ```
   Prepares the item dataset for training.

2. Generate Item Prior
   ```bash
   bash run/generate_prior.sh
   ```
   Creates the prior class images for item customization tasks.

3. Process Domain Dataset
   ```bash
   bash run/process_domain.sh
   ```
   Prepares the domain dataset for training.

4. Extract T5 Features (optional)
   ```bash
   bash run/extract_t5_feature.sh
   ```
   Extracts T5 text features of domain dataset.

## 🔥 Training and Generation

### Sequential Item Customization Training
1. Sequential Training
   ```bash
   bash run/run_item.sh
   ```

2. Generate Inference Images
   - Generate images using the trained item-based model

### Sequential Domain Enhancement Training
1. Sequential Training
   ```bash
   bash run/run_domain.sh
   ```

2. Generate Inference Images
   - Generate images using the trained domain-based model

### Sequential Item-Domain Adaptation Training
1. Sequential Training
   ```bash
   bash run/run_item-domain.sh
   ```

2. Generate Inference Images
   - Generate images using the combined model

## Acknowledgements
We would like to thank the following repositories:
- [Diffusers Dreambooth Example](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth)
- [PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha)
