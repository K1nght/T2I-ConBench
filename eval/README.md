# Evaluation Pipeline of T2I-ConBench

<table align="center">
  <tr>
    <td align="center"> 
      <img src="assets/cross_eval.png" alt="Teaser" style="width: 1000px;"/> 
      <br>
      <em style="font-size: 18px;"><strong style="font-size: 18px;">Evaluation pipeline of cross-task generalization.</em>
    </td>
  </tr>
</table>

## 🔧 Dependencies and Installation
```bash
conda create -n t2i-conbench-eval python=3.9
conda activate t2i-conbench-eval

# FID 
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/boomb0om/text2image-benchmark

# HPS
pip install hpsv2

# 


pip install -r requirements.txt
```

## 🔥 Evaluation

1. FID on MS-COCO dataset

```bash
bash run/fid.sh
```

2. Unique-Sim and Class-Forget for Sequential Item Customization

```bash
bash run/item.sh
```

3. HPS for Sequential Domain Enhancement

```bash
bash run/hps.sh
```

4. Cross-task Generalization for Sequential Item-Domain Adaptation

```bash
bash run/cross.sh
```

## Acknowledgements
We would like to thank the following repositories:
- [T2I-Benchmark](https://github.com/boomb0om/text2image-benchmark)
- [HPS](https://github.com/lucidrains/hpsv2)

