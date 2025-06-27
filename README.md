### <div align="center"> T2I-ConBench: Text-to-Image Benchmark for Continual Post-training </div>

<div align="center">
  <a href="https://k1nght.github.io/T2I-ConBench/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/abs/2505.16875"><img src="https://img.shields.io/static/v1?label=Paper&message=arXiv&color=red&logo=arxiv"></a> &ensp;
  <a href="https://huggingface.co/datasets/T2I-ConBench/T2I-ConBench"><img src="https://img.shields.io/static/v1?label=Dataset&message=HuggingFace&color=yellow"></a> &ensp;

</div>

---
This repository contains code for the paper of <strong>T2I-ConBench</strong>, a comprehensive benchmark for continual post-training of T2I diffusion models. 
> [**T2I-ConBench: Text-to-Image Benchmark for Continual Post-training**](https://arxiv.org/pdf/2505.16875)<br>
> [Zhehao Huang](https://k1nght.github.io/huangzhehao.github.io/)\*, [Yuhang Liu]()\*, [Yixin Lou]()\*, [Zhengbao He](), [Mingzhen He](), [Wenxing Zhou](), [Tao Li](https://nblt.github.io/), [Kehan Li](), [Zeyi Huang](), [Xiaolin Huang]()<br>
> Shanghai Jiao Tong University, Huawei

# Abstract 
Continual post‑training adapts a single text‑to‑image diffusion model to learn new tasks without incurring the cost of separate models, but naïve post-training causes forgetting of pretrained knowledge and undermines zero‑shot compositionality. We observe that the absence of a standardized evaluation protocol hampers related research for continual post‑training. To address this, we introduce <strong>T2I‑ConBench</strong>, a unified benchmark for continual post-training of text-to-image models. T2I-ConBench focuses on two practical scenarios, <em>item customization</em> and <em>domain enhancement</em>, and analyzes four dimensions: (1) retention of generality, (2) target-task performance, (3) catastrophic forgetting, and (4) cross-task generalization. It combines automated metrics, human‑preference modeling, and vision‑language QA for comprehensive assessment. We benchmark ten representative methods across three realistic task sequences and find that no approach excels on all fronts. Even joint "oracle" training does not succeed for every task, and cross-task generalization remains unsolved. We release all datasets, code, and evaluation tools to accelerate research in continual post‑training for text‑to‑image models.

<table align="center">
  <tr>
    <td align="center"> 
      <img src="assets/overview.png" alt="Teaser" style="width: 1000px;"/> 
      <br>
      <em style="font-size: 18px;"><strong style="font-size: 18px;"><strong>Overview of T2I-ConBench.</strong> Our benchmark consists of four components: (1) challenging continual post‑training task sequences, (2) the curation of diverse item and domain datasets, (3) an automated evaluation pipeline, and (4) comprehensive metrics to fully assess each continual learning method's ability to update knowledge, resist forgetting, and generalize across tasks.</em>
    </td>
  </tr>
</table>

# 🔥 Getting Started 

* [Continual Post-training and Inference](train) 

* [Evaluation](eval) 

# 📚 Citation
```
@misc{huang2025t2iconbenchtexttoimagebenchmarkcontinual,
      title={T2I-ConBench: Text-to-Image Benchmark for Continual Post-training}, 
      author={Zhehao Huang and Yuhang Liu and Yixin Lou and Zhengbao He and Mingzhen He and Wenxing Zhou and Tao Li and Kehan Li and Zeyi Huang and Xiaolin Huang},
      year={2025},
      eprint={2505.16875},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.16875}, 
}
```