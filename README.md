# <p align=center>`Skating-Mixer: Multimodal MLP for Scoring Figure Skating`</p><!-- omit in toc -->
The implementation of [AAAI2023 paper](https://arxiv.org/pdf/2203.03990.pdf).

# Introduction
Figure skating scoring is a challenging task because it requires judging players’ technical moves as well as coordination with the background music. Prior learning-based work cannot solve it well for two reasons: 1) each move in figure skating changes quickly, hence simply applying traditional frame sampling will lose a lot of valuable information, especially in a 3-5 minutes lasting video, so an extremely long-range representation learning is necessary; 2) prior methods rarely considered the critical audio-visual relationship in their models. Thus, we introduce a multimodal MLP architecture, named Skating-Mixer. It extends the MLP-Mixer-based framework into a multimodal fashion and effectively learns long-term representations through our designed memory recurrent unit (MRU). Aside from the model, we also collected a high-quality audio-visual FS1000 dataset, which contains over 1000 videos on 8 types of programs with 7 different rating metrics, overtaking other datasets in both quantity and diversity. Experiments show the proposed method outperforms SOTAs over all major metrics on the public Fis-V and our FS1000 dataset. In addition, we include an analysis applying our method to recent competitions that occurred in Beijing 2022 Winter Olympic Games, proving our method has strong robustness.

# Dataset
The proposed Timesformer, AST, C3D and VGGish feature of the proposed dataset can be found [here](https://pan.baidu.com/s/1SGbvK6vDGR7ZP0PxakUO7g?pwd=9tma). 
Also, if you need the raw videos, please require jingfeixia708@gmail.com. 

# Citation
```
@article{xia2022skating,
  title={Skating-Mixer: Multimodal MLP for Scoring Figure Skating},
  author={Xia, Jingfei and Zhuge, Mingchen and Geng, Tiantian and Fan, Shun and Wei, Yuantai and He, Zhenyu and Zheng, Feng},
  journal={arXiv preprint arXiv:2203.03990},
  year={2022}
}
```
