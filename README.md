# RoR
This repo provides the code for reproducing the experiments in Findings-EMNLP 2021 paper: [RoR: Read-over-Read for Long Document Machine Reading Comprehension](https://aclanthology.org/2021.findings-emnlp.160.pdf). This code is adapted from the repos of  [longformer](https://github.com/allenai/longformer).

<p align="center"><img src="/RoR.png" width=700></p>
<p align="center"><i>Figure : Illustrations of RoR framework</i></p>


## Environment
transformers @ git+http://github.com/ibeltagy/transformers.git@longformer_encoder_decoder#egg=transformers <br>
pytorch-lightning @ git+http://github.com/ibeltagy/pytorch-lightning.git@v0.8.5_fixes#egg=pytorch-lightning <br>
torch>=1.6.0 <br>
test-tube==0.7.5 


## TriviaQA
To reproduce the results, you don’t have to train any model and can use the pretrained [triviaqa-longformer-large](https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/triviaqa-longformer-large.tar.gz) to infer directly. The detailed inference steps are given in the inference.txt.

## QuAC
The code is realeased in [QuAC](https://github.com/pierre-zhao/RoR)

## Citation

```
@inproceedings{zhao-etal-2021-ror-read,
    title = "{R}o{R}: Read-over-Read for Long Document Machine Reading Comprehension",
    author = "Zhao, Jing  and
      Bao, Junwei  and
      Wang, Yifan  and
      Zhou, Yongwei  and
      Wu, Youzheng  and
      He, Xiaodong  and
      Zhou, Bowen",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.160",
    pages = "1862--1872",
}

```

