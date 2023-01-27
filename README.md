# Learnable Bernoulli Dropout

[Learnable Bernoulli Dropout for Bayesian Deep Learning](https://proceedings.mlr.press/v108/boluki20a)

Shahin Boluki, Randy Ardywibowo, Siamak Zamani Dadaneh, Mingyuan Zhou, Xiaoning Qian

## Overview

In this work, we propose learnable Bernoulli dropout (LBD), a new model-agnostic dropout scheme that considers the dropout rates as parameters jointly optimized with other model parameters. By probabilistic modeling of Bernoulli dropout, our method enables more robust prediction and uncertainty quantification in deep models. Especially, when combined with variational auto-encoders (VAEs), LBD enables flexible semi-implicit posterior representations, leading to new semi-implicit VAE (SIVAE) models. We solve the optimization for training with respect to the dropout parameters using Augment-REINFORCE-Merge (ARM), an unbiased and low-variance gradient estimator. Our experiments on a range of tasks show the superior performance of our approach compared with other commonly used dropout schemes. Overall, LBD leads to improved accuracy and uncertainty estimates in image classification and semantic segmentation. Moreover, using SIVAE, we can achieve state-of-the-art performance on collaborative filtering for implicit feedback on several public datasets.

## Citation

Please consider citing our paper if you find the software useful for your work.

```

@InProceedings{pmlr-v108-boluki20a,
  title = 	 {Learnable Bernoulli Dropout for Bayesian Deep Learning},
  author =       {Boluki, Shahin and Ardywibowo, Randy and Dadaneh, Siamak Zamani and Zhou, Mingyuan and Qian, Xiaoning},
  booktitle = 	 {Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics},
  pages = 	 {3905--3916},
  year = 	 {2020},
  editor = 	 {Chiappa, Silvia and Calandra, Roberto},
  volume = 	 {108},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {26--28 Aug},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v108/boluki20a/boluki20a.pdf},
  url = 	 {https://proceedings.mlr.press/v108/boluki20a.html},
```
