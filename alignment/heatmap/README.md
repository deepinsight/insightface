We provide our implementation of ``Stacked Dense U-Nets with Dual Transformers for Robust Face Alignment`` here at [BMVC](http://bmvc2018.org/contents/papers/0051.pdf) or link at [Arxiv](https://arxiv.org/abs/1812.01936).

We also provide some popular heatmap based approaches like stacked hourglass, etc..  You can define different loss-type/network structure/dataset in ``config.py``(from ``sample_config.py``).

For example, by default, you can train our approach by ``train.py --network sdu`` or train hourglass network by ``train.py --network hourglass``.

2D training/validation dataset is now available at [baiducloud](https://pan.baidu.com/s/1kdquiIGTlK7l26SPWO_cmw) or [dropbox](https://www.dropbox.com/s/por6mbguegmywo6/bmvc_sdu_data2d.zip?dl=0)

3D training/validation dataset is now available at [baiducloud](https://pan.baidu.com/s/1VjFWm6eEtIqGKk92GE2rgw) or [dropbox](https://www.dropbox.com/s/tjze176lh76nciw/bmvc_sdu_data3d.zip?dl=0)

