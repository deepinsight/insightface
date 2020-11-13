[The Lightweight Face Recognition Challenge & Workshop](https://ibug.doc.ic.ac.uk/resources/lightweight-face-recognition-challenge-workshop/) will be held in conjunction with the International Conference on Computer Vision (ICCV) 2019, Seoul Korea. 

Please strictly follow the rules. For example, please use the same [method](https://github.com/deepinsight/insightface/blob/master/common/flops_counter.py) for the FLOPs calculation regardless of your training framework is insightface or not.

[Test Server](http://www.insightface-challenge.com/overview) 

**Sponsors:**

The Lightweight Face Recognition Challenge has been supported by 

EPSRC project FACER2VM (EP/N007743/1)

Huawei (5000$)

DeepGlint (3000$)

iQIYI (3000$)

Kingsoft Cloud (3000$)

Pensees (3000$)

Dynamic funding pool: (17000$)

Cash sponsors and gift donations are welcome.

Contact:
insightface.challenge@gmail.com

**Discussion Group**

*For Chinese:*

![wechat](https://github.com/deepinsight/insightface/blob/master/resources/lfr19_wechat1.jpg)

*For English:*

(in #lfr2019 channel)
https://join.slack.com/t/insightface/shared_invite/enQtNjU0NDk2MjYyMTMzLTIzNDEwNmIxMjU5OGYzYzFhMjlkNjlhMTBkNWFiNjU4MTVhNTgzYjQ5ZTZiMGM3MzUyNzQ3OTBhZTg3MzM5M2I


**NEWS**

``2019.06.21`` We updated the groundtruth of Glint test dataset.

``2019.06.04`` We will clean the groundtruth on deepglint testset.

``2019.05.21`` Baseline models and training logs available.

``2019.05.16`` The four tracks (deepglint-light, deepglint-large, iQIYI-light, iQIYI-large) will equally share the dynamic funding pool (14000$). From each track, the top 3 players will share the funding pool for 50%, 30% and 20% respectively.

==================

**How To Start:**

**Training:**

1. Download ms1m-retinaface from [baiducloud](https://pan.baidu.com/s/1rQxJ3drqm_071vpxBtp98A) or [dropbox](https://www.dropbox.com/s/ev5ezzcz79p2hge/ms1m-retinaface-t1.zip?dl=0) and unzip it to `$INSIGHTFACE_ROOT/datasets/`
2. Go into `$INSIGHTFACE_ROOT/recognition/`
3. Refer to the `retina` dataset configuration section in `sample_config.py` and copy it as your own configuration file `config.py`.
4. Start training with `CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --dataset retina --network [your-network] --loss arcface`. It will output the accuracy of lfw, cfp_fp and agedb_30 every 2000 batches by default.
5. Putting the training dataset on SSD hard disk will achieve better training efficiency.

------------------

**Testing:**

1. Download testdata-image from [baiducloud](https://pan.baidu.com/s/1UKUYsRfVTSzj1tfU3BVFrw) or [dropbox](https://www.dropbox.com/s/r5y6xt754m36rh8/iccv19-challenge-data-v1.zip?dl=0). These face images are all pre-processed and aligned.
2. To download testdata-video from iQIYI, please visit <http://challenge.ai.iqiyi.com/data-cluster>. You need to download iQIYI-VID-FACE.z01, iQIYI-VID-FACE.z02 and iQIYI-VID-FACE.zip after registration. These face frames are also pre-processed and aligned.
   1. Unzip: ``zip iQIYI_VID_FACE.zip -s=0 --out iQIYI_VID_FACE_ALL.zip; unzip iQIYI_VID_FACE_ALL.zip``
   2. We can get a directory named ``iQIYI_VID_FACE`` after decompression. Then, we have to move ``video_filelist.txt`` in testdata-image package to ``iQIYI_VID_FACE/filelist.txt``, to indicate the order of videos in our submission feature file.
3. To generate image feature submission file: check ``gen_image_feature.py``
4. To generate video feature submission file: check ``gen_video_feature.py``
5. Submit binary feature to the right track of the test server.

You can also check the verification performance during training time on LFW,CFP_FP,AgeDB_30 datasets.

------------------

**Evaluation:**

Final ranking is determined by the TAR under 1:1 protocal only, for all valid submissions. 

For image testset, we evaluate the TAR under FAR@e-8 while we choose the TAR under FAR@e-4 for video testset.

------------------

**Baseline:**

1. Network y2(a deeper mobilefacenet): 933M FLOPs. TAR_image: 0.64691, TAR_video: 0.47191
2. Network r100fc(ResNet100FC-IR): 24G FLOPs. TAR_image: 0.80312, TAR_video: 0.64894

Baseline models download link: [baidu cloud](https://pan.baidu.com/s/1Em0ZFnefSoTsZoTd-9m8Nw)    [dropbox](https://www.dropbox.com/s/yqaziktiv38ehrv/iccv19-baseline-models.zip?dl=0)

Training logs: [baidu cloud](https://pan.baidu.com/s/12rsp-oMzsjTeU6nugEvA9g)   [dropbox](https://www.dropbox.com/s/4ufb9g7n76rfav5/iccv-baseline-log.zip?dl=0)

------------------

**Discussion:**

[https://github.com/deepinsight/insightface/issues/632](https://github.com/deepinsight/insightface/issues/632)

------------------

**Candidate solutions:**

1. Manually design or automatically search different networks/losses.
2. Use slightly deeper or wider mobile-level networks.
3. [OctConv](https://arxiv.org/abs/1904.05049), to reduce FLOPs.
4. [HRNet](https://arxiv.org/abs/1904.04514), for large FLOPs track.
and so on
