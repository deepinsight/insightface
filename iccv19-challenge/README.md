[The Lightweight Face Recognition Challenge & Workshop](https://ibug.doc.ic.ac.uk/resources/lightweight-face-recognition-challenge-workshop/) will be held in conjunction with the International Conference on Computer Vision (ICCV) 2019, Seoul Korea. 

Please read carefully and strictly follow the rules.

[Test Submission Server](http://www.insightface-challenge.com/overview) 

**NEWS**

``2019.05.16`` The four sections(glint-large, glint-light, iqiyi-large, iqiyi-light) will share the price pool for 1/4 each respectively. From each section, the top 3 players share the section price pool for 50%, 30% and 20% respectively.

``2019.05.11`` We updated the groundtruth of iQIYI video testset to v0.2. Please re-summit the feature set for iQIYI sections.



==================

**How To Start:**

**Training:**

1. Download ms1m-retinaface from [baiducloud](https://pan.baidu.com/s/1rQxJ3drqm_071vpxBtp98A) or [dropbox](https://www.dropbox.com/s/ev5ezzcz79p2hge/ms1m-retinaface-t1.zip?dl=0) and unzip it to `$INSIGHTFACE_ROOT/datasets/`
2. Go into `$INSIGHTFACE_ROOT/recognition/`
3. Refer to the `retina` dataset config section in `sample_config.py` and copy it to your own`config.py`.
4. Start training with `CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --dataset retina --network [your-network] --loss arcface`. It will output the accuracy of lfw, cfp_fp and agedb_30 every 2000 batches by default.
5. It is better to put the training dataset on SSD hard disk, to obtain good training performance.

------------------

**Testing:**

1. testdata-image from [baiducloud](https://pan.baidu.com/s/1UKUYsRfVTSzj1tfU3BVFrw) or [dropbox](https://www.dropbox.com/s/r5y6xt754m36rh8/iccv19-challenge-data-v1.zip?dl=0). These face images are all pre-processed and aligned so no need to do further modification.
2. To download testdata-video from iQIYI, please visit <http://challenge.ai.iqiyi.com/data-cluster>. You must download iQIYI-VID-FACE.z01, iQIYI-VID-FACE.z02 and iQIYI-VID-FACE.zip after signin. These face images are all pre-processed and aligned so no need to do further modification.
   1. To unzip: ``zip iQIYI_VID_FACE.zip -s=0 --out iQIYI_VID_FACE_ALL.zip; unzip iQIYI_VID_FACE_ALL.zip``
   2. We can get a directory named ``iQIYI_VID_FACE`` after decompression. Then we have to move ``video_filelist.txt`` in testdata-image package to ``iQIYI_VID_FACE/filelist.txt``, to indicate the order of videos in our submission feature file.
3. To generate image feature submission file: check ``gen_image_feature.py``
4. To generate video feature submission file: check ``gen_video_feature.py``
5. Submit binary feature to the right section on test server.

You can also check the verification performance during training time on LFW,CFP_FP,AgeDB_30 datasets.

------------------

**Evaluation:**

Final ranking is determined by the TAR under 1:1 protocal only, for all valid submissions. 

For image testset, we evaluate the TAR under FAR@e-8 while we choose the TAR under FAR@e-4 for video testset.


------------------

**Discussion:**

[https://github.com/deepinsight/insightface/issues/632](https://github.com/deepinsight/insightface/issues/632)

------------------

**Baseline:**

1. Network y2(a deeper mobilefacenet): 933M FLOPs. TAR_image: 0.64691, TAR_video: 0.47191
2. Network r100fc(ResNet100FC-IR): 24G FLOPs. TAR_image: 0.80312, TAR_video: 0.64894

------------------

**Candidate solutions:**

1. Use slightly deeper or wider mobile-level networks.
2. Try different training methods/losses than straightforward arcface.
3. [OctConv](https://arxiv.org/abs/1904.05049), to reduce FLOPs.
4. [HRNet](https://arxiv.org/abs/1904.04514), for large FLOPs track.
and so on
