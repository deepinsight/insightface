

The Lightweight Face Recognition Challenge & Workshop will be held in conjunction with the International Conference on Computer Vision (ICCV) 2019, Seoul Korea. 



Firstly please read the workshop [homepage](https://ibug.doc.ic.ac.uk/resources/lightweight-face-recognition-challenge-workshop/) carefully about the training and testing rules.



Test Submission Server[TODO]

==================

**How To Start:**

**Training:**

1. Download ms1m-retinaface from [baiducloud](https://pan.baidu.com/s/1rQxJ3drqm_071vpxBtp98A) or [dropbox](https://www.dropbox.com/s/ev5ezzcz79p2hge/ms1m-retinaface-t1.zip?dl=0) and unzip it to `$INSIGHTFACE_ROOT/datasets/`
2. Go into `$INSIGHTFACE_ROOT/recognition/`
3. Refer to the `retina` dataset config section in `sample_config.py` and copy it to your own`config.py`.
4. Start training with `CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train.py --dataset retina --network [your-network] --loss arcface`. It will output the accuracy of lfw, cfp_fp and agedb_30 every 2000 batches by default.

------------------

**Testing:**

1. testdata-image from [baiducloud](https://pan.baidu.com/s/1UKUYsRfVTSzj1tfU3BVFrw) or dropbox. These face images are all pre-processed and aligned so no need to do further modification.
2. To download testdata-video from iQIYI, please visit <http://challenge.ai.iqiyi.com/data-cluster>. You must download iQIYI-VID-FACE.z01, iQIYI-VID-FACE.z02 and iQIYI-VID-FACE.zip after signin. These face images are all pre-processed and aligned so no need to do further modification.
   1. To unzip: ``zip iQIYI_VID_FACE.zip -s=0 --out iQIYI_VID_FACE_ALL.zip; unzip iQIYI_VID_FACE_ALL.zip``
   2. We can get a directory named ``iQIYI_VID_FACE`` after decompression. Then we have to move ``video_filelist.txt`` in testdata-image package to ``iQIYI_VID_FACE/filelist.txt``, to indicate the order of videos in our submission feature file.
3. To generate image feature submission file: check ``gen_image_feature.py``
4. To generate video feature submission file: check ``gen_video_feature.py``
5. Submit binary feature to the right section on test server.

You can also check the verification performance during training time on LFW,CFP_FP,AgeDB_30 datasets.

------------------

**Evaluation:**

Final ranking is determined by accuracy only, for all valid submissions. For example, score of track-1 will be calculated by ``TAR_glint-light+TAR_iqiyi-light`` while ``TAR_glint-large+TAR_iqiyi-large`` for track-2.

------------------

**Discussion:**

[https://github.com/deepinsight/insightface/issues/632](https://github.com/deepinsight/insightface/issues/632)

------------------

**Baseline:**

1. Network y2(a deeper mobilefacenet): 933M FLOPs. TAR_image: [TODO], TAR_video: [TODO]
2. Network r100fc(ResNet100FC-IR): 24G FLOPs. TAR_image: [TODO], TAR_video: [TODO]

------------------

**Candidate solutions:**

1. Use slightly deeper or wider mobile-level networks.
2. Try different training methods/losses than straightforward arcface.
3. [OctConv](https://arxiv.org/abs/1904.05049), to reduce FLOPs.
4. [HRNet](https://arxiv.org/abs/1904.04514), for large FLOPs track.
