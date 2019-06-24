[2018.12.26] Now you can take a look at new megaface testing tool at ``https://github.com/deepinsight/insightface/tree/master/Evaluation/Megaface``. It is more easy to use.

Please strictly follow these rules if you want to use our MegaFace noises list.

* Please cite our paper and git repo if you want to use this list in your paper.
* Please include the information like `We used the noises list proposed by InsightFace, at https://github.com/deepinsight/insightface` if you want to submit the result to MegaFace challenge.
* To be fair, if you want to submit MegaFace result, please ensure there's no training set overlaps with FaceScrub identities. You can do this by removing identities from your training set whose cosine similarity is larger than 0.4 with any FaceScrub identity by comparing their centre feature vectors. 
* If you find more overlaps noise, please open an issue at InsightFace.
