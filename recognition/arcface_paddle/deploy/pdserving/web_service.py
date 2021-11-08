# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from paddle_serving_server.web_service import WebService, Op

import numpy as np
import cv2
import base64


class ArcFaceOp(Op):
    def init_op(self):
        pass

    def preprocess(self, input_dicts, data_id, log_id):
        (_, input_dict), = input_dicts.items()
        data = base64.b64decode(input_dict["image"])
        data = np.frombuffer(data, np.uint8)
        # Note: class variables(self.var) can only be used in process op mode
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        img = cv2.resize(img,(112,112))
        # normalize to mean 0.5, std 0.5
        img = (img - 127.5) * 0.00784313725
        # BGR2RGB
        img = img[:, :, ::-1]
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, 0)
        img = img.astype('float32')
        return {"x":img.copy()}, False, None, ""

    def postprocess(self, input_dicts, fetch_dict, log_id):
        out = fetch_dict["save_infer_model/scale_0.tmp_1"]
        out_dict = {"out": out}

        return out_dict, None, ""

class ArcFaceService(WebService):
    def get_pipeline_response(self, read_op):
        arcface_op = ArcFaceOp(name="ArcFace", input_ops=[read_op])
        return arcface_op


arcface_service = ArcFaceService(name="ArcFace")
arcface_service.prepare_pipeline_config("config.yml")
arcface_service.run_service()