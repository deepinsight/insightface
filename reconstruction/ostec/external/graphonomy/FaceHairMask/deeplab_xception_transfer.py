import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from collections import OrderedDict
from . import deeplab_xception, gcn
from torch.nn.parameter import Parameter


#######################
# base model
#######################


class deeplab_xception_transfer_basemodel(deeplab_xception.DeepLabv3_plus):
    def __init__(
        self,
        nInputChannels=3,
        n_classes=7,
        os=16,
        input_channels=256,
        hidden_layers=128,
        out_channels=256,
    ):
        super(deeplab_xception_transfer_basemodel, self).__init__(
            nInputChannels=nInputChannels,
            n_classes=n_classes,
            os=os,
        )
        ### source graph
        # self.source_featuremap_2_graph = gcn.Featuremaps_to_Graph(input_channels=input_channels, hidden_layers=hidden_layers,
        #                                                    nodes=n_classes)
        # self.source_graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        # self.source_graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        # self.source_graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        #
        # self.source_graph_2_fea = gcn.Graph_to_Featuremaps(input_channels=input_channels, output_channels=out_channels,
        #                                             hidden_layers=hidden_layers, nodes=n_classes
        #                                             )
        # self.source_skip_conv = nn.Sequential(*[nn.Conv2d(input_channels, input_channels, kernel_size=1),
        #                                  nn.ReLU(True)])

        ### target graph
        self.target_featuremap_2_graph = gcn.Featuremaps_to_Graph(
            input_channels=input_channels, hidden_layers=hidden_layers, nodes=n_classes
        )
        self.target_graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.target_graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.target_graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)

        self.target_graph_2_fea = gcn.Graph_to_Featuremaps(
            input_channels=input_channels,
            output_channels=out_channels,
            hidden_layers=hidden_layers,
            nodes=n_classes,
        )
        self.target_skip_conv = nn.Sequential(
            *[nn.Conv2d(input_channels, input_channels, kernel_size=1), nn.ReLU(True)]
        )

    def load_source_model(self, state_dict):
        own_state = self.state_dict()
        # for name inshop_cos own_state:
        #    print name
        new_state_dict = OrderedDict()
        for name, param in state_dict.items():
            name = name.replace("module.", "")
            if (
                "graph" in name
                and "source" not in name
                and "target" not in name
                and "fc_graph" not in name
                and "transpose_graph" not in name
            ):
                if "featuremap_2_graph" in name:
                    name = name.replace(
                        "featuremap_2_graph", "source_featuremap_2_graph"
                    )
                else:
                    name = name.replace("graph", "source_graph")
            new_state_dict[name] = 0
            if name not in own_state:
                if "num_batch" in name:
                    continue
                print('unexpected key "{}" in state_dict'.format(name))
                continue
                # if isinstance(param, own_state):
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print(
                    "While copying the parameter named {}, whose dimensions in the model are"
                    " {} and whose dimensions in the checkpoint are {}, ...".format(
                        name, own_state[name].size(), param.size()
                    )
                )
                continue  # i add inshop_cos 2018/02/01
            own_state[name].copy_(param)
            # print 'copying %s' %name

        missing = set(own_state.keys()) - set(new_state_dict.keys())
        if len(missing) > 0:
            print('missing keys in state_dict: "{}"'.format(missing))

    def get_target_parameter(self):
        l = []
        other = []
        for name, k in self.named_parameters():
            if "target" in name or "semantic" in name:
                l.append(k)
            else:
                other.append(k)
        return l, other

    def get_semantic_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if "semantic" in name:
                l.append(k)
        return l

    def get_source_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if "source" in name:
                l.append(k)
        return l

    def forward(self, input, adj1_target=None, adj2_source=None, adj3_transfer=None):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode="bilinear", align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(
            x, size=low_level_features.size()[2:], mode="bilinear", align_corners=True
        )

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)

        ### add graph

        # target graph
        # print('x size',x.size(),adj1.size())
        graph = self.target_featuremap_2_graph(x)

        # graph combine
        # print(graph.size(),source_2_target_graph.size())
        # graph = self.fc_graph.forward(graph,relu=True)
        # print(graph.size())

        graph = self.target_graph_conv1.forward(graph, adj=adj1_target, relu=True)
        graph = self.target_graph_conv2.forward(graph, adj=adj1_target, relu=True)
        graph = self.target_graph_conv3.forward(graph, adj=adj1_target, relu=True)
        # print(graph.size(),x.size())
        # graph = self.gcn_encode.forward(graph,relu=True)
        # graph = self.graph_conv2.forward(graph,adj=adj2,relu=True)
        # graph = self.gcn_decode.forward(graph,relu=True)
        graph = self.target_graph_2_fea.forward(graph, x)
        x = self.target_skip_conv(x)
        x = x + graph

        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode="bilinear", align_corners=True)

        return x


class deeplab_xception_transfer_basemodel_savememory(deeplab_xception.DeepLabv3_plus):
    def __init__(
        self,
        nInputChannels=3,
        n_classes=7,
        os=16,
        input_channels=256,
        hidden_layers=128,
        out_channels=256,
    ):
        super(deeplab_xception_transfer_basemodel_savememory, self).__init__(
            nInputChannels=nInputChannels,
            n_classes=n_classes,
            os=os,
        )
        ### source graph

        ### target graph
        self.target_featuremap_2_graph = gcn.Featuremaps_to_Graph(
            input_channels=input_channels, hidden_layers=hidden_layers, nodes=n_classes
        )
        self.target_graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.target_graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.target_graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)

        self.target_graph_2_fea = gcn.Graph_to_Featuremaps_savemem(
            input_channels=input_channels,
            output_channels=out_channels,
            hidden_layers=hidden_layers,
            nodes=n_classes,
        )
        self.target_skip_conv = nn.Sequential(
            *[nn.Conv2d(input_channels, input_channels, kernel_size=1), nn.ReLU(True)]
        )

    def load_source_model(self, state_dict):
        own_state = self.state_dict()
        # for name inshop_cos own_state:
        #    print name
        new_state_dict = OrderedDict()
        for name, param in state_dict.items():
            name = name.replace("module.", "")
            if (
                "graph" in name
                and "source" not in name
                and "target" not in name
                and "fc_graph" not in name
                and "transpose_graph" not in name
            ):
                if "featuremap_2_graph" in name:
                    name = name.replace(
                        "featuremap_2_graph", "source_featuremap_2_graph"
                    )
                else:
                    name = name.replace("graph", "source_graph")
            new_state_dict[name] = 0
            if name not in own_state:
                if "num_batch" in name:
                    continue
                print('unexpected key "{}" in state_dict'.format(name))
                continue
                # if isinstance(param, own_state):
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print(
                    "While copying the parameter named {}, whose dimensions in the model are"
                    " {} and whose dimensions in the checkpoint are {}, ...".format(
                        name, own_state[name].size(), param.size()
                    )
                )
                continue  # i add inshop_cos 2018/02/01
            own_state[name].copy_(param)
            # print 'copying %s' %name

        missing = set(own_state.keys()) - set(new_state_dict.keys())
        if len(missing) > 0:
            print('missing keys in state_dict: "{}"'.format(missing))

    def get_target_parameter(self):
        l = []
        other = []
        for name, k in self.named_parameters():
            if "target" in name or "semantic" in name:
                l.append(k)
            else:
                other.append(k)
        return l, other

    def get_semantic_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if "semantic" in name:
                l.append(k)
        return l

    def get_source_parameter(self):
        l = []
        for name, k in self.named_parameters():
            if "source" in name:
                l.append(k)
        return l

    def forward(self, input, adj1_target=None, adj2_source=None, adj3_transfer=None):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode="bilinear", align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(
            x, size=low_level_features.size()[2:], mode="bilinear", align_corners=True
        )

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)

        ### add graph

        # target graph
        # print('x size',x.size(),adj1.size())
        graph = self.target_featuremap_2_graph(x)

        # graph combine
        # print(graph.size(),source_2_target_graph.size())
        # graph = self.fc_graph.forward(graph,relu=True)
        # print(graph.size())

        graph = self.target_graph_conv1.forward(graph, adj=adj1_target, relu=True)
        graph = self.target_graph_conv2.forward(graph, adj=adj1_target, relu=True)
        graph = self.target_graph_conv3.forward(graph, adj=adj1_target, relu=True)
        # print(graph.size(),x.size())
        # graph = self.gcn_encode.forward(graph,relu=True)
        # graph = self.graph_conv2.forward(graph,adj=adj2,relu=True)
        # graph = self.gcn_decode.forward(graph,relu=True)
        graph = self.target_graph_2_fea.forward(graph, x)
        x = self.target_skip_conv(x)
        x = x + graph

        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode="bilinear", align_corners=True)

        return x


#######################
# transfer model
#######################


class deeplab_xception_transfer_projection(deeplab_xception_transfer_basemodel):
    def __init__(
        self,
        nInputChannels=3,
        n_classes=7,
        os=16,
        input_channels=256,
        hidden_layers=128,
        out_channels=256,
        transfer_graph=None,
        source_classes=20,
    ):
        super(deeplab_xception_transfer_projection, self).__init__(
            nInputChannels=nInputChannels,
            n_classes=n_classes,
            os=os,
            input_channels=input_channels,
            hidden_layers=hidden_layers,
            out_channels=out_channels,
        )
        self.source_featuremap_2_graph = gcn.Featuremaps_to_Graph(
            input_channels=input_channels,
            hidden_layers=hidden_layers,
            nodes=source_classes,
        )
        self.source_graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.source_graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.source_graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.transpose_graph = gcn.Graph_trans(
            in_features=hidden_layers,
            out_features=hidden_layers,
            adj=transfer_graph,
            begin_nodes=source_classes,
            end_nodes=n_classes,
        )
        self.fc_graph = gcn.GraphConvolution(hidden_layers * 3, hidden_layers)

    def forward(self, input, adj1_target=None, adj2_source=None, adj3_transfer=None):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode="bilinear", align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(
            x, size=low_level_features.size()[2:], mode="bilinear", align_corners=True
        )

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)

        ### add graph
        # source graph
        source_graph = self.source_featuremap_2_graph(x)
        source_graph1 = self.source_graph_conv1.forward(
            source_graph, adj=adj2_source, relu=True
        )
        source_graph2 = self.source_graph_conv2.forward(
            source_graph1, adj=adj2_source, relu=True
        )
        source_graph3 = self.source_graph_conv2.forward(
            source_graph2, adj=adj2_source, relu=True
        )

        source_2_target_graph1_v5 = self.transpose_graph.forward(
            source_graph1, adj=adj3_transfer, relu=True
        )
        source_2_target_graph2_v5 = self.transpose_graph.forward(
            source_graph2, adj=adj3_transfer, relu=True
        )
        source_2_target_graph3_v5 = self.transpose_graph.forward(
            source_graph3, adj=adj3_transfer, relu=True
        )

        # target graph
        # print('x size',x.size(),adj1.size())
        graph = self.target_featuremap_2_graph(x)

        source_2_target_graph1 = self.similarity_trans(source_graph1, graph)
        # graph combine 1
        # print(graph.size())
        # print(source_2_target_graph1.size())
        # print(source_2_target_graph1_v5.size())
        graph = torch.cat(
            (
                graph,
                source_2_target_graph1.squeeze(0),
                source_2_target_graph1_v5.squeeze(0),
            ),
            dim=-1,
        )
        graph = self.fc_graph.forward(graph, relu=True)

        graph = self.target_graph_conv1.forward(graph, adj=adj1_target, relu=True)

        source_2_target_graph2 = self.similarity_trans(source_graph2, graph)
        # graph combine 2
        graph = torch.cat(
            (graph, source_2_target_graph2, source_2_target_graph2_v5), dim=-1
        )
        graph = self.fc_graph.forward(graph, relu=True)

        graph = self.target_graph_conv2.forward(graph, adj=adj1_target, relu=True)

        source_2_target_graph3 = self.similarity_trans(source_graph3, graph)
        # graph combine 3
        graph = torch.cat(
            (graph, source_2_target_graph3, source_2_target_graph3_v5), dim=-1
        )
        graph = self.fc_graph.forward(graph, relu=True)

        graph = self.target_graph_conv3.forward(graph, adj=adj1_target, relu=True)

        # print(graph.size(),x.size())

        graph = self.target_graph_2_fea.forward(graph, x)
        x = self.target_skip_conv(x)
        x = x + graph

        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode="bilinear", align_corners=True)

        return x

    def similarity_trans(self, source, target):
        sim = torch.matmul(
            F.normalize(target, p=2, dim=-1),
            F.normalize(source, p=2, dim=-1).transpose(-1, -2),
        )
        sim = F.softmax(sim, dim=-1)
        return torch.matmul(sim, source)

    def load_source_model(self, state_dict):
        own_state = self.state_dict()
        # for name inshop_cos own_state:
        #    print name
        new_state_dict = OrderedDict()
        for name, param in state_dict.items():
            name = name.replace("module.", "")

            if (
                "graph" in name
                and "source" not in name
                and "target" not in name
                and "fc_" not in name
                and "transpose_graph" not in name
            ):
                if "featuremap_2_graph" in name:
                    name = name.replace(
                        "featuremap_2_graph", "source_featuremap_2_graph"
                    )
                else:
                    name = name.replace("graph", "source_graph")
            new_state_dict[name] = 0
            if name not in own_state:
                if "num_batch" in name:
                    continue
                print('unexpected key "{}" in state_dict'.format(name))
                continue
                # if isinstance(param, own_state):
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print(
                    "While copying the parameter named {}, whose dimensions in the model are"
                    " {} and whose dimensions in the checkpoint are {}, ...".format(
                        name, own_state[name].size(), param.size()
                    )
                )
                continue  # i add inshop_cos 2018/02/01
            own_state[name].copy_(param)
            # print 'copying %s' %name

        missing = set(own_state.keys()) - set(new_state_dict.keys())
        if len(missing) > 0:
            print('missing keys in state_dict: "{}"'.format(missing))


class deeplab_xception_transfer_projection_savemem(
    deeplab_xception_transfer_basemodel_savememory
):
    def __init__(
        self,
        nInputChannels=3,
        n_classes=7,
        os=16,
        input_channels=256,
        hidden_layers=128,
        out_channels=256,
        transfer_graph=None,
        source_classes=20,
    ):
        super(deeplab_xception_transfer_projection_savemem, self).__init__(
            nInputChannels=nInputChannels,
            n_classes=n_classes,
            os=os,
            input_channels=input_channels,
            hidden_layers=hidden_layers,
            out_channels=out_channels,
        )
        self.source_featuremap_2_graph = gcn.Featuremaps_to_Graph(
            input_channels=input_channels,
            hidden_layers=hidden_layers,
            nodes=source_classes,
        )
        self.source_graph_conv1 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.source_graph_conv2 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.source_graph_conv3 = gcn.GraphConvolution(hidden_layers, hidden_layers)
        self.transpose_graph = gcn.Graph_trans(
            in_features=hidden_layers,
            out_features=hidden_layers,
            adj=transfer_graph,
            begin_nodes=source_classes,
            end_nodes=n_classes,
        )
        self.fc_graph = gcn.GraphConvolution(hidden_layers * 3, hidden_layers)

    def forward(self, input, adj1_target=None, adj2_source=None, adj3_transfer=None):
        x, low_level_features = self.xception_features(input)
        # print(x.size())
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode="bilinear", align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.concat_projection_conv1(x)
        x = self.concat_projection_bn1(x)
        x = self.relu(x)
        # print(x.size())
        x = F.upsample(
            x, size=low_level_features.size()[2:], mode="bilinear", align_corners=True
        )

        low_level_features = self.feature_projection_conv1(low_level_features)
        low_level_features = self.feature_projection_bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        # print(low_level_features.size())
        # print(x.size())
        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)

        ### add graph
        # source graph
        source_graph = self.source_featuremap_2_graph(x)
        source_graph1 = self.source_graph_conv1.forward(
            source_graph, adj=adj2_source, relu=True
        )
        source_graph2 = self.source_graph_conv2.forward(
            source_graph1, adj=adj2_source, relu=True
        )
        source_graph3 = self.source_graph_conv2.forward(
            source_graph2, adj=adj2_source, relu=True
        )

        source_2_target_graph1_v5 = self.transpose_graph.forward(
            source_graph1, adj=adj3_transfer, relu=True
        )
        source_2_target_graph2_v5 = self.transpose_graph.forward(
            source_graph2, adj=adj3_transfer, relu=True
        )
        source_2_target_graph3_v5 = self.transpose_graph.forward(
            source_graph3, adj=adj3_transfer, relu=True
        )

        # target graph
        # print('x size',x.size(),adj1.size())
        graph = self.target_featuremap_2_graph(x)

        source_2_target_graph1 = self.similarity_trans(source_graph1, graph)
        # graph combine 1
        graph = torch.cat(
            (
                graph,
                source_2_target_graph1.squeeze(0),
                source_2_target_graph1_v5.squeeze(0),
            ),
            dim=-1,
        )
        graph = self.fc_graph.forward(graph, relu=True)

        graph = self.target_graph_conv1.forward(graph, adj=adj1_target, relu=True)

        source_2_target_graph2 = self.similarity_trans(source_graph2, graph)
        # graph combine 2
        graph = torch.cat(
            (graph, source_2_target_graph2, source_2_target_graph2_v5), dim=-1
        )
        graph = self.fc_graph.forward(graph, relu=True)

        graph = self.target_graph_conv2.forward(graph, adj=adj1_target, relu=True)

        source_2_target_graph3 = self.similarity_trans(source_graph3, graph)
        # graph combine 3
        graph = torch.cat(
            (graph, source_2_target_graph3, source_2_target_graph3_v5), dim=-1
        )
        graph = self.fc_graph.forward(graph, relu=True)

        graph = self.target_graph_conv3.forward(graph, adj=adj1_target, relu=True)

        # print(graph.size(),x.size())

        graph = self.target_graph_2_fea.forward(graph, x)
        x = self.target_skip_conv(x)
        x = x + graph

        ###
        x = self.semantic(x)
        x = F.upsample(x, size=input.size()[2:], mode="bilinear", align_corners=True)

        return x

    def similarity_trans(self, source, target):
        sim = torch.matmul(
            F.normalize(target, p=2, dim=-1),
            F.normalize(source, p=2, dim=-1).transpose(-1, -2),
        )
        sim = F.softmax(sim, dim=-1)
        return torch.matmul(sim, source)

    def load_source_model(self, state_dict):
        own_state = self.state_dict()
        # for name inshop_cos own_state:
        #    print name
        new_state_dict = OrderedDict()
        for name, param in state_dict.items():
            name = name.replace("module.", "")

            if (
                "graph" in name
                and "source" not in name
                and "target" not in name
                and "fc_" not in name
                and "transpose_graph" not in name
            ):
                if "featuremap_2_graph" in name:
                    name = name.replace(
                        "featuremap_2_graph", "source_featuremap_2_graph"
                    )
                else:
                    name = name.replace("graph", "source_graph")
            new_state_dict[name] = 0
            if name not in own_state:
                if "num_batch" in name:
                    continue
                print('unexpected key "{}" in state_dict'.format(name))
                continue
                # if isinstance(param, own_state):
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print(
                    "While copying the parameter named {}, whose dimensions in the model are"
                    " {} and whose dimensions in the checkpoint are {}, ...".format(
                        name, own_state[name].size(), param.size()
                    )
                )
                continue  # i add inshop_cos 2018/02/01
            own_state[name].copy_(param)
            # print 'copying %s' %name

        missing = set(own_state.keys()) - set(new_state_dict.keys())
        if len(missing) > 0:
            print('missing keys in state_dict: "{}"'.format(missing))


# if __name__ == '__main__':
# net = deeplab_xception_transfer_projection_v3v5_more_savemem()
# img = torch.rand((2,3,128,128))
# net.eval()
# a = torch.rand((1,1,7,7))
# net.forward(img, adj1_target=a)
