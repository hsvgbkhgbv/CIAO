import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import torch.distributions as dist
import dgl


class RFMBlock(nn.Module):
    def __init__(self, dim_in_node, dim_in_edge, dim_in_u, hidden_dim, dim_out, args):
        super(RFMBlock, self).__init__()
        self.fc_edge = nn.Linear(dim_in_edge,hidden_dim)
        self.fc_edge2 = nn.Linear(hidden_dim, dim_out)
        self.fc_node = nn.Linear(dim_in_node, hidden_dim)
        self.fc_node2 = nn.Linear(hidden_dim, dim_out)
        self.fc_u = nn.Linear(dim_in_u, hidden_dim)
        self.fc_u2 = nn.Linear(hidden_dim, dim_out)
        # Check Graph batch

        self.graph_msg = fn.copy_edge(edge='edge_feat', out='m')
        self.graph_reduce = fn.sum(msg='m', out='h')

        self.args = args

        if self.args['act_func'] == 'relu':
            self.act = nn.ReLU()
        elif self.args['act_func'] == 'leaky_relu':
            self.act = nn.LeakyReLU()

    def graph_message_func(self,edges):
        return {'m': edges.data['edge_feat'] }

    def graph_reduce_func(self,nodes):
        msgs = torch.sum(nodes.mailbox['m'], dim=1)
        return {'h': msgs}

    def compute_edge_repr(self, graph, edges, g_repr):
        edge_nums = graph.batch_num_edges
        u = torch.cat([g[None,:].repeat(num_edge,1) for g, num_edge
                       in zip(g_repr,edge_nums)], dim=0)
        inp = torch.cat([edges.data['edge_feat'],edges.src['node_feat'],edges.dst['node_feat'], u], dim=-1)

        return {'edge_feat' : self.fc_edge2(self.act(self.fc_edge(inp)))}

    def compute_node_repr(self, graph, nodes, g_repr):
        node_nums = graph.batch_num_nodes
        u = torch.cat([g[None, :].repeat(num_node, 1) for g, num_node
                       in zip(g_repr, node_nums)], dim=0)
        inp = torch.cat([nodes.data['node_feat'], nodes.data['h'], u], dim=-1)

        return {'node_feat' : self.fc_node2(F.relu(self.fc_node(inp)))}

    def compute_u_repr(self, n_comb, e_comb, g_repr):
        inp = torch.cat([n_comb, e_comb, g_repr], dim=-1)

        return self.fc_u2(self.act(self.fc_u(inp)))

    def forward(self, graph, edge_feat, node_feat, g_repr):
        graph.edata['edge_feat'] = edge_feat

        graph.ndata['node_feat'] = node_feat

        node_trf_func = lambda x: self.compute_node_repr(nodes=x, graph=graph, g_repr=g_repr)
        edge_trf_func = lambda x: self.compute_edge_repr(edges=x, graph=graph, g_repr=g_repr)
        graph.apply_edges(edge_trf_func)
        graph.update_all(self.graph_message_func, self.graph_reduce_func, node_trf_func)
        e_comb = dgl.sum_edges(graph, 'edge_feat')
        n_comb = dgl.sum_nodes(graph, 'node_feat')
        g_repr = self.compute_u_repr(n_comb, e_comb, g_repr)

        e_out = graph.edata['edge_feat']
        n_out = graph.ndata['node_feat']

        e_keys = list(graph.edata.keys())
        n_keys = list(graph.ndata.keys())
        for key in e_keys:
            graph.edata.pop(key)
        for key in n_keys:
            graph.ndata.pop(key)

        return e_out, n_out, g_repr


class UtilLayer(nn.Module):
    def __init__(self, dim_in_node, mid_pair, mid_nodes, num_acts,
                 pair_comp="avg", mid_pair_out=8, device='cpu', args=None):
        super(UtilLayer, self).__init__()
        self.pair_comp = pair_comp
        self.mid_pair = mid_pair
        self.num_acts = num_acts

        if self.pair_comp=="bmm":
            self.ju1 = nn.Linear(3*dim_in_node, self.mid_pair)
            self.ju3 = nn.Linear(self.mid_pair, self.mid_pair)
        else:
            self.ju3 = nn.Linear(self.mid_pair, self.mid_pair)
            self.ju1 = nn.Linear(3*dim_in_node, self.mid_pair)

        if self.pair_comp=="bmm":
            self.mid_pair_out = mid_pair_out
            self.ju2 = nn.Linear(self.mid_pair,num_acts*self.mid_pair_out)
        else:
            self.ju2 = nn.Linear(self.mid_pair, num_acts * num_acts)

        self.iu1 = nn.Linear(2*dim_in_node, mid_nodes)
        self.iu3 = nn.Linear(mid_nodes, mid_nodes)
        self.iu2 = nn.Linear(mid_nodes, num_acts)

        self.num_acts = num_acts
        self.device = device

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if args['act_func'] == 'relu':
            self.act = nn.ReLU()
        elif args['act_func'] == 'leaky_relu':
            self.act = nn.LeakyReLU()
        self.args = args

    def compute_node_data(self, nodes):
        # TODO: JIANHONG
        # return {'indiv_util': self.iu2(self.act(self.iu3(self.act(self.iu1(nodes.data['node_feat_u'])))))}
        # if self.args['pos_indiv']:
        #     return {'indiv_util': torch.abs(self.iu2(self.act(self.iu3(self.act(self.iu1(nodes.data['node_feat_u']))))))}
        # elif self.args['neg_indiv']:
        #     return {'indiv_util': -torch.abs(self.iu2(self.act(self.iu3(self.act(self.iu1(nodes.data['node_feat_u']))))))}
        # elif self.args['zero_indiv']:
        #     return {'indiv_util': torch.abs(self.iu2(self.act(self.iu3(self.act(self.iu1(nodes.data['node_feat_u'])))))).zero_()}
        # else:
        #     return {'indiv_util': self.iu2(self.act(self.iu3(self.act(self.iu1(nodes.data['node_feat_u'])))))}
        if self.args['indiv_range']=="pos":
            return {'indiv_util': torch.abs(self.iu2(self.act(self.iu3(self.act(self.iu1(nodes.data['node_feat_u']))))))}
        elif self.args['indiv_range']=="neg":
            return {'indiv_util': -torch.abs(self.iu2(self.act(self.iu3(self.act(self.iu1(nodes.data['node_feat_u']))))))}
        elif self.args['indiv_range']=="zero":
            return {'indiv_util': torch.abs(self.iu2(self.act(self.iu3(self.act(self.iu1(nodes.data['node_feat_u'])))))).zero_()}
        elif self.args['indiv_range']=="free":
            return {'indiv_util': self.iu2(self.act(self.iu3(self.act(self.iu1(nodes.data['node_feat_u'])))))}
        else:
            raise NotImplementedError(f"The indiv_range {self.args['indiv_range']} has not been implemented.")

    def compute_edge_data(self, edges):
        inp_u = edges.data['edge_feat_u']
        inp_reflected_u = edges.data['edge_feat_reflected_u']

        if self.pair_comp == 'bmm':
            # Compute the util components
            util_comp = self.ju2(self.act(self.ju3(self.act(self.ju1(inp_u))))).view(-1,self.num_acts, self.mid_pair_out)
            util_comp_reflected = self.ju2(self.act(self.ju3(self.act(self.ju1(inp_reflected_u))))).view(-1,self.num_acts,
                                                                                   self.mid_pair_out).permute(0,2,1)

            util_vals = torch.bmm(util_comp, util_comp_reflected).permute(0,2,1)
        else:
            util_comp = self.ju2(self.act(self.ju3(self.act(self.ju1(inp_u))))).view(-1, self.num_acts, self.num_acts)
            util_comp_reflected = self.ju2(self.act(self.ju3(self.act(self.ju1(inp_reflected_u))))).view(-1, self.num_acts,
                                                                                 self.num_acts).permute(0,2,1)

            util_vals = ((util_comp + util_comp_reflected)/2.0).permute(0,2,1)

        # final_u_factor = util_vals
        # TODO: JIANHONG
        # if self.args['pos_pair']:
        #     final_u_factor = torch.abs(util_vals)
        # elif self.args['neg_pair']:
        #     final_u_factor = -torch.abs(util_vals)
        # else:
        #     final_u_factor = util_vals
        if self.args['pair_range']=='pos':
            final_u_factor = torch.abs(util_vals)
        elif self.args['pair_range']=='neg':
            final_u_factor = -torch.abs(util_vals)
        elif self.args['pair_range']=='free':
            final_u_factor = util_vals
        else:
            raise NotImplementedError(f"The pair_range {self.args['pair_range']} has not been implemented.")
        reflected_util_vals = final_u_factor.permute(0, 2, 1)

        return {'util_vals': final_u_factor,
                'reflected_util_vals': reflected_util_vals}

    def graph_pair_inference_func(self, edges):
        src_prob, dst_prob = edges.src["probs"], edges.dst["probs"]
        edge_all_sum = (edges.data["util_vals"] * src_prob.unsqueeze(1) *
                        dst_prob.unsqueeze(-1)).sum(dim=-1).sum(dim=-1,
                        keepdim=True)
        return {'edge_all_sum_prob': edge_all_sum}

    def graph_dst_inference_func(self, edges):
        src_prob = edges.src["probs"]
        u_message = (edges.data["util_vals"] * src_prob.unsqueeze(1)).sum(dim=-1)
        return {'marginalized_u' : u_message}

    def graph_node_inference_func(self, nodes):
        indiv_util = nodes.data["indiv_util"]
        weighting = nodes.data["probs"]
        return {"expected_indiv_util" : (indiv_util*weighting).sum(dim=-1)}

    def graph_reduce_func(self, nodes):
        util_msg = torch.sum(nodes.mailbox['marginalized_u'], dim=1)
        return {'util_dst': util_msg}

    def graph_u_sum(self, graph, edges, acts):
        src, dst = graph.edges()
        acts_src = torch.Tensor([acts[idx] for idx in src.tolist()])

        u = edges.data['util_vals']
        reshaped_acts = acts_src.view(u.shape[0], 1, -1).long().repeat(1, self.num_acts, 1)
        u_msg = u.gather(-1, reshaped_acts).permute(0,2,1).squeeze(1)
        return {'u_msg': u_msg}

    def graph_sum_all(self, nodes):
        util_msg = torch.sum(nodes.mailbox['u_msg'], dim=1)
        return {'u_msg_sum': util_msg}
    
    def forward(self, graph, edge_feats_u, node_feats_u,
                edge_feat_reflected_u, mode="train",
                node_probability = None,
                joint_acts=None):

        graph.edata['edge_feat_u'] = edge_feats_u
        graph.edata['edge_feat_reflected_u'] = edge_feat_reflected_u
        graph.ndata['node_feat_u'] = node_feats_u

        n_weights = torch.zeros([node_feats_u.shape[0],1])

        zero_indexes, offset = [0], 0
        num_nodes = graph.batch_num_nodes

        # Mark all 0-th index nodes
        # NOTE: JIANHONG This is for filtering out the ad hoc agent
        for a in num_nodes[:-1]:
            offset += a
            zero_indexes.append(offset)

        n_weights[zero_indexes] = 1
        graph.ndata['weights'] = n_weights
        graph.ndata['mod_weights'] = 1-n_weights

        graph.apply_nodes(self.compute_node_data)
        graph.apply_edges(self.compute_edge_data)

        if "inference" in mode:
            graph.ndata["probs"] = node_probability
            src, dst = graph.edges()
            src_list, dst_list = src.tolist(), dst.tolist()

            # Mark edges not connected to zero
            e_nc_zero_weight = torch.zeros([edge_feats_u.shape[0],1])
            all_nc_edges = [idx for idx, (src, dst) in enumerate(zip(src_list,dst_list)) if
                            (not src in zero_indexes) and (not dst in zero_indexes)]
            e_nc_zero_weight[all_nc_edges] = 0.5
            graph.edata["nc_zero_weight"] = e_nc_zero_weight

            graph.apply_edges(self.graph_pair_inference_func)
            graph.update_all(message_func=self.graph_dst_inference_func, reduce_func=self.graph_reduce_func,
                             apply_node_func=self.graph_node_inference_func)

            # NOTE: JIANHONG This is for removing the excessive effect of message passing for the ad hoc agent
            total_connected = dgl.sum_nodes(graph, 'util_dst', 'weights') # the pair including ad hoc agent
            total_n_connected = dgl.sum_edges(graph, 'edge_all_sum_prob', 'nc_zero_weight') # the pair excluding ad hoc agent
            total_expected_others_util = dgl.sum_nodes(graph, "expected_indiv_util", "mod_weights").view(-1,1) # the individual of teammates
            total_indiv_util_zero = dgl.sum_nodes(graph, "indiv_util", "weights") # the individual of ad hoc agent

            returned_values = (total_connected + total_n_connected) + \
                              (total_expected_others_util + total_indiv_util_zero)

            e_keys = list(graph.edata.keys())
            n_keys = list(graph.ndata.keys())

            for key in e_keys:
                graph.edata.pop(key)

            for key in n_keys:
                graph.ndata.pop(key)

            return returned_values

        # NOTE: During training the representation of the pred Q-value is different from target
        m_func = lambda x: self.graph_u_sum(graph, x, joint_acts)
        graph.update_all(message_func=m_func,
                        reduce_func=self.graph_sum_all)

        indiv_u_zeros = graph.ndata['indiv_util']
        u_msg_sum_zeros = 0.5 * graph.ndata['u_msg_sum']

        graph.ndata['utils_sum_all'] = (indiv_u_zeros + u_msg_sum_zeros).gather(-1,
                                                                                torch.Tensor(joint_acts)[:,None].long())
        q_values = dgl.sum_nodes(graph, 'utils_sum_all')
        
        # NOTE: JIANHONG CHANGE 01/01/2024
        # graph.ndata['target_indiv'] = indiv_u_zeros.gather(-1, torch.Tensor(joint_acts)[:,None].long()) * graph.ndata['mod_weights']
        # graph.ndata['pred_indiv']   = indiv_u_zeros.gather(-1, torch.Tensor(joint_acts)[:,None].long()) * graph.ndata['weights']
        # target_reg_values = dgl.sum_nodes(graph, 'target_indiv')
        # pred_reg_values   = dgl.sum_nodes(graph, 'pred_indiv')
        # NOTE: JIANHONG CHANGE 06/01/2024
        if self.args['graph']=="complete":
            graph.ndata['pred_indiv']   = indiv_u_zeros.gather(-1, torch.Tensor(joint_acts)[:,None].long())
            graph.ndata['target_indiv'] = indiv_u_zeros.gather(-1, torch.Tensor(joint_acts)[:,None].long()) * graph.ndata['weights']
            def message_func_l(edges):
                return {'m_l': edges.src['target_indiv']}
            def reduce_func_l(nodes):
                return {'target_indiv_sum': torch.sum(nodes.mailbox['m_l'], dim=1)}
            graph.update_all(message_func=message_func_l, 
                            reduce_func=reduce_func_l)
            target_reg_values = graph.ndata['target_indiv_sum'] + graph.ndata['target_indiv']
            pred_reg_values   = graph.ndata['pred_indiv']
        elif self.args['graph']=="star":
            graph.ndata['target_indiv'] = indiv_u_zeros.gather(-1, torch.Tensor(joint_acts)[:,None].long()) * graph.ndata['mod_weights']
            graph.ndata['pred_indiv']   = indiv_u_zeros.gather(-1, torch.Tensor(joint_acts)[:,None].long()) * graph.ndata['weights']
            target_reg_values = dgl.sum_nodes(graph, 'target_indiv')
            pred_reg_values   = dgl.sum_nodes(graph, 'pred_indiv')
        else:
            raise NotImplementedError(f"The dynamic affinity graph structure {self.args['graph']} has not been implemented.")

        e_keys = list(graph.edata.keys())
        n_keys = list(graph.ndata.keys())

        for key in e_keys:
            graph.edata.pop(key)

        for key in n_keys:
            graph.ndata.pop(key)
        
        # return q_values
        # NOTE: JIANHONG CHANGE 01/01/2024
        return q_values, pred_reg_values, target_reg_values


class OppoModelNet(nn.Module):
    def __init__(self, dim_in_node, dim_in_u, hidden_dim,
                 dim_lstm_out, dim_mid, dim_out, act_dims,
                 dim_last, rfm_hidden_dim, last_hidden, args):
        super(OppoModelNet, self).__init__()
        self.dim_lstm_out = dim_lstm_out
        self.act_dims = act_dims

        self.mlp1 = nn.Linear(dim_lstm_out, dim_out)
        self.mlp1a = nn.Linear(dim_in_node + dim_in_u, hidden_dim)
        self.mlp1b = nn.Linear(hidden_dim, dim_mid)

        if args['seq_model'] == 'gru':
            self.lstm1 = nn.GRU(dim_mid, dim_lstm_out, batch_first=True)
        elif args['seq_model'] == 'lstm':
            self.lstm1 = nn.LSTM(dim_mid, dim_lstm_out, batch_first=True)

        self.mlp1_readout = nn.Linear(dim_last, last_hidden)
        
        self.mlp1_readout2 = nn.Linear(last_hidden, act_dims)

        self.GNBlock_theta = RFMBlock(dim_last+dim_out, 2*dim_out, 2*dim_last, rfm_hidden_dim, 
                                    dim_last, args)

        self.args = args
        if self.args['act_func'] == 'relu':
            self.act = nn.ReLU()
        elif self.args['act_func'] == 'leaky_relu':
            self.act = nn.LeakyReLU()

    def forward(self, graph, obs, hidden_n, mode="theta", add_acts=None):
        updated_n_feat = self.mlp1b(self.act(self.mlp1a(obs)))
        edge_feat = torch.zeros([graph.number_of_edges(), 0])
        g_repr = torch.zeros([len(graph.batch_num_nodes), 0])
        updated_n_feat, n_hid = self.lstm1(updated_n_feat.view(updated_n_feat.shape[0], 1, -1), hidden_n)
        updated_n_feat = self.mlp1(self.act(updated_n_feat.squeeze(1))) # 0
        # if self.args['training_mode'] == 'baseline':
        updated_e_feat, updated_n_feat, updated_u_feat = self.GNBlock_theta.forward(graph, edge_feat,
                                                                            updated_n_feat, g_repr)

        return self.mlp1_readout2(self.act(self.mlp1_readout(updated_n_feat))), n_hid


class LSTMMRF(nn.Module):
    def __init__(self, dim_in_node, dim_in_u, hidden_dim, dim_lstm_out,
                 dim_mid, dim_out, act_dims, dim_last, f_rfm_hidden_dim,
                 f_last_hidden, mid_pair=60, mid_nodes=60, pair_comp="avg",
                 mid_pair_out=5, device='cpu', args=None):
        super(LSTMMRF, self).__init__()
        self.device = device
        self.dim_lstm_out = dim_lstm_out
        self.act_dims = act_dims
        
        self.u_mlp1a = nn.Linear(dim_in_node + dim_in_u, hidden_dim)

        self.u_mlp1b = nn.Linear(hidden_dim, dim_mid)

        if args['seq_model'] == 'gru':
            self.u_lstm2 = nn.GRU(dim_mid, dim_lstm_out, batch_first=True)
        elif args['seq_model'] == 'lstm':
            self.u_lstm2 = nn.LSTM(dim_mid, dim_lstm_out, batch_first=True)

        self.u_mlp2 = nn.Linear(dim_lstm_out, dim_out)

        self.q_net = UtilLayer(dim_out, mid_pair, mid_nodes,
                               act_dims, mid_pair_out=mid_pair_out,
                               pair_comp=pair_comp, device=self.device, args=args)

        self.oppo_model_net = OppoModelNet(
            dim_in_node, dim_in_u, hidden_dim, dim_lstm_out,
            dim_mid, dim_out, act_dims,
            dim_last, f_rfm_hidden_dim, f_last_hidden, args
        )

        self.args = args
        if args['act_func'] == 'relu':
            self.act = nn.ReLU()
        elif args['act_func'] == 'leaky_relu':
            self.act = nn.LeakyReLU()

    def forward(self, graph, node_feat, node_feat_u,
                g_repr, hidden_n, hidden_n_u,
                mrf_mode="inference", joint_acts=None):
        # node_feat=node_feat_u, g_repr, hidden_n, hidden_n_u,
        # n_ob, u_ob, n_hid, n_hid_u : torch.Size([3, 2]) torch. Size([1, 12]) torch.Size([1, 3, 100])*2 torch.Size([1, 3, 100])*2 is a tuple

        u_obs = g_repr
        batch_num_nodes = graph.batch_num_nodes
        add_obs = torch.cat([feat.view(1, -1).repeat(r_num, 1) for
                             feat, r_num in zip(u_obs, batch_num_nodes)], dim=0)
        
        # obs is n_ob (is the location of all players) + u_ob (is the food infomation and orientation)
        obs = torch.cat([node_feat, add_obs], dim=-1)

        updated_n_feat_u = self.u_mlp1b(self.act(self.u_mlp1a(obs)))
        updated_n_feat_u, n_hid_u = self.u_lstm2(updated_n_feat_u.view(updated_n_feat_u.shape[0], 1, -1),
                                                 hidden_n_u)
        updated_n_feat_u_half = self.u_mlp2(self.act(updated_n_feat_u.squeeze(1)))
        
        first_elements = [0]
        offset = 0
        for a in batch_num_nodes[:-1]:
            offset += a
            first_elements.append(offset)

        # NOTE: add each node the first element (the ad hoc agent's???) of each batch of nodes
        # combined with the type (choice) of each agent as its initial node repr
        first_elements_u = updated_n_feat_u_half[first_elements, :]
        add_first_elements = torch.cat([feat.view(1, -1).repeat(r_num, 1) for
                                        feat, r_num in zip(first_elements_u, batch_num_nodes)], dim=0)
        updated_n_feat_u = torch.cat([updated_n_feat_u_half, add_first_elements], dim=-1)
        # TODO: JIANHONG
        # if self.args['learner_type']:
        #     updated_n_feat_u = torch.cat([updated_n_feat_u_half, add_first_elements], dim=-1)
        # else:
        #     updated_n_feat_u = torch.cat([updated_n_feat_u_half, add_first_elements.zero_()], dim=-1)

        edges = graph.edges()
        e_feat_u_src = updated_n_feat_u_half[edges[0]]
        e_feat_u_dst = updated_n_feat_u_half[edges[1]]

        batch_num_edges = graph.batch_num_edges
        add_first_elements_edge = torch.cat([feat.view(1, -1).repeat(r_num, 1) for
                                             feat, r_num in zip(first_elements_u, batch_num_edges)], dim=0)

        updated_e_feat_u = torch.cat([e_feat_u_src, e_feat_u_dst, add_first_elements_edge], dim=-1)
        reverse_feats_u = torch.cat([e_feat_u_dst, e_feat_u_src, add_first_elements_edge], dim=-1)

        if "inference" in mrf_mode:
            act_logits, model_hid = self.oppo_model_net(graph, obs, hidden_n)
            node_probs = dist.Categorical(logits=act_logits).probs

            out = self.q_net(
                graph, updated_e_feat_u, updated_n_feat_u, reverse_feats_u,
                mode=mrf_mode, node_probability=node_probs, joint_acts=joint_acts
            )

            return out, act_logits, model_hid, n_hid_u
        else:
            # joint_acts example: [2, 2.0, 0.0]
            # add_acts example: tensor([[0., 0., 1., 0., 0.],
            #         [0., 0., 1., 0., 0.],
            #         [1., 0., 0., 0., 0.]])
            # add_acts = torch.eye(self.act_dims)[torch.Tensor(joint_acts).long(),:]
            # act_logits, model_hid, theta_history = self.oppo_model_net(graph, obs, hidden_n, mode="pi", add_acts=add_acts)
            # out = self.q_net(
            #     graph, updated_e_feat_u,
            #     updated_n_feat_u, reverse_feats_u,
            #     mode=mrf_mode, joint_acts=joint_acts
            # )
            
            # NOTE: JIANHONG CHANGE 01/01/2024
            out, pred_reg_values, target_reg_values = self.q_net(
                                                        graph, updated_e_feat_u,
                                                        updated_n_feat_u, reverse_feats_u,
                                                        mode=mrf_mode, joint_acts=joint_acts
                                                    )

            # NOTE: JIANHONG CHANGE 01/01/2024
            # return out, out
            return out, pred_reg_values, target_reg_values

    def hard_copy_fs(self, source):
        for (k, l), (m, n), in zip(self.named_parameters(), source.named_parameters()):
            if ('oppo_model_net' in k):
                l.data.copy_(n.data)
