import random
from Network import *
import torch
import torch.nn as nn
import torch.optim as optim
import dgl
import numpy as np

def hard_copy(target_net, source_net):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(param.data)

def soft_copy(target_net, source_net, tau=0.001):
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau*param + (1-tau)*target_param)

class MRFAgent(object):
    def __init__(self, args=None, optimizer=None, device='cpu', writer=None,
                 epsilon=1.0, added_u_dim=0, mode="train", gumbel_temp=None):

        self.args = args
        self.added_u_dim = added_u_dim

        self.pair_comp = self.args['pair_comp']
        self.writer = writer
        self.num_updates = 0

        # Initialize neural network dimensions.
        self.dim_lstm_out = 100
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dqn_net = LSTMMRF(2, self.added_u_dim, 100, 100, 100, 70,
                                      5, 70, 30, 20, pair_comp=self.pair_comp, args=args).to(self.device)
        self.target_dqn_net = LSTMMRF(2, self.added_u_dim, 100, 100, 100, 70,
                                            5, 70, 30, 20, pair_comp=self.pair_comp, args=args).to(self.device)

        hard_copy(self.target_dqn_net, self.dqn_net)
        self.mode = mode

        # Initialize hidden states for prediction.
        self.hiddens = None
        self.hiddens_u = None
        self.prev_hiddens = None
        self.prev_hiddens_u = None
        self.target_hiddens = None
        self.target_hiddens_u = None

        # Set params for Ad Hoc BPTT
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.dqn_net.parameters(), lr=self.args['lr'])

        self.targ_vals = []
        self.pred_vals = []
        self.logit_probs_pi = []
        self.logit_probs_theta = []
        self.joint_actions = []
        # NOTE: JIANHONG CHANGE 01/01/2024
        self.pred_reg_vals = []
        self.target_reg_vals = []

        self.epsilon = epsilon
        self.loss_module = nn.MSELoss()
        self.loss_module_f = nn.CrossEntropyLoss()
        # NOTE: JIANHONG CHANGE 01/01/2024
        self.loss_reg = nn.MSELoss()

        self.outs = []

    def step(self, obs, eval=False):
        # torch.Size([3, 2]) torch.Size([1, 12]) torch.Size([1, 3, 100])*2 torch.Size([1, 3, 100])*2 is a tuple
        p_graph, n_ob, u_ob, n_hiddens, n_hiddens_u = self.prep(
            obs, self.hiddens, self.hiddens_u
        )

        out, log_prob_theta, hids, hids_u = self.dqn_net(
            p_graph, n_ob, n_ob, u_ob, n_hiddens, n_hiddens_u
        )

        self.prev_hiddens = self.hiddens
        self.prev_hiddens_u = self.hiddens_u
        self.hiddens = hids
        self.hiddens_u = hids_u
        
        acts = torch.argmax(out, dim=-1).tolist()

        # Select outputs so likelihood only computed for nodes other than agent.
        zero_indexes, offset = [0], 0
        num_nodes = p_graph.batch_num_nodes
        all_num_nodes = sum(num_nodes)

        for a in num_nodes[:-1]:
            offset += a
            zero_indexes.append(offset)

        non_zero_indices = torch.Tensor([k for k in range(all_num_nodes) if not (k in zero_indexes)]).long()

        if not eval:
            acts = [a if random.random() > self.epsilon else random.randint(0, 4) for a in acts]
            self.logit_probs_theta.append(log_prob_theta[non_zero_indices,:])
            self.outs.append(out.gather(1, torch.tensor(acts)[:, None]))

        return acts

    def compute_target(self, obs, acts, rewards, done, n_obs, add_storage=True):

        if add_storage:
            prev_p_graph, prev_n_ob, prev_u_ob, prev_n_hiddens, prev_n_hiddens_u = self.prep(
                obs, self.prev_hiddens, self.prev_hiddens_u
            )

        prep_outs = self.prep(
            n_obs, self.target_hiddens, self.target_hiddens_u, with_acts=add_storage, add_acts=acts
        )

        target_out, _, target_hids, target_hids_u = self.target_dqn_net(
            prep_outs[0], prep_outs[1], prep_outs[1],
            prep_outs[2], prep_outs[3], prep_outs[4]
        )

        self.target_hiddens = target_hids
        self.target_hiddens_u = target_hids_u

        if add_storage:
            out = self.dqn_net(
                prev_p_graph, prev_n_ob, prev_n_ob, prev_u_ob,
                prev_n_hiddens, prev_n_hiddens_u,
                mrf_mode="joint", joint_acts=prep_outs[5]
            )

            # Select outputs so pseudolikelihood only computed for nodes other than agent.
            zero_indexes, offset = [0], 0
            num_nodes = prev_p_graph.batch_num_nodes
            all_num_nodes = sum(num_nodes)

            for a in num_nodes[:-1]:
                offset += a
                zero_indexes.append(offset)

            non_zero_indices_list = [k for k in range(all_num_nodes) if not (k in zero_indexes)]

            self.joint_actions.extend([x for idx, x in enumerate(prep_outs[5]) if idx in non_zero_indices_list])
            # print (f'This is the shape of out[0]: {out[0].shape}')
            # NOTE: JIANHONG CHANGE 23/03/2024
            # print (f'This is the shape of out[0]: {out[0].shape}')
            # print (f'This is the shape of out[1]: {out[1].shape}')
            # print (f'This is the shape of prep_outs[6]: {prep_outs[6].shape}')
            # self.pred_vals.append(out[0])
            self.pred_vals.append(out[0][prep_outs[6]] if self.args['update_manner']=='variant' else out[0])
            # NOTE: JIANHONG CHANGE 01/01/2024
            # self.pred_reg_vals.append(out[1])
            # self.target_reg_vals.append(out[2])
            if self.args['graph']=='complete':
                reg_agent_mask = []
                for i, a in enumerate(num_nodes):
                    # print (f'This is i: {i}')
                    # print (f'This is a: {a}')
                    # print (f'This is the [prep_outs[6][i]]*a: \n{[prep_outs[6][i]]*a}')
                    reg_agent_mask += [prep_outs[6][i]]*a
                reg_agent_mask = torch.tensor(reg_agent_mask)
                # print (f'This is the shape of reg_agent_mask: {reg_agent_mask.shape}')
            elif self.args['graph']=='star':
                reg_agent_mask = prep_outs[6]
            else:
                raise NotImplementedError("The graph structure has not been implemented.")

            self.pred_reg_vals.append(out[1][reg_agent_mask] if self.args['update_manner']=='variant' else out[1])
            self.target_reg_vals.append(out[2][reg_agent_mask] if self.args['update_manner']=='variant' else out[2])

            rew_t = torch.Tensor(rewards)[:, None].to(self.device)
            dones = torch.Tensor(done)[:, None].to(self.device)

            targets = rew_t + self.args['gamma'] * (1 - dones) * torch.max(target_out, dim=-1, keepdim=True)[0]
            # self.targ_vals.append(targets)
            self.targ_vals.append(targets[prep_outs[6]] if self.args['update_manner']=='variant' else targets)

    def detach_hiddens(self):
        if self.args['seq_model'] == 'gru':
            self.hiddens = self.hiddens.detach()
            self.hiddens_u = self.hiddens_u.detach()
        elif self.args['seq_model'] == 'lstm':
            self.hiddens = (self.hiddens[0].detach(), self.hiddens[1].detach())
            self.hiddens_u = (self.hiddens_u[0].detach(), self.hiddens_u[1].detach())
 
    def prep(self, obs, hiddens, hiddens_u, with_acts=False, add_acts=None):
        # obs for env=1
        # OrderedDict([('num_agents', array([[3.]], dtype=float32)), 
        # ('oppo_actions', array([[ 3.,  2., -1., -1.]], dtype=float32)),
        #  ('opponent_info', array([[3., 1., 4., 4., 1., 0., 0., 0., 1., 0., 0., 0.]], dtype=float32)), 
        # ('remaining_flags', array([[ 1.,  1.,  1., -1., -1.]], dtype=float32)), 
        # 2*max_player_num=10 ('teammate_location', array([[ 0.,  9.,  6.,  6.,  4.,  1., -1., -1., -1., -1.]], dtype=float32))])
        graph_list = []
        num_agents = [num_agent[0] for num_agent in obs["num_agents"]]
        prev_num_agents = [(a != -1).sum() for a in obs["remaining_flags"]]
        unc_complete_filter = [ob[:n_p_agent] for ob, n_p_agent in zip(obs["remaining_flags"], prev_num_agents)]
        complete_filter = np.concatenate(
            unc_complete_filter,
            axis=-1
        )
        # NOTE: JIANHONG CHANGE 23/03/2024
        agent_mask_flag = (torch.tensor(num_agents) <= torch.tensor(prev_num_agents))

        # Create graphs inputted to GNN.
        for num_agent in num_agents:
            num_agent = int(num_agent)
            graph_ob = dgl.DGLGraph()
            graph_ob.add_nodes(num_agent)
            # if self.args['star_graph']:
            #     src, dst = zip(*([(a, 0) for a in range(num_agent) if a != 0]+[(0, a) for a in range(num_agent) if a != 0]))
            # else:
            #     src, dst = zip(*[(a, b) for a in range(num_agent) for b in range(num_agent) if a != b])
            if self.args['graph']=='star':
                src, dst = zip(*([(a, 0) for a in range(num_agent) if a != 0]+[(0, a) for a in range(num_agent) if a != 0]))
            elif self.args['graph']=='complete':
                src, dst = zip(*[(a, b) for a in range(num_agent) for b in range(num_agent) if a != b])
            else:
                raise NotImplementedError(f"The dynamic affinity graph structure {self.args['graph']} has not been implemented.")
            graph_ob.add_edges(src, dst)
            graph_list.append(graph_ob)

        graph_batch = dgl.batch(graph_list)
        
        # Parse inputs into node inputs
        num_nodes = graph_batch.batch_num_nodes
        # n_ob is the location of all players
        n_ob = torch.cat([torch.Tensor([obs['teammate_location'][id][2 * idx:2 * idx + 2]]).float()
                        for id, num_node in enumerate(num_nodes) for idx in range(num_node)], dim=0)
        # u_ob is the food infomation and orientation
        u_ob = torch.Tensor(obs["opponent_info"])

        # Create filters to decide which hidden vectors to maintain
        # For newly added agents, hiddens set to zeros
        # For remaining agents, hiddens continues from prev timestep
        node_filter_np = np.where(complete_filter == 1)[0]
        node_filter = torch.Tensor(node_filter_np).long()

        current_node_offsets, offset = [0], 0
        for cur_num_node in num_nodes[:-1]:
            offset += cur_num_node
            current_node_offsets.append(offset)

        # NOTE: set up the indices of the remaining agents in the current agent array
        new_indices = []
        filter_idxes = [np.arange((filter == 1).sum()) for filter in unc_complete_filter]
        for offset, filter in zip(current_node_offsets, filter_idxes):
            new_indices.append(torch.Tensor(offset + filter).long())
        complete_new_filter = torch.cat(
            new_indices,
            dim=-1
        )

        # Create action vectors for opponent modelling
        if with_acts:
            acts = []
            for first_act, last_act, prev_node in zip(add_acts, obs["oppo_actions"], prev_num_agents):
                acts.append(first_act)
                acts.extend(last_act[:prev_node-1])

        if self.args['seq_model'] == 'gru':
            # Filter hidden vectors for remaining agents
            # Add zero vectors for newly added agents
            n_hid = torch.zeros([1, graph_batch.number_of_nodes(), self.dim_lstm_out])

            if not (hiddens is None):
                collected_hiddens = hiddens[:,node_filter,:]
                n_hid[:, complete_new_filter, :] = collected_hiddens

            n_hid_u = torch.zeros([1, graph_batch.number_of_nodes(), self.dim_lstm_out])

            if not (hiddens_u is None):
                collected_hiddens = hiddens_u[:, node_filter, :]
                n_hid_u[:, complete_new_filter, :] = collected_hiddens

        elif self.args['seq_model'] == 'lstm':
            # Filter hidden vectors for remaining agents
            # Add zero vectors for newly added agents
            n_hid = (torch.zeros([1, graph_batch.number_of_nodes(), self.dim_lstm_out]),
                     torch.zeros([1, graph_batch.number_of_nodes(), self.dim_lstm_out]))

            if not (hiddens is None):
                collected_hiddens = (hiddens[0][:,node_filter,:], hiddens[1][:,node_filter,:])
                n_hid[0][:, complete_new_filter, :] = collected_hiddens[0]
                n_hid[1][:, complete_new_filter, :] = collected_hiddens[1]

            n_hid_u = (torch.zeros([1, graph_batch.number_of_nodes(), self.dim_lstm_out]),
                         torch.zeros([1,graph_batch.number_of_nodes(), self.dim_lstm_out]))

            if not (hiddens_u is None):
                collected_hiddens = (hiddens_u[0][:, node_filter, :], hiddens_u[1][:, node_filter, :])
                n_hid_u[0][:, complete_new_filter, :] = collected_hiddens[0]
                n_hid_u[1][:, complete_new_filter, :] = collected_hiddens[1]

        if with_acts:
            # NOTE: JIANHONG CAHNGE 23/03/2024
            # return graph_batch, n_ob, u_ob, n_hid, n_hid_u, acts
            return graph_batch, n_ob, u_ob, n_hid, n_hid_u, acts, agent_mask_flag
        
        # n_ob, u_ob, n_hid, n_hid_u : torch.Size([3, 2]) torch.Size([1, 12]) torch.Size([1, 3, 100])*2 torch.Size([1, 3, 100])*2 is a tuple
        return graph_batch, n_ob, u_ob, n_hid, n_hid_u

    def reset(self):
        self.hiddens = None
        self.hiddens_u = None
        self.target_hiddens = None
        self.target_hiddens_u = None

        self.targ_vals = []
        self.pred_vals = []
        self.logit_probs_pi = []
        self.logit_probs_theta = []
        self.joint_actions = []
        # NOTE: JIANHONG CHANGE 01/01/2024
        self.pred_reg_vals = []
        self.target_reg_vals = []

        self.outs = []

    def load_parameters(self, filename):
        self.dqn_net.load_state_dict(torch.load(filename))
        self.target_dqn_net.state_dict(torch.load(filename + "_target_dqn"))

    def save_parameters(self, filename):
        torch.save(self.dqn_net.state_dict(), filename)
        torch.save(self.target_dqn_net.state_dict(), filename + "_target_dqn")

    def set_epsilon(self, eps):
        self.epsilon = eps
    
    def update(self):
        self.optimizer.zero_grad()

        # Util losses
        pred_tensor = torch.cat(self.pred_vals, dim=0)
        target_tensor = torch.cat(self.targ_vals, dim=0)

        # NOTE: JIANHONG CHANGE 01/01/2024
        pred_reg_vals = torch.cat(self.pred_reg_vals)
        target_reg_vals = torch.cat(self.target_reg_vals)

        # modelling losses
        joint_actions = torch.Tensor(self.joint_actions).long()

        joint_logit_theta = torch.cat(self.logit_probs_theta, dim=0)
        theta_cross_entropy = self.loss_module_f(joint_logit_theta, joint_actions)

        loss_pred = theta_cross_entropy
        val_loss = self.loss_module(pred_tensor, target_tensor.detach())

        # NOTE: JIANHONG CHANGE 01/01/2024
        reg_loss =self.loss_reg(pred_reg_vals, target_reg_vals.detach())

        # loss = val_loss + self.args['weight_predict'] * loss_pred
        loss = val_loss + self.args['weight_predict'] * loss_pred + self.args['weight_regularizer'] * reg_loss

        self.writer.add_scalar('loss/q_loss', val_loss, self.num_updates)
        self.writer.add_scalar('loss/theta_cross_entropy', theta_cross_entropy, self.num_updates)
        # NOTE: JIANHONG CHANGE 01/01/2024
        self.writer.add_scalar('loss/reg_loss', reg_loss, self.num_updates)

        loss.backward()
        self.optimizer.step()

        soft_copy(self.target_dqn_net, self.dqn_net, self.args['tau'])
        self.target_dqn_net.hard_copy_fs(self.dqn_net)

        self.targ_vals = []
        self.pred_vals = []
        self.logit_probs_pi = []
        self.logit_probs_theta = []
        self.joint_actions = []
        # NOTE: JIANHONG CHANGE 01/01/2024
        self.pred_reg_vals = []
        self.target_reg_vals = []

        self.outs = []

        self.detach_hiddens()

        self.num_updates += 1
        
