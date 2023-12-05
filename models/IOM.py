from __future__ import division

import math
import sys

import numpy as np
import scipy.io as scio
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from utils.model_util import norm_col_init, weights_init

from .model_io import ModelOutput

object_list = ['AlarmClock', 'Book', 'Bowl', 'CellPhone', 'Chair', 'CoffeeMachine', 'DeskLamp', 'FloorLamp',
'Fridge', 'GarbageCan', 'Kettle', 'Laptop', 'LightSwitch', 'Microwave', 'Pan', 'Plate', 'Pot',
'RemoteControl', 'Sink', 'StoveBurner', 'Television', 'Toaster',]


class Multihead_Attention(nn.Module):     
    """
    multihead_attention
    """
    def __init__(self,
                 hidden_dim,
                 C_q=None,
                 C_k=None,
                 num_heads=1,                   
                 dropout_rate=0.0):
        super(Multihead_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        C_q = C_q if C_q else hidden_dim
        C_k = C_k if C_k else hidden_dim
        self.linear_Q = nn.Linear(C_q, hidden_dim)   
        self.linear_K = nn.Linear(C_k, hidden_dim)
        self.linear_V = nn.Linear(C_k, hidden_dim)
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=dropout_rate)
        self.linear_out = nn.Linear(num_heads, 1)

    def forward(self, Q, K, V = None):
        """
        :param Q: A 3d tensor with shape of [T_q, C_q]   
        :param K: A 3d tensor with shape of [T_k, C_k]   
        :param V: A 3d tensor with shape of [T_v, C_v]   
        :return:
        """
        # num_heads = 8
        num_heads = self.num_heads
        N = 1                                           #batch
        # Q  [1 ,1 ,576]
        Q = Q.unsqueeze(dim = 0)             
        # K [1, 22, 262]
        K = K.unsqueeze(dim = 0)

        Q_l = nn.ReLU()(self.linear_Q(Q)) #   Q_l [1 ,1 ,512] 
            
        # else:
        #     K_l = nn.ReLU()(self.linear_K(K))
        K_l = nn.ReLU()(self.linear_K(K))
        
        if V != None:
            V_l = nn.ReLU()(self.linear_V(V))
            # if V_num - V_l.size()[0] > 0:
            #     make_num = torch.zeros(V_num - V_l.size()[0], V_l.size()[1]).to(V_l.device) 
            #     V_l = torch.cat((V_l, make_num), dim = 0)

        # self.hidden_dim = 512
        Q_split = Q_l.split(split_size=self.hidden_dim // num_heads, dim=2)  
        K_split = K_l.split(split_size=self.hidden_dim // num_heads, dim=2)

        # Q_是torch.Size([8, 1, 64]), K_ 是torch.Size([8, 22, 64])
        Q_ = torch.cat(Q_split, dim=0)  # (h*N, T_q, C/h)                    
        K_ = torch.cat(K_split, dim=0)  # (h*N, T_k, C/h)

        # 这一步就是计算注意力机制，
        outputs = torch.bmm(Q_, K_.transpose(2, 1))    #(h*N, T_q(1), T_k)
        # Scale
        outputs = outputs / (K_.size()[-1] ** 0.5)   
        # Dropouts   
        outputs = self.dropout(outputs) 
        outputs = outputs.permute(1, 2, 0)
        
        outputs = self.linear_out(outputs)   ## (1, num_point, 1)
        attention = nn.Softmax(dim=1)(outputs).squeeze(dim=2) 
        # if V != None:
        #     attention = torch.cat((attention, torch.zeros(1, V_num - V.size()[0]).to(V_l.device)), dim=1)            

        out = None
        if V != None:
            out = torch.mm(attention, V_l)

        return out, attention.transpose(1, 0)  


class IOM(torch.nn.Module):
    def __init__(self, args):
        action_space = args.action_space
        self.num_cate = args.num_category
        resnet_embedding_sz = args.hidden_state_sz
        hidden_state_sz = args.hidden_state_sz
        super(IOM, self).__init__()

        self.image_size = 300
        self.conv1 = nn.Conv2d(resnet_embedding_sz, 64, 1)

        self.action_at_a = nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), requires_grad=False)
        self.action_at_b = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]), requires_grad=False)
        self.action_at_scale = nn.Parameter(torch.tensor(0.58), requires_grad=False) 
        self.graph_detection_feature = nn.Sequential(
            nn.Linear(262, 128),
            nn.ReLU(),
            nn.Linear(128, 49),
        )

        self.embed_action = nn.Linear(action_space, 10)

        self.nav_embedding_dim = 32
        pointwise_in_channels = 64 + self.num_cate + 10 + self.nav_embedding_dim  + 32 #  

        self.pointwise = nn.Conv2d(pointwise_in_channels, 64, 1, 1)

        self.lstm_input_sz = 7 * 7 * 64

        self.hidden_state_sz = hidden_state_sz
        self.lstm = nn.LSTM(self.lstm_input_sz, hidden_state_sz, 2)
        num_outputs = action_space
        self.critic_linear_1 = nn.Linear(hidden_state_sz, 64)
        self.critic_linear_2 = nn.Linear(64, 1)
        self.actor_linear = nn.Linear(hidden_state_sz, num_outputs)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain("relu")
        self.conv1.weight.data.mul_(relu_gain)

        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01
        )
        self.actor_linear.bias.data.fill_(0)

        self.critic_linear_1.weight.data = norm_col_init(
            self.critic_linear_1.weight.data, 1.0
        )
        self.critic_linear_1.bias.data.fill_(0)
        self.critic_linear_2.weight.data = norm_col_init(
            self.critic_linear_2.weight.data, 1.0
        )
        self.critic_linear_2.bias.data.fill_(0)

        self.lstm.bias_ih_l0.data.fill_(0)
        self.lstm.bias_ih_l1.data.fill_(0)
        self.lstm.bias_hh_l0.data.fill_(0)
        self.lstm.bias_hh_l1.data.fill_(0)
        self.dropout_rate = 0.35
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.info_embedding = nn.Linear(5,49)

        
        self.target_object_attention = torch.nn.Parameter(torch.FloatTensor(self.num_cate, self.num_cate), requires_grad=True)
        self.target_object_attention.data.fill_(1/22)   

        self.scene_object_attention = torch.nn.Parameter(torch.FloatTensor(4, self.num_cate, self.num_cate), requires_grad=True)
        self.scene_object_attention.data.fill_(1/22)   

        self.attention_weight = torch.nn.Parameter(torch.FloatTensor(2), requires_grad=True)
        self.attention_weight.data.fill_(1/2)


        self.avgpool = nn.AdaptiveAvgPool2d((1,1))    #(64,7,7) -> (64,1,1)

        self.muti_head_attention = Multihead_Attention(hidden_dim = 512, C_q = resnet_embedding_sz + 64, C_k = 262, num_heads = 8, dropout_rate = 0.3)
        self.conf_threshod = 0.6
        self.tar_cross_attention = Multihead_Attention(hidden_dim = 32, C_q = 32, C_k = 32, num_heads = 4, dropout_rate = 0.1)

        self.num_cate_embed = nn.Sequential(
            nn.Linear(self.num_cate, 32),  
            nn.ReLU(),
            nn.Linear(32, 64),  
            nn.ReLU(),
        )

        self.nav_dim = args.nav_dim  
        self.nav_length = args.nav_length

        self.linear_graph = nn.Sequential(
            nn.Linear(self.nav_dim + 2, 16),  
            nn.ReLU(),
            nn.Linear(16, 32),  
            nn.ReLU(),
        )

        self.map_length = 21
        self.map_embedding_dim = 32
        self.linear_obstacle_map = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
        )
        self.reduce_obstacle_map = nn.Sequential(
            nn.Linear(14, 1),
            nn.ReLU(),
        )

        self.coord_dim = 6
    
        self.norm  = nn.LayerNorm(pointwise_in_channels)
        self.target_attent_coord = nn.Sequential(
            nn.Linear(70, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # self.navgraph_embedding = Navigation_Graph_Embedding(hidden_dim = self.nav_embedding_dim, C_in = 16, num_heads = 3, dropout = 0, nav_length = self.nav_length)


    def one_hot(self, spa):  

        y = torch.arange(spa).unsqueeze(-1)  
        y_onehot = torch.FloatTensor(spa, spa)  

        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)    

        return y_onehot   ## (22,22)

    def get_coord(self, coord, start_coord):   
        
        if start_coord['horizon'] == -10:   
            start_coord['x'] = coord.x
            # 【】【】
            start_coord['y'] = coord.z
            start_coord['rotation'] = coord.rotation * math.pi/180
            start_coord['horizon'] = coord.horizon * math.pi/180

        x = coord.x - start_coord['x']
        # 【】【】
        y = coord.z - start_coord['y']
        rotate = coord.rotation * math.pi/180 - start_coord['rotation']
        horizon = coord.horizon * math.pi/180  
      
        coord = torch.tensor([x, y, rotate, horizon])
        return coord, start_coord
    
    def coord_centring(self, navgraph, coord):
        navgraph_copy = torch.zeros(navgraph.shape[0], self.nav_dim + 2).to(navgraph.device)
        nav_coord = navgraph[:,5: 7]
        rotation = navgraph[:,7]
        horizon = navgraph[:,8]   

        navgraph_copy[:,: 5] = navgraph[:,: 5]         
        navgraph_copy[:,5: 7] = nav_coord - coord[:2]  
        navgraph_copy[:,7] = torch.sin(rotation - coord[2])
        navgraph_copy[:,8] = torch.cos(rotation - coord[2])
        navgraph_copy[:,9] = torch.sin(horizon - coord[3])
        navgraph_copy[:,10] = torch.cos(horizon - coord[3])

        return navgraph_copy
    
    def map_coord_centring(self, obstacle_map, coord):
        obstacle_map_copy = torch.zeros(obstacle_map.shape[0], 10).to(obstacle_map.device)
        obstacle_map_copy[:, : -2] = obstacle_map[:, : -2] 
        obstacle_map_copy[:,-2: ] = obstacle_map[:, -2: ] - coord[ : 2]

        return obstacle_map_copy


    def embedding(self, state, target, imp_obstacle_map, obstacle_index, last_action, last_action_success, action_embedding_input, 
                target_object, nav_graph, coord, start_coord):


        at_v = torch.mul(target['scores'].unsqueeze(dim=1), target['indicator']) 
        at = torch.mul(torch.max(at_v), self.action_at_scale) 
        action_at = torch.mul(at, self.action_at_a) + self.action_at_b  

        target_object = target['indicator']                  
        
        action_embedding = F.relu(self.embed_action(action_embedding_input)) 
        action_reshaped = action_embedding.view(1, 10, 1, 1).repeat(1, 1, 7, 7)  

        image_embedding = F.relu(self.conv1(state))  

        x = self.dropout(image_embedding)

        target_appear = target['features']
        target_conf = target['scores'].unsqueeze(dim=1)
        target_bbox = target['bboxes'] / self.image_size

        target = torch.cat((target_appear, target_bbox, target_conf, target_object), dim=1)  

        target_object_attention = F.softmax(self.target_object_attention, 0)        
       
        attention_weight = F.softmax(self.attention_weight, 0)    
           
        object_attention = target_object_attention * attention_weight[0]
        
        object_select = torch.sign(target_conf - 0.6)  #(22,1)
        object_select[object_select > 0] = 0                       
        object_select[object_select < 0] = - object_select[object_select < 0]   #(1,22)      
        object_select_appear = object_select.squeeze().expand(262, 22).bool()           
        target_mutiHead = target.masked_fill(object_select_appear.t(),0)             

        image_object_attention = self.avgpool(state).squeeze(dim = 2).squeeze(dim = 0).t()   #(1,512)  
        spa = self.one_hot(self.num_cate).to(target.device)      
        num_cate_index = torch.mm(spa.t(), target_object).t()
        num_cate_index = self.num_cate_embed(num_cate_index)   
        image_object_attention = torch.cat((image_object_attention, num_cate_index), dim = 1)  #(1,512+64=576)
        image_object_attention = self.muti_head_attention(image_object_attention, target_mutiHead)[1]

        target_attention= torch.mm(object_attention, target_object)   
        target_attention = target_attention + image_object_attention * attention_weight[1]  
        
        target = F.relu(self.graph_detection_feature(target))    
        target = target * target_attention                       
        target_embedding = target.reshape(1, self.num_cate, 7, 7)    
        target_embedding = self.dropout(target_embedding)  

        
        last_rotate = coord.rotation
        coord, start_coord = self.get_coord(coord, start_coord)
        map_coord = coord
        coord = coord.to(target.device)      
        
        target_bbox = torch.mm(target_bbox.t(), target_object).t()      

        nav_node = torch.zeros(1,self.nav_dim).to(target.device)    
        nav_node[:,:4] = target_bbox  
        nav_node[:,4] = at             
        nav_node[:,5: 5 + 4] = coord    

        if at/self.action_at_scale > 0.4:  
            if nav_graph.sum() == 0:   
                nav_graph = nav_node
            else:
                if nav_graph.shape[0] < self.nav_length:   
                    nav_graph = torch.cat((nav_graph, nav_node), dim = 0)
                else:
                    nav_graph = torch.cat((nav_graph[1:, :], nav_node), dim = 0)

        # Non-local target memory aggregation
        if nav_graph.sum() == 0:     
            nav_graph_mean = torch.zeros(1 ,self.nav_embedding_dim, 7, 7).to(target.device)    # (1, 32)
        else:
            nav_graph_center = self.coord_centring(nav_graph, coord)  
            target_index = torch.cat((num_cate_index, 
                           torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 1.0]).unsqueeze(0).to(target.device)), dim = 1)
            nav_graph_embeded = self.linear_graph(nav_graph_center)  #  (n, 11) -> (n, 16)

            target_embeded = self.target_attent_coord(target_index)
            nav_embeded = self.tar_cross_attention(target_embeded, nav_graph_embeded, nav_graph_embeded)[0]

                
            nav_graph_mean = nav_embeded.view(1, self.nav_embedding_dim, 1, 1).repeat(1, 1, 7, 7)  #  (1, 32, 7, 7)
            nav_graph_mean = self.dropout(nav_graph_mean)
        
        # Implicit obstacle map 
        obstacle_node = torch.cat((torch.zeros(1, 8), map_coord[0 : 2].unsqueeze(0)), dim = 1).to(target.device)
        if last_action == 0:  
            if last_action_success:
                imp_obstacle_map[obstacle_index[-1].type(torch.long), int(last_rotate/45)] = 1
                
                index =torch.where(((imp_obstacle_map[:, -2:] - obstacle_node[:, -2:]).pow(2).sum(1)) == 0)[0]
                if index.size(0) == 0:
                    if imp_obstacle_map.size()[0] < 14:
                        imp_obstacle_map = torch.cat((imp_obstacle_map, obstacle_node), dim = 0)
                    else:
                        imp_obstacle_map = torch.cat((imp_obstacle_map[1:, :], obstacle_node), dim = 0)
                    obstacle_index.append((torch.ones(1, 1) * -1).to(target.device)) 
                else:
                    # imp_obstacle_map[obstacle_index.type(torch.long), int(last_rotate/45)] = 1
                    obstacle_index.append(index) 
            else:
                imp_obstacle_map[obstacle_index[-1].type(torch.long), int(last_rotate/45)] = -1
        
        # print(obstacle_index, '\n')
        # print(imp_obstacle_map, '\n')
        
        if imp_obstacle_map.sum() == 0:  
            map_embedding = torch.zeros(1 , self.map_embedding_dim).to(target.device)
        else:    
            obstacle_map = self.map_coord_centring(imp_obstacle_map, coord)
            obs_map_fea = self.linear_obstacle_map(obstacle_map)
            if 14 - obs_map_fea.size()[0] > 0:
                make_num2 = torch.zeros(14 - obs_map_fea.size()[0], obs_map_fea.size()[1]).to(target.device) 
                obs_map_fea = torch.cat((obs_map_fea, make_num2), dim = 0)
            
            map_embedding = self.reduce_obstacle_map(obs_map_fea.t()).t()

            # map_embedding = torch.mean(obs_map_fea, dim=0).unsqueeze(0) 
        
        obs_map_embedding = map_embedding.view(1, self.map_embedding_dim, 1, 1).repeat(1, 1, 7, 7)  #  (1, 32, 7, 7)
        obs_map_embedding = self.dropout(obs_map_embedding)
        
        #Adaptive Fusion
        x = torch.cat((x, target_embedding, action_reshaped, nav_graph_mean, obs_map_embedding), dim=1)      
        x = self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        x = F.relu(self.pointwise(x))
        x = self.dropout(x)
        
        # 【】【】
        out = x.contiguous().view(x.size(0), -1)  

        return out, image_embedding, action_at, nav_graph, start_coord, imp_obstacle_map, obstacle_index

    def a3clstm(self, embedding, prev_hidden_h, prev_hidden_c):

        embedding = embedding.reshape([1, 1, self.lstm_input_sz])      
        output, (hx, cx) = self.lstm(embedding, (prev_hidden_h, prev_hidden_c))  
  
        x = output.reshape([1, self.hidden_state_sz])    

        actor_out = self.actor_linear(x)   
        critic_out = self.critic_linear_1(x)    
        critic_out = self.critic_linear_2(critic_out)   

        return actor_out, critic_out, (hx, cx)

    def forward(self, model_input, model_options):
        coord = model_input.coord
        start_coord = model_input.start_coord
        nav_graph = model_input.nav_graph                                            
        target_object = model_input.target_object  

        state = model_input.state  
        (hx, cx) = model_input.hidden   

        target = model_input.target_class_embedding  
        action_probs = model_input.action_probs 

        if len(model_input.actions):
            last_action = model_input.actions[-1][0, 0]
        else:
            last_action = -1

        last_action_success = model_input.last_action_success
        imp_obstacle_map = model_input.imp_obstacle_map
        obstacle_index = model_input.obstacle_index    

        x, image_embedding , action_at,  nav_graph, start_coord, imp_obstacle_map, obstacle_index= self.embedding(state, target, imp_obstacle_map, obstacle_index, last_action, last_action_success,
                                                                action_probs, target_object, nav_graph, coord, start_coord)
        actor_out, critic_out, (hx, cx) = self.a3clstm(x, hx, cx)
        actor_out = torch.mul(actor_out, action_at)  
        return ModelOutput(
            value=critic_out,            
            logit=actor_out,            
            hidden=(hx, cx),            
            embedding=image_embedding,   
            nav_graph = nav_graph,
            start_coord = start_coord,
            imp_obstacle_map = imp_obstacle_map,
            obstacle_index = obstacle_index
        )
