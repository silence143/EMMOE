import habitat
import socket
from habitat.utils.visualizations.utils import observations_to_image
# from habitat_extensions.utils.visualizations.utils import observations_to_image
from PIL import Image
import os
import pdb
import json
import numpy as np
import gzip
from shutil import copyfileobj

import argparse
import os.path as osp
import random
import time

import torch
torch.backends.cudnn.enabled = False

from habitat.config import Config
from habitat_baselines.common.baseline_registry import baseline_registry

import habitat_extensions.tasks.rearrange
import mobile_manipulation.ppo
from mobile_manipulation.config import get_config
from mobile_manipulation.utils.common import get_run_name, warn

from mobile_manipulation.utils.env_utils import (
    VectorEnv,
    construct_envs,
    make_env_fn,
)
from habitat.core.environments import get_env_class
from mobile_manipulation.utils.wrappers import HabitatActionWrapper
from mobile_manipulation.test_ppo_trainer_interface import PPOTrainerCTL
import imageio


class Communicator_LLE:
    def __init__(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect(('127.0.0.1', 10010))

    def send_env_images(self, images):
        message = json.dumps(images)
        self.s.sendall(message.encode('utf-8'))

    def receive_subtask(self):
        subtask = self.s.recv(1024).decode()
        return subtask

    def send_feedback(self, feedback, signal):
        message = json.dumps([feedback, signal])
        self.s.sendall(message.encode('utf-8'))
        self.s.recv(1024).decode()

    def close_connection(self):
        self.s.close()

def save_numpy_images_as_video(images, output_path, fps=30, quality=5):
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        quality=quality
    )
    for im in images:
        writer.append_data(im)
    writer.close()

def get_env_images(save_path, state_info, cnt):
    config = make_cfg('go to', 'TV', state_info)
    env = make_env_fn(
            config,
            get_env_class(config.ENV_NAME),
            wrappers=[HabitatActionWrapper],
        )
    obs = env.reset()

    images = []
    for i in range(4):
        for j in range(19):
            base_action = [0, 1]
            result, _, _, _ = env.step({"action": "BaseVelAction",
            "action_args": {
                "velocity": base_action,
            }})
        draw_ob = result['robot_head_rgb_256']   
        ob = Image.fromarray(draw_ob)
        directions = ['left', 'back', 'right', 'front']
        image_path = os.path.join(save_path, f"subtask{cnt}_{directions[i]}.png")
        ob.save(image_path)
        images.append(image_path)
    
    env.close()
    return images

def init_state():
    config = make_cfg('go to', 'TV', None)
    env = make_env_fn(
            config,
            get_env_class(config.ENV_NAME),
            wrappers=[HabitatActionWrapper],
        )
    obs = env.reset()
    env_state_info = env.env._env._sim.get_state()
    for joint_id, (motor_id, jms) in env_state_info['robot_state']["motors"].items():
        new_jms = {}
        for k in ["motor_type","position_target","position_gain","velocity_target","velocity_gain","max_impulse"]:
            # print('!!!!!!!!!!!!!!!!',"{:s}={:s}".format(k, str(getattr(jms, k))))
            new_jms[k] = getattr(jms, k)
        env_state_info['robot_state']["motors"][joint_id] = (motor_id, new_jms)

    if env_state_info['grasped_obj'] is not None:
        env_state_info['grasped_obj'] = env_state_info['grasped_obj'].handle
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!',env_state_info['grasped_obj'])
    state_info = env_state_info
    env.close()
    return state_info    

def reset_arm(state_info):
    config = make_cfg('go to', 'TV', state_info)
    env = make_env_fn(
            config,
            get_env_class(config.ENV_NAME),
            wrappers=[HabitatActionWrapper],
        )

    obs = env.reset()

    images = {'head':[], 'third':[]}
    # reset arm
    arm_tgt_qpos = env.env._env._sim.pyb_robot.IK(np.array([0.5, 0.0, 1.0]), max_iters=100)
    cur_qpos = np.array(env.env._env._sim.robot.arm_joint_pos)
    tgt_qpos = np.array(arm_tgt_qpos)
    n_step = np.ceil(np.max(np.abs(tgt_qpos - cur_qpos)) / 0.1)
    n_step = max(1, int(n_step))
    plan = np.linspace(cur_qpos, tgt_qpos, n_step)
    for i in plan:
        env.env._env._sim.robot.arm_motor_pos = i
        base_action = [0, 0]
        result, _, _, _ = env.step({"action": "BaseVelAction",
            "action_args": {
                "velocity": base_action,
            }})
        images['head'].append(result['robot_head_rgb_256'])
        images['third'].append(result['robot_third_rgb'])

    env_state_info = env.env._env._sim.get_state()
    for joint_id, (motor_id, jms) in env_state_info['robot_state']["motors"].items():
        new_jms = {}
        for k in ["motor_type","position_target","position_gain","velocity_target","velocity_gain","max_impulse"]:
            # print('!!!!!!!!!!!!!!!!',"{:s}={:s}".format(k, str(getattr(jms, k))))
            new_jms[k] = getattr(jms, k)
        env_state_info['robot_state']["motors"][joint_id] = (motor_id, new_jms)

    if env_state_info['grasped_obj'] is not None:
        env_state_info['grasped_obj'] = env_state_info['grasped_obj'].handle
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!',env_state_info['grasped_obj'])
    state_info = env_state_info
    env.close()
    return images, state_info

def execution(action, input_item, inventory, state_info):
    if action not in ["go to", "open", "close", "pick", "put", "end"]:
        return 'fail', f'{action} is not in the action list! You should only choose actions in the list.', state_info, None
    
    # Logical error
    if inventory != 'None' and action in ['pick', 'open', 'close']:
        return 'fail', f'Unable to {action}, the hand is full', state_info, None
    if inventory == 'None' and action == 'put':
        return 'fail', f'Unable to {action}, the hand is empty', state_info, None
    if action == 'put' and state_info['art_objs_qpos']['fridge_:0000'][1] < 0.8  and input_item == 'fridge': 
        return 'fail', f'Unable to {action}, the {input_item} is closed, you should open it first', state_info, None
    if action == 'put' and state_info['art_objs_qpos']['kitchen_counter_:0000'][drawer_cnt] < 0.25  and  'drawer' in input_item:
        return 'fail', f'Unable to {action}, the {input_item} is closed, you should open it first', state_info, None
    if action in ['open','close'] and 'drawer' not in input_item and 'fridge' not in input_item:
        return 'fail', f'Can not {action} {input_item}! Please choose another object', state_info, None
    def load_name_mapping():
        with open('name_dict.txt', 'r') as file:
            content = file.read()

        lines = content.split('\n')
        print(lines[-1])
        name_dict = {}
        for i in range(0, len(lines), 3):
            value = lines[i].strip().strip(':')
            key = lines[i + 1].strip()
            keys = key.split('/')
            for i in keys:
                name_dict[i] = value
            # print(name_dict)

        return name_dict
    # mapping item name
    mapping_dict = load_name_mapping()
    if input_item in mapping_dict:
        item = mapping_dict[input_item]
    else:
        return 'fail', f'{input_item} does not exist! Please choose another object', state_info, None

    # find exact item                                           
    obj_pos = None          
    all_possible_items = []
    rigid_data = state_info['rigid_objs_T']
    for obj in rigid_data.items():
        name = obj[0].split(':')[0][:-1]
        if name == item:
            all_possible_items.append(obj)

    agent_pos = state_info['robot_state']['T'] 
    min_distance = float('inf')
    agent_pos = [agent_pos[3][0], agent_pos[3][1], agent_pos[3][2]]
    nearest_obj = None
    min_index = 0
    if len(all_possible_items)!= 0:
        for i in range(len(all_possible_items)):
            obj = all_possible_items[i]
            obj_pos = [obj[1][3][0], obj[1][3][1], obj[1][3][2]]
            distance = np.linalg.norm(np.array(agent_pos) - np.array(obj_pos))
            if distance < min_distance:
                min_distance = distance
                nearest_obj = obj
                min_index = i
        
    if nearest_obj is not None:
        print(nearest_obj)
        obj_pos = [[nearest_obj[1][0][0], nearest_obj[1][1][0], nearest_obj[1][2][0], nearest_obj[1][3][0]],
                   [nearest_obj[1][0][1], nearest_obj[1][1][1], nearest_obj[1][2][1], nearest_obj[1][3][1]],
                   [nearest_obj[1][0][2], nearest_obj[1][1][2], nearest_obj[1][2][2], nearest_obj[1][3][2]],
                   [nearest_obj[1][0][3], nearest_obj[1][1][3], nearest_obj[1][2][3], nearest_obj[1][3][3]]] 

    with open(data_path, 'rb') as f:
        file_content = f.read()
    bg_data = json.loads(file_content.decode('utf-8'))
    episodes = bg_data['episodes'][0]
    scene_ins_path = episodes['scene_id']
    with open(scene_ins_path, 'rb') as f:
        scene_content = f.read()
    scene_data = json.loads(scene_content.decode('utf-8'))

    if action in ['put', 'open', 'close']:             
        if item in  ['fridge', 'kitchen_counter']:
            obj_list = scene_data['articulated_object_instances']
            for i in obj_list:
                if i["template_name"] == item:
                    obj_pos_trans = i['translation']
                    obj_pos_rota = i['rotation']
                    obj_pos = create_transformation_matrix(obj_pos_trans, obj_pos_rota).tolist()
        else:
            obj_list = scene_data['object_instances']
            for i in obj_list:
                if i["template_name"] == item:
                    obj_pos_trans = i['translation']
                    obj_pos_rota = i['rotation']
                    obj_pos = create_transformation_matrix(obj_pos_trans, obj_pos_rota).tolist()
        if item == 'living_room':
            obj_pos = [[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,2.0],[0.0,0.0,0.0,1.0]]
        if item == 'stairs':
                obj_pos = [[1.0,0.0,0.0,1.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,-0.8],[0.0,0.0,0.0,1.0]]
        if item == 'second_floor':
                obj_pos = [[1.0,0.0,0.0,2.6],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,-1.2],[0.0,0.0,0.0,1.0]]

    # Distance error  
    if action in ['pick', 'put', 'open', 'close']:
        if obj_pos is None:
            return 'fail', f'{input_item} does not exist! Please choose another object', state_info, None
        obj_pos_np = np.array([obj_pos[0][3],obj_pos[1][3],obj_pos[2][3]])  
        agent_pos_np = np.array(agent_pos)                       
        distance = np.linalg.norm(agent_pos_np - obj_pos_np)
        if distance > 2:
            return 'fail', f'Unable to {action}, the target is far away', state_info, None
        if distance < 0.1:
            return 'fail', f'Unable to {action}, the target is too close', state_info, None
    
    obj_pos_ins = None   
    if action == 'go to':         
        if item in  ['fridge', 'kitchen_counter']:
            obj_list = scene_data['articulated_object_instances']
            for i in obj_list:
                if i["template_name"] == item:
                    obj_pos_ins_trans = i['translation']
                    obj_pos_ins_rota = i['rotation']
                    obj_pos_ins = create_transformation_matrix(obj_pos_ins_trans, obj_pos_ins_rota).tolist()

        elif obj_pos is None:
            obj_list = scene_data['object_instances']
            for i in obj_list:
                if i["template_name"] == item:
                    obj_pos_ins_trans = i['translation']
                    obj_pos_ins_rota = i['rotation']
                    obj_pos_ins = create_transformation_matrix(obj_pos_ins_trans, obj_pos_ins_rota).tolist()
            if item == 'living_room':
                obj_pos_ins = [[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,2.0],[0.0,0.0,0.0,1.0]]
            if item == 'stairs':
                obj_pos_ins = [[1.0,0.0,0.0,1.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,-0.8],[0.0,0.0,0.0,1.0]]
            if item == 'second_floor':
                obj_pos_ins = [[1.0,0.0,0.0,2.6],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,-1.2],[0.0,0.0,0.0,1.0]]
            if obj_pos_ins is None:
                return 'fail', f'{input_item} does not exist! Please choose another object', state_info, None
    
    # execute
    if retry == 0:
        buffer.add_subtask(action+','+input_item)
    signal, info, state_info, video_images = execute_M3(action, input_item, item, obj_pos, obj_pos_ins, min_index, state_info)

    if signal == 'fail':
        return 'fail', f'Unable to {action}, time out', state_info, video_images
    return signal, info, state_info, video_images

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
    ])

def create_transformation_matrix(translation, rotation_quaternion):
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = quaternion_to_rotation_matrix(rotation_quaternion)
    transformation_matrix[:3, 3] = translation
    return transformation_matrix

def execute_M3(action, input_item, item, obj_pos, obj_pos_ins, min_index, state_info):
    signal ='success'
    info = "None"
    global drawer_cnt

    # current dataset
    with open(data_path, 'rb') as f_in:
        file_content = json.loads(f_in.read().decode('utf-8'))
        if action == 'go to':
            if obj_pos is not None:
                file_content['episodes'][0]["targets"] = {item+f'_:{min_index:04d}': obj_pos}
                file_content['episodes'][0]['target_receptacles'] = [['',0]]
            else:
                temp_item = file_content['episodes'][0]["rigid_objs"][-1][0].split('.')[0]
                if item == 'fridge':
                    file_content['episodes'][0]["targets"] = {temp_item+f'_:0000': obj_pos_ins}
                    file_content['episodes'][0]['target_receptacles'] = [["fridge_:0000",0]]
                elif item == 'kitchen_counter':
                    if 'drawer' in input_item:
                        if retry == 0:
                            drawer_cnt = (drawer_cnt +1)%(len(drawer_list))
                        file_content['episodes'][0]["targets"] = {temp_item+f'_:0000': obj_pos_ins}
                        file_content['episodes'][0]['target_receptacles'] = [["kitchen_counter_:0000",drawer_list[drawer_cnt]]]  #left top 2, middle top 5, right top 7                        
                    elif 'sink' in input_item:
                        obj_pos_ins[0][3] = obj_pos_ins[0][3] +0.25
                        obj_pos_ins[1][3] = obj_pos_ins[1][3] 
                        obj_pos_ins[2][3] = obj_pos_ins[2][3] +0.31989800930023193
                        file_content['episodes'][0]["targets"] = {temp_item+f'_:0000': obj_pos_ins}
                        file_content['episodes'][0]['target_receptacles'] = [['',0]]  
                    else:                                                                               # left part
                        obj_pos_ins[0][3] = obj_pos_ins[0][3] +0.25
                        obj_pos_ins[1][3] = obj_pos_ins[1][3] 
                        obj_pos_ins[2][3] = obj_pos_ins[2][3] +1.0976535081863403
                        file_content['episodes'][0]["targets"] = {temp_item+f'_:0000': obj_pos_ins}
                        file_content['episodes'][0]['target_receptacles'] = [["",0]]  
                else:
                    file_content['episodes'][0]["targets"] = {temp_item+f'_:0000': obj_pos_ins}
                    file_content['episodes'][0]['target_receptacles'] = [['',0]] 
            file_content['episodes'][0]['goal_receptacles'] = [['',0]]

        elif action == 'pick':
            file_content['episodes'][0]["targets"] = {item+f'_:{min_index:04d}': obj_pos}
            file_content['episodes'][0]['target_receptacles'] = [['',0]]
            file_content['episodes'][0]['goal_receptacles'] = [['',0]]

        elif action == 'put':
            if item == 'fridge':
                obj_pos[0][3] = obj_pos[0][3] + np.random.uniform(0,0.05)
                obj_pos[1][3] = obj_pos[1][3] +0.4                
            elif item == 'kitchen_counter':
                if 'drawer' in input_item:
                    drawer_put_list = [1.0976535081863403, -0.2, -1]                                 # left top  middle top   right top
                    obj_pos[0][3] = obj_pos[0][3] +0.3
                    obj_pos[1][3] = 1.0                                 #              0.68072               
                    obj_pos[2][3] = obj_pos[2][3] +drawer_put_list[drawer_cnt]          #  1.0976535081863403    -0.2     -1
                elif 'sink' in input_item:
                    obj_pos[0][3] = obj_pos[0][3] -0.16285690665245056
                    obj_pos[1][3] = obj_pos[1][3] +0.7953284978866577
                    obj_pos[2][3] = obj_pos[2][3] +0.31989800930023193 + np.random.uniform(0,0.1)
                else:                                                            # left part
                    obj_pos[0][3] = obj_pos[0][3] + np.random.uniform(-0.1,0.1)
                    obj_pos[1][3] = obj_pos[1][3] +0.7953284978866577
                    obj_pos[2][3] = obj_pos[2][3] +1.0976535081863403 + np.random.uniform(-0.3,0.3)
            else:
                obj_pos[0][3] = obj_pos[0][3] + np.random.uniform(0,0.05)
                obj_pos[2][3] = obj_pos[2][3] + np.random.uniform(0,0.1)
            file_content['episodes'][0]["targets"] = {state_info['grasped_obj']: obj_pos}   ############################
            file_content['episodes'][0]['target_receptacles'] = [['',0]]
            file_content['episodes'][0]['goal_receptacles'] = [['',0]]

        elif action in ['open', 'close']:
            temp_item = file_content['episodes'][0]["rigid_objs"][-1][0].split('.')[0]
            file_content['episodes'][0]['targets'] = {temp_item+f'_:0000': [[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,2.0],[0.0,0.0,0.0,1.0]]}
            file_content['episodes'][0]['goal_receptacles'] = [['',0]]
            if item == 'fridge':
                file_content['episodes'][0]['target_receptacles'] = [["fridge_:0000",0]]
            if item == 'kitchen_counter':
                file_content['episodes'][0]['target_receptacles'] = [["kitchen_counter_:0000",drawer_list[drawer_cnt]]]  #left top 2, middle top 5, right top 7

        with open('mobile_manipulation/temp1.json', 'w') as f_out:
            json.dump(file_content, f_out, indent=4)
        
    with open('mobile_manipulation/temp1.json', 'rb') as f_in, gzip.open(input_path, 'wb') as f_out:
        copyfileobj(f_in, f_out)
    
    os.remove('mobile_manipulation/temp1.json')
    

    signal, state_info, video_images = M3(action, item, state_info)
    return signal, info, state_info, video_images

def M3(action, item, state_info):
    run_type = 'eval'
    config = make_cfg(action, item, state_info)
    env = make_env_fn(
            config,
            get_env_class(config.ENV_NAME),
            wrappers=[HabitatActionWrapper],
        )
    signal, state_info, video_images = execute_exp(config, run_type, env)
    return signal, state_info, video_images

def preprocess_config(config: Config, config_path: str, run_type: str):

    config.defrost()

    # placeholders supported in config
    fileName = osp.splitext(osp.basename(config_path))[0]
    runName = get_run_name()
    timestamp = time.strftime("%y%m%d")
    substitutes = dict(
        fileName=fileName,
        runName=runName,
        runType=run_type,
        timestamp=timestamp,
    )

    config.PREFIX = config.PREFIX.format(**substitutes)
    config.BASE_RUN_DIR = config.BASE_RUN_DIR.format(**substitutes)

    for key in ["CHECKPOINT_FOLDER"]:
        config[key] = config[key].format(
            prefix=config.PREFIX, baseRunDir=config.BASE_RUN_DIR, **substitutes
        )

    for key in ["LOG_FILE", "TENSORBOARD_DIR", "VIDEO_DIR"]:
        if key not in config:
            warn(f"'{key}' is missed in the config")
            continue
        if run_type == "train":
            prefix = config.PREFIX
        else:
            prefix = config.EVAL.PREFIX or config.PREFIX
        config[key] = config[key].format(
            prefix=prefix,
            baseRunDir=config.BASE_RUN_DIR,
            **substitutes,
        )

    # Support relative path like "@/ckpt.pth"
    config.EVAL.CKPT_PATH = config.EVAL.CKPT_PATH.replace(
        "@", config.CHECKPOINT_FOLDER
    )
    config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT

    config.freeze()

def make_cfg(action, item, state_info):
    config_list = {
        'go to':'configs/rearrange/skills/set_table/nav_v1_disc_SCR.yaml',  
                # configs/rearrange/skills/tidy_house/nav_v1_disc_SCR.yaml   configs/rearrange/skills/prepare_groceries/nav_v1_disc_SCR.yaml
        'pick':'configs/rearrange/skills/tidy_house/pick_v1_joint_SCR.yaml',
                # configs/rearrange/skills/set_table/pick_v1_drawer_joint_SCR.yaml  (drawer)  configs/rearrange/skills/prepare_groceries/pick_v1_joint_SCR.yaml  (fridge)
        'put':'configs/rearrange/skills/tidy_house/place_v1_joint_SCR.yaml',
                # configs/rearrange/skills/prepare_groceries/place_v1_joint_SCR.yaml (fridge)
        'open':{'kitchen_counter':'configs/rearrange/skills/set_table/open_drawer_v0A_joint_SCR.yaml','fridge':'configs/rearrange/skills/set_table/open_fridge_v0A_joint_SCR.yaml'},
        'close':{'kitchen_counter':'configs/rearrange/skills/set_table/close_drawer_v0A_joint_SCR.yaml', 'fridge':'configs/rearrange/skills/set_table/close_fridge_v0A_joint_SCR.yaml'}        
    }


    if action in ['open','close'] and item in ['kitchen_counter','fridge']:
        config_path = config_list[action][item]
    elif action == 'put':
        if item == 'fridge':
            config_path = 'configs/rearrange/skills/prepare_groceries/place_v1_joint_SCR.yaml'
        else:
            config_path = 'configs/rearrange/skills/tidy_house/place_v1_joint_SCR.yaml'
    elif action == 'pick':
        if buffer.check_condition() == 'fridge':
            config_path = 'configs/rearrange/skills/prepare_groceries/pick_v1_joint_SCR.yaml'
        elif buffer.check_condition() == 'drawer':
            config_path = 'configs/rearrange/skills/set_table/pick_v1_drawer_joint_SCR.yaml'
        else:
            config_path = 'configs/rearrange/skills/tidy_house/pick_v1_joint_SCR.yaml'
    elif action == 'go to':
        config_path = 'configs/rearrange/skills/set_table/nav_v1_disc_SCR.yaml'
        # configs/rearrange/skills/tidy_house/nav_v1_disc_SCR.yaml   
        # configs/rearrange/skills/prepare_groceries/nav_v1_disc_SCR.yaml


    opts = ['PREFIX', 'seed=100']
    run_type = 'eval'

    # args.opts = args.opts + ['TARGET_INDEX', '1']
    ckpt_base = config_path.split("/")[-2]
    my_action = config_path.split("/")[-1].split("_")[0]


    # input art pos
    input_qpos = None
    # input robot state
    input_start_state = None 
    # index of the target  
    my_obj = 0 
    # index of the receptacle   
    my_receptacle = 0     
    # only for my_action == 'nav' 
    # pick, place, open_fridge, close_fridge, open_drawer, close_drawer    
    sub_task = ['place']
    # assert False, 
    config = get_config(config_path, opts)
    config['TASK_CONFIG']['TASK']['ART_POS'] = input_qpos
    config['TASK_CONFIG']['TASK']['START_STATE'] = input_start_state
    config['TASK_CONFIG']['TASK']['TARGET_INDEX'] = my_obj

    if my_action in ['nav']:
        config['TASK_CONFIG']['TASK']['SUB_TASKS'] = sub_task
        my_obj = str(my_obj) + '_' + sub_task[0]

    if my_action in ['open','close']:
        config['TASK_CONFIG']['TASK']['HOLDER_INDEX'] = my_receptacle
        my_action = my_action + '_' + config_path.split("/")[-1].split("_")[1]

    config['LOG_FILE'] = f'mobile_manipulation/z_low_level_execution/{my_action}/{ckpt_base}/{item}/log.test.txt'
    config['VIDEO_DIR'] = f'mobile_manipulation/z_low_level_execution/{my_action}/{ckpt_base}/{item}/video'
    config['TENSORBOARD_DIR'] = f'mobile_manipulation/z_low_level_execution/{my_action}/{ckpt_base}/{item}/tb'

    config['TASK_CONFIG']['DATASET']['DATA_PATH'] = input_path

    config['TASK_CONFIG']['TASK']['STATE_INFO'] = state_info



    preprocess_config(config, config_path, run_type)

    return config

def execute_exp(config: Config, run_type: str, env) -> None:
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    if config.FORCE_TORCH_SINGLE_THREADED and torch.cuda.is_available():
        torch.set_num_threads(1)

    trainer = PPOTrainerCTL(config, env)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()
    else:
        raise NotImplementedError(run_type)
    
    return trainer.signal, trainer.state_info, trainer.images

class SubtaskBuffer:
    def __init__(self, max_size=3):
        self.buffer = []
        self.max_size = max_size

    def reset(self):
        self.buffer = []

    def add_subtask(self, subtask):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(subtask)

    def check_condition(self):
        if 'open,fridge' in self.buffer or 'go to,fridge' in self.buffer:
            return 'fridge'
        elif 'open,drawer' in self.buffer or 'go to,drawer' in self.buffer:
            return 'drawer'
        else: 
            return None

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

max_count = 20
comm = Communicator_LLE()

save_dir = "../infer"
data_path = "../EMMOE-100/data/train/1/scene.json"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
input_path = 'mobile_manipulation/M3_data1.json.gz'

for i in range(1, 4):
    save_path = os.path.join(save_dir, f"{i}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(data_path, 'rb') as f_in:
        file_content = json.loads(f_in.read().decode('utf-8'))
        temp_item_init = file_content['episodes'][0]["rigid_objs"][-1][0].split('.')[0]
        file_content['episodes'][0]['targets'] = {temp_item_init+f'_:0000': [[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,2.0],[0.0,0.0,0.0,1.0]]}
        file_content['episodes'][0]['target_receptacles'] = [['',0]] 
        file_content['episodes'][0]['goal_receptacles'] = [['',0]]
        with open('mobile_manipulation/temp1.json', 'w') as f_out:
            json.dump(file_content, f_out, indent=4)
            
    with open('mobile_manipulation/temp1.json', 'rb') as f_in, gzip.open(input_path, 'wb') as f_out:
        copyfileobj(f_in, f_out)
    os.remove('mobile_manipulation/temp1.json')

    buffer = SubtaskBuffer()
    count_steps = 1
    whole_video_head = []
    whole_video_third = []

    drawer_list = [2,4,5,7]
    drawer_cnt = 0

    state_info = init_state()
            
    while count_steps <= max_count:
        images = get_env_images(save_path, state_info, count_steps)
        comm.send_env_images(images)
        info = comm.receive_subtask()
        if "end" in info.lower():
            comm.send_feedback("None", "success")
            break

        def process_info(info):
            infos = info.split('|')
            subtask_info = infos[0].strip().split(",")
            model_choice = infos[1].strip()
            inventory = infos[2].strip()
            action = subtask_info[0].strip().lower()
            item = subtask_info[-1].strip().lower()

            return action, item, inventory
                
        action, item, inventory = process_info(info)
        # [action, target] model obj
        head_images = []
        third_images = []            
        subtask_dir = os.path.join(save_path, f"{count_steps}_{action}_{item}")
        if not os.path.exists(subtask_dir):
            os.makedirs(subtask_dir)
        for retry in range(3):
            signal, feedback, state_info, video_images = execution(action, item, inventory, state_info)
            if video_images is not None:
                    
                head_images.extend(video_images['head'])
                third_images.extend(video_images['third'])

            output_images, state_info = reset_arm(state_info)
            head_images.extend(output_images['head'])
            third_images.extend(output_images['third'])

            if action == 'put': 
                if 'time out' in feedback and state_info['grasped_obj'] is None:
                    feedback = f'Unable to {action}, and the object is missing'
                break
            if action in ['pick','open','close'] and 'time out' in feedback:
                state_info['grasped_obj'] = None

            if 'time out' in feedback:
                if retry == 2:
                    feedback = f'Unable to {action}, the subtask is too difficult to perform'
            else:
                break
                
        save_numpy_images_as_video(head_images, os.path.join(subtask_dir, "video_head.mp4"))
        save_numpy_images_as_video(third_images, os.path.join(subtask_dir, "video_third.mp4"))
        whole_video_head.extend(head_images)
        whole_video_third.extend(third_images)
                ### 
        count_steps += 1

        comm.send_feedback(feedback, signal)

    buffer.reset()
        
    save_numpy_images_as_video(whole_video_head, os.path.join(save_path, "video_head.mp4"))
    save_numpy_images_as_video(whole_video_third, os.path.join(save_path, "video_third.mp4"))