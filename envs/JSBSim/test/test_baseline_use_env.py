from abc import ABC
import sys
import os
# Deal with import error
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import torch
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Literal
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.utils.utils import in_range_rad, get_root_dir, in_range_rad
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv
from envs.JSBSim.model.baseline_actor import BaselineActor

class BaselineAgent(ABC):
    def __init__(self, agent_id) -> None:
        self.model_path = get_root_dir() + '/model/baseline_model.pt'
        self.actor = BaselineActor()
        self.actor.load_state_dict(torch.load(self.model_path, weights_only=True))
        self.actor.eval()
        self.agent_id = agent_id
        self.state_var = [
            c.delta_altitude,                   #  0. delta_h   (unit: m)
            c.delta_heading,                    #  1. delta_heading  (unit: °)
            c.delta_velocities_u,               #  2. delta_v   (unit: m/s)
            c.attitude_roll_rad,                #  3. roll      (unit: rad)
            c.attitude_pitch_rad,               #  4. pitch     (unit: rad)
            c.velocities_u_mps,                 #  5. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 #  6. v_body_y   (unit: m/s)
            c.velocities_w_mps,                 #  7. v_body_z   (unit: m/s)
            c.velocities_vc_mps,                #  8. vc        (unit: m/s)
            c.position_h_sl_m                   #  9. altitude  (unit: m)
        ]
        self.reset()

    def reset(self):
        self.rnn_states = np.zeros((1, 1, 128))

    @abstractmethod
    def set_delta_value(self, env, task):
        raise NotImplementedError

    def get_observation(self, env, task, delta_value):
        uid = list(env.agents.keys())[self.agent_id]
        obs = env.agents[uid].get_property_values(self.state_var)
        norm_obs = np.zeros(12)
        norm_obs[0] = delta_value[0] / 1000          #  0. ego delta altitude  (unit: 1km)
        norm_obs[1] = in_range_rad(delta_value[1])   #  1. ego delta heading   (unit rad)
        norm_obs[2] = delta_value[2] / 340           #  2. ego delta velocities_u  (unit: mh)
        norm_obs[3] = obs[9] / 5000                  #  3. ego_altitude (unit: km)
        norm_obs[4] = np.sin(obs[3])                 #  4. ego_roll_sin
        norm_obs[5] = np.cos(obs[3])                 #  5. ego_roll_cos
        norm_obs[6] = np.sin(obs[4])                 #  6. ego_pitch_sin
        norm_obs[7] = np.cos(obs[4])                 #  7. ego_pitch_cos
        norm_obs[8] = obs[5] / 340                   #  8. ego_v_x   (unit: mh)
        norm_obs[9] = obs[6] / 340                   #  9. ego_v_y    (unit: mh)
        norm_obs[10] = obs[7] / 340                  #  10. ego_v_z    (unit: mh)
        norm_obs[11] = obs[8] / 340                  #  11. ego_vc        (unit: mh)
        norm_obs = np.expand_dims(norm_obs, axis=0)  # dim: (1,12)
        return norm_obs

    def get_action(self, env, task):
        delta_value = self.set_delta_value(env, task)
        observation = self.get_observation(env, task, delta_value)
        _action, self.rnn_states = self.actor(observation, self.rnn_states)
        action = _action.detach().cpu().numpy().squeeze()
        return action

class ManeuverTestAgent(BaselineAgent):
    def __init__(self, agent_id,testType) -> None:
        super().__init__(agent_id)
        self.turn_interval = 30
        self.dodge_missile = False  # if set true, start turn when missile is detected
        self.type = testType
        if testType =='alt':
            self.target_heading_list = [0] *6
            self.target_altitude_list = [6000, 6500, 7000, 6500,6000,5500]
            self.target_velocity_list = [243] * 6
        elif testType =='heading':
            self.target_heading_list = [np.pi / 3, np.pi, -np.pi / 3] * 2
            self.target_altitude_list = [6000] * 6
            self.target_velocity_list = [243] * 6
        else :
            self.target_heading_list = [0] *6
            self.target_altitude_list = [20000*0.3048] *6
            self.target_velocity_list =  [203]*2+[153]*2+[153]*2


        self.target_headding = self.init_heading
        self.target_alt = self.target_altitude_list[0]
        self.target_velocity = self.target_velocity_list[0]


    def reset(self):
        self.step = 0
        self.rnn_states = np.zeros((1, 1, 128))
        self.init_heading = None
        self.target_headding = None
        self.target_alt = None
        self.target_velocity = None

    def set_delta_value(self, env, task):
        step_list = np.arange(1, len(self.target_heading_list) + 1) * self.turn_interval / env.time_interval
        uid = list(env.agents.keys())[self.agent_id]
        cur_heading = env.agents[uid].get_property_value(c.attitude_heading_true_rad)
        if self.init_heading is None:
            self.init_heading = cur_heading
            self.target_headding = cur_heading
            self.target_alt = self.target_altitude_list[0]
            self.target_velocity = self.target_velocity_list[0]
        if not self.dodge_missile or task._check_missile_warning(env, self.agent_id) is not None:
            for i, interval in enumerate(step_list):
                if self.step <= interval:
                    break
            delta_heading = self.init_heading + self.target_heading_list[i] - cur_heading
            delta_altitude = self.target_altitude_list[i] - env.agents[uid].get_property_value(c.position_h_sl_m)
            delta_velocity = self.target_velocity_list[i] - env.agents[uid].get_property_value(c.velocities_u_mps) # velocities_u_mps
            self.target_headding = self.init_heading + self.target_heading_list[i]
            self.target_alt = self.target_altitude_list[i]
            self.target_velocity = self.target_velocity_list[i]
            self.step += 1
        else:
            delta_heading = self.init_heading - cur_heading
            delta_altitude = 6000 - env.agents[uid].get_property_value(c.position_h_sl_m)
            delta_velocity = 143 - env.agents[uid].get_property_value(c.velocities_u_mps)
            print("ERROR!!")

        return np.array([delta_altitude, delta_heading, delta_velocity])

    def get_target_headding(self):
        return self.target_headding

    def get_target_alt(self):
        return self.target_alt

    def get_target_v(self):
        return self.target_velocity

class PursueAgent(BaselineAgent):
    def __init__(self, agent_id) -> None:
        super().__init__(agent_id)

    def set_delta_value(self, env, task):
        # NOTE: only adapt for 1v1
        ego_uid, enm_uid = list(env.agents.keys())[self.agent_id], list(env.agents.keys())[(self.agent_id+1)%2] 
        ego_x, ego_y, ego_z = env.agents[ego_uid].get_position()
        ego_vx, ego_vy, ego_vz = env.agents[ego_uid].get_velocity()
        enm_x, enm_y, enm_z = env.agents[enm_uid].get_position()
        # delta altitude
        delta_altitude = enm_z - ego_z
        # delta heading
        ego_v = np.linalg.norm([ego_vx, ego_vy])
        delta_x, delta_y = enm_x - ego_x, enm_y - ego_y
        R = np.linalg.norm([delta_x, delta_y])
        proj_dist = delta_x * ego_vx + delta_y * ego_vy
        ego_AO = np.arccos(np.clip(proj_dist / (R * ego_v + 1e-8), -1, 1))
        side_flag = np.sign(np.cross([ego_vx, ego_vy], [delta_x, delta_y]))
        delta_heading = ego_AO * side_flag
        # delta velocity
        delta_velocity = env.agents[enm_uid].get_property_value(c.velocities_u_mps) - \
                         env.agents[ego_uid].get_property_value(c.velocities_u_mps)
        return np.array([delta_altitude, delta_heading, delta_velocity])


class ManeuverAgent(BaselineAgent):
    def __init__(self, agent_id, maneuver: Literal['l', 'r', 'n']) -> None:
        super().__init__(agent_id)
        self.turn_interval = 30
        self.dodge_missile = False # if set true, start turn when missile is detected
        if maneuver == 'l':
            self.target_heading_list = [0]
        elif maneuver == 'r':
            self.target_heading_list = [np.pi/2, np.pi/2, np.pi/2, np.pi/2]
        elif maneuver == 'n':
            self.target_heading_list = [np.pi, np.pi, np.pi, np.pi]
        elif maneuver == 'triangle':
            self.target_heading_list = [np.pi/3, np.pi, -np.pi/3]*2
        self.target_altitude_list = [6000] * 6
        self.target_velocity_list = [243]  * 6
        self.target_headding=self.init_heading

    def reset(self):
        self.step = 0
        self.rnn_states = np.zeros((1, 1, 128))
        self.init_heading = None
        self.target_headding = None

    def set_delta_value(self, env, task):
        step_list = np.arange(1, len(self.target_heading_list)+1) * self.turn_interval / env.time_interval
        uid = list(env.agents.keys())[self.agent_id]
        cur_heading = env.agents[uid].get_property_value(c.attitude_heading_true_rad)
        if self.init_heading is None:
            self.init_heading = cur_heading
            self.target_headding= cur_heading
        if not self.dodge_missile or task._check_missile_warning(env, self.agent_id) is not None:
            for i, interval in enumerate(step_list):
                if self.step <= interval:
                    break
            delta_heading = self.init_heading + self.target_heading_list[i] - cur_heading
            self.target_headding=self.init_heading + self.target_heading_list[i]
            delta_altitude = self.target_altitude_list[i] - env.agents[uid].get_property_value(c.position_h_sl_m)
            delta_velocity = self.target_velocity_list[i] - env.agents[uid].get_property_value(c.velocities_u_mps)
            self.step += 1
        else:
            delta_heading = self.init_heading  - cur_heading
            delta_altitude = 6000 - env.agents[uid].get_property_value(c.position_h_sl_m)
            delta_velocity = 243 - env.agents[uid].get_property_value(c.velocities_u_mps)

        return np.array([delta_altitude, delta_heading, delta_velocity])


    def get_target_headding(self):
        return self.target_headding

def test_maneuver_heading():
    env = SingleCombatEnv(config_name='1v1/NoWeapon/test/opposite')
    obs = env.reset()
    env.render(filepath="control.txt.acmi")
    agent0 = ManeuverAgent(agent_id=0, maneuver='triangle') # 'A0100'    # 两种agent底层都使用网络， ManeuverAgent的驾驶是基于规则，每x秒切换。固定高度速度
    agent1 = PursueAgent(agent_id=1) # 'B0100'
    reward_list = []
    target_heading = []
    ego_heading = []

    while True:
        action0 = agent0.get_action(env, env.task)
        action1 = agent1.get_action(env, env.task)
        actions = [action0, action1]
        obs, reward, done, info = env.step(actions)
        env.render(filepath="control.txt.acmi")
        reward_list.append(reward[0])
        target_heading.append(env.agents['A0100'].get_property_value(c.attitude_heading_true_rad))
        ego_heading.append(agent0.get_target_headding())
        if np.array(done).all():
            print(info)
            break
    plt.plot(target_heading)
    plt.plot(ego_heading)
    plt.savefig('error_heading.png')


def test_maneuver_alt():
    env = SingleCombatEnv(config_name='1v1/NoWeapon/test/opposite')
    obs = env.reset()
    env.render(filepath="control.txt.acmi")
    agent0 = ManeuverAgent(agent_id=0,
                           maneuver='triangle')  # 'A0100'    # 两种agent底层都使用网络， ManeuverAgent的驾驶是基于规则，每x秒切换。固定高度速度
    agent1 = PursueAgent(agent_id=1)  # 'B0100'
    reward_list = []
    target_heading = []
    ego_heading = []
    while True:
        action0 = agent0.get_action(env, env.task)
        action1 = agent1.get_action(env, env.task)
        actions = [action0, action1]
        obs, reward, done, info = env.step(actions)
        env.render(filepath="control.txt.acmi")
        reward_list.append(reward[0])
        target_heading.append(env.agents['A0100'].get_property_value(c.attitude_heading_true_rad))
        ego_heading.append(agent0.get_target_headding())
        if np.array(done).all():
            print(info)
            break
    plt.plot(target_heading)
    plt.plot(ego_heading)
    plt.savefig('error_heading.png')


def draw_pid():
    path = 'D:\\HCH\\LAG\\pid\\v\\ego_value.txt'
    rewards = []

    # 读取文件中的数字并存入数组
    with open(path, 'r') as file:
        for line in file:
            try:
                # 转换每行内容为浮点数并添加到数组
                rewards.append(float(line.strip()))
            except ValueError:
                print(f"Skipping invalid line: {line.strip()}")
    plt.plot(rewards)


def test_maneuver(test_type):
    env = SingleCombatEnv(config_name='1v1/NoWeapon/test/opposite')
    obs = env.reset()
    env.render(filepath="control.txt.acmi")
    agent1 = PursueAgent(agent_id=1)  # 'B0100'
    agent0 = ManeuverTestAgent(agent_id=0,testType=test_type)
    target_value = []
    ego_value = []
    deta_value = []
    while True:
        action0 = agent0.get_action(env, env.task)
        action1 = agent1.get_action(env, env.task)
        actions = [action0, action1]
        obs, reward, done, info = env.step(actions)
        env.render(filepath="control.txt.acmi")
        if test_type == 'alt':
            ego_value.append(env.agents['A0100'].get_property_value(c.position_h_sl_ft )* 0.3048)
            target_value.append(agent0.get_target_alt())
        elif test_type == 'heading':
            ego_value.append(env.agents['A0100'].get_property_value(c.attitude_heading_true_rad))
            target_value.append(agent0.get_target_headding())
        else:
            ego_value.append(env.agents['A0100'].get_property_value(c.velocities_u_mps)) #velocities_u_mps OR velocities_vc_mps
            target_value.append(agent0.get_target_v())
            deta_value.append(env.agents['A0100'].get_property_value(c.delta_velocities_u)) #delta_velocities_u
        if np.array(done).all():
            print(info)
            break
    plt.plot(target_value)
    plt.plot(ego_value)
    #plt.plot(deta_value)
    draw_pid()
    plt.savefig("compare_v.png")

def draw_reward():
    path='D:\\HCH\\LAG\\log\\reward_list.txt'
    rewards = []

    # 读取文件中的数字并存入数组
    with open(path, 'r') as file:
        for line in file:
            try:
                # 转换每行内容为浮点数并添加到数组
                rewards.append(float(line.strip()))
            except ValueError:
                print(f"Skipping invalid line: {line.strip()}")
    plt.plot(rewards)
    plt.savefig("train_reward.png")


if __name__ == '__main__':
    # alt heading v 三种类型
    test_maneuver('v')
    #draw_reward()