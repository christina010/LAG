import numpy as np
from gym import spaces
from typing import Tuple
import torch
from .task_base import BaseTask
from ..core.catalog import Catalog as c
from ..core.simulatior import MissileSimulator
from ..reward_functions import AltitudeReward, PostureReward, EventDrivenReward, MissilePostureReward
from ..reward_functions.collision_reward import CollisionReward
from ..reward_functions.formation_reward import FormationReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, SafeReturn
from ..utils.utils import get_AO_TA_R, LLA2NEU, get_root_dir
from ..model.baseline_actor import BaselineActor

class FormationTask(BaseTask):

    def __init__(self, config: str):
        super().__init__(config)
        self.lowlevel_policy = BaselineActor()
        self.lowlevel_policy.load_state_dict(
            torch.load(get_root_dir() + '/model/baseline_model.pt', map_location=torch.device('cuda:0')))
        self.lowlevel_policy.eval()
        self.norm_delta_altitude = np.array([0.1, 0, -0.1])
        self.norm_delta_heading = np.array([-np.pi / 6, -np.pi / 12, 0, np.pi / 12, np.pi / 6])
        self.norm_delta_velocity = np.array([0.05, 0, -0.05])
        # TODO: 1. reward function 2. 限制范围
        self.leader_id='A0100'
        self.reward_functions = [
            CollisionReward(self.config),
            FormationReward(self.config),
            ## 怎么控制leader飞行，leader的和follower隔离，只有follower是多智能体。

            LeaderSetting(self.config),
        ]

        self.termination_conditions = [
            SafeReturn(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            Timeout(self.config),
        ]

        self.formation_offset= {
        "A0200": [50, 0, 0],
        "A0300": [-50, 0, 0]
        }

    @property
    def num_agents(self) -> int:
        return 3

    def load_variables(self):
        self.state_var = [
            c.position_long_gc_deg,             # 0. lontitude  (unit: °)
            c.position_lat_geod_deg,            # 1. latitude   (unit: °)
            c.position_h_sl_m,                  # 2. altitude   (unit: m)
            c.attitude_roll_rad,                # 3. roll       (unit: rad)
            c.attitude_pitch_rad,               # 4. pitch      (unit: rad)
            c.attitude_heading_true_rad,        # 5. yaw        (unit: rad)
            c.velocities_v_north_mps,           # 6. v_north    (unit: m/s)
            c.velocities_v_east_mps,            # 7. v_east     (unit: m/s)
            c.velocities_v_down_mps,            # 8. v_down     (unit: m/s)
            c.velocities_u_mps,                 # 9. v_body_x   (unit: m/s)
            c.velocities_v_mps,                 # 10. v_body_y  (unit: m/s)
            c.velocities_w_mps,                 # 11. v_body_z  (unit: m/s)
            c.velocities_vc_mps,                # 12. vc        (unit: m/s)
            c.accelerations_n_pilot_x_norm,     # 13. a_north   (unit: G)
            c.accelerations_n_pilot_y_norm,     # 14. a_east    (unit: G)
            c.accelerations_n_pilot_z_norm,     # 15. a_down    (unit: G)
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,             # [-1., 1.]
            c.fcs_elevator_cmd_norm,            # [-1., 1.]
            c.fcs_rudder_cmd_norm,              # [-1., 1.]
            c.fcs_throttle_cmd_norm,            # [0.4, 0.9]
        ]
        self.render_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
        ]

    def load_action_space(self):
        self.action_space = spaces.MultiDiscrete([3, 5, 3])

    def load_observation_space(self):
        self.obs_length = 20 + (self.num_agents-1) * 11
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=(self.num_agents * self.obs_length,))

    def get_obs(self, env, agent_id):
        leader_state =  np.array(env.agents[self.leader_id].get_property_values(self.state_var))
        norm_obs = np.zeros(self.obs_length)
        ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
        # leader-follower neu
        dis_vector=LLA2NEU(*leader_state[:3],*ego_state[:3])
        # (1) leader state
        dis= np.linalg.norm(dis_vector)
        norm_obs[0] = dis /5000          # 0. leader_ego distance
        norm_obs[1] = np.sin(leader_state[3])  # 1. ego_roll_sin
        norm_obs[2] = np.cos(leader_state[3])  # 2. ego_roll_cos
        norm_obs[3] = np.sin(leader_state[4])  # 3. ego_pitch_sin
        norm_obs[4] = np.cos(leader_state[4])  # 4. ego_pitch_cos
        norm_obs[5] = leader_state[9] / 340  # 5. ego v_body_x   (unit: mh)
        norm_obs[6] = leader_state[10] / 340  # 6. ego v_body_y   (unit: mh)
        norm_obs[7] = leader_state[11] / 340  # 7. ego v_body_z   (unit: mh)
        norm_obs[8] = leader_state[12] / 340  # 8. ego vc   (unit: mh)(unit: 5G)
        norm_obs[9] = dis_vector[0] /5000  # 9. leader_ego north
        norm_obs[10] = dis_vector[1] /5000 # 10. leader_ego east
        norm_obs[11] = dis_vector[2] /5000 # 11. leader_ego up
        #(2)ego state
        norm_obs[12] = np.sin(ego_state[3])  # 12. ego_roll_sin
        norm_obs[13] = np.cos(ego_state[3])  # 13. ego_roll_cos
        norm_obs[14] = np.sin(ego_state[4])  # 14. ego_pitch_sin
        norm_obs[15] = np.cos(ego_state[4])  # 15. ego_pitch_cos
        norm_obs[16] = ego_state[9] / 340  # 16. ego v_body_x   (unit: mh)
        norm_obs[17] = ego_state[10] / 340  # 17. ego v_body_y   (unit: mh)
        norm_obs[18] = ego_state[11] / 340  # 18. ego v_body_z   (unit: mh)
        norm_obs[19] = ego_state[12] / 340  # 19. ego vc   (unit: mh)(unit: 5G)
        # (3) relative inof w.r.t partner 位置，速度
        offset = 19
        for sim in env.agents[agent_id].partners :
            state = np.array(sim.get_property_values(self.state_var))
            cur_ned = LLA2NEU(*state[:3], *ego_state[:3])
            # other follower state
            norm_obs[offset + 1] = cur_ned[0] / 5000
            norm_obs[offset + 2] = cur_ned[1] / 5000
            norm_obs[offset + 3] = cur_ned[2] / 5000
            norm_obs[offset + 4] = np.sin(state[3])  # 12. ego_roll_sin
            norm_obs[offset + 5] = np.cos(state[3])  # 13. ego_roll_cos
            norm_obs[offset + 6] = np.sin(state[4])  # 14. ego_pitch_sin
            norm_obs[offset + 7] = np.cos(state[4])  # 15. ego_pitch_cos
            norm_obs[offset + 8] = state[9] / 340  # 16. ego v_body_x   (unit: mh)
            norm_obs[offset + 9] = state[10] / 340  # 17. ego v_body_y   (unit: mh)
            norm_obs[offset + 10] = state[11] / 340  # 18. ego v_body_z   (unit: mh)
            norm_obs[offset + 11] = state[12] / 340  # 19. ego vc   (unit: mh)(unit: 5G)
            offset += 11

        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        """Convert high-level action into low-level action.
        """
        # generate low-level input_obs
        raw_obs = self.get_obs(env, agent_id)
        input_obs = np.zeros(12)
        # (1) delta altitude/heading/velocity
        input_obs[0] = self.norm_delta_altitude[action[0]]
        input_obs[1] = self.norm_delta_heading[action[1]]
        input_obs[2] = self.norm_delta_velocity[action[2]]
        # (1) ego info
        input_obs[3:12] = raw_obs[:9]
        input_obs = np.expand_dims(input_obs, axis=0)
        # output low-level action
        _action, _rnn_states = self.lowlevel_policy(input_obs, self._inner_rnn_states[agent_id])
        action = _action.detach().cpu().numpy().squeeze(0)
        self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()
        # normalize low-level action
        norm_act = np.zeros(4)
        norm_act[0] = action[0] / 20 - 1.
        norm_act[1] = action[1] / 20 - 1.
        norm_act[2] = action[2] / 20 - 1.
        norm_act[3] = action[3] / 58 + 0.4
        return norm_act

    def reset(self, env):
        """Task-specific reset, include reward function reset.
        """
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        return super().reset(env)
        # TODO 确定通过super调用reset在formation场景中的异同

    # def step(self, env):
    #     # 空战中step用来计算血量和发射导弹。编队不需要
    #     pass


    ### 北东地 NED 东北天 ENU  目前使用ned
    def follower_position(self,leader_pos, leader_yaw,agent_id):
        cos_yaw = np.cos(leader_yaw)
        sin_yaw = np.sin(leader_yaw)
        R = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        offset_vector = np.array(self.formation_offset.get(agent_id))
        follower_pos = leader_pos + np.dot(R, offset_vector)
        return follower_pos



