import math

import numpy as np
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c
from ..utils.utils import get_AO_TA_R, LLA2NEU, get_root_dir

class FormationReward(BaseRewardFunction):

    def __init__(self, config):
        super().__init__(config)
        self.safe_altitude = getattr(self.config, f'{self.__class__.__name__}_safe_altitude', 4.0)         # km
        self.danger_altitude = getattr(self.config, f'{self.__class__.__name__}_danger_altitude', 3.5)     # km
        self.Kv = getattr(self.config, f'{self.__class__.__name__}_Kv', 0.2)     # mh

        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_Pv', '_PH']]

        self.formation_offset = {
            "A0200": [50, 0, 0],
            "A0300": [-50, 0, 0]
        }
        self.leader_id='A0100'

        self.state_var=[
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

    def get_reward(self, task, env, agent_id):
        """
        Reward is the sum of all the punishments.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        leader_state = np.array(env.agents[self.leader_id].get_property_values(self.state_var))
        ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))
        cur_dis_vector = LLA2NEU(*leader_state[:3], *ego_state[:3])
        tar_dis_vector=self.follower_position(*leader_state[5],agent_id)
        error_vector=cur_dis_vector-tar_dis_vector
        reward=0
        for num in error_vector:
            reward += math.exp((-num/1)**2)
        reward = reward **(1/3)
        return self._process(reward, agent_id, None)


    ### 北东地 NED 东北天 ENU  目前使用ned
    def follower_position(self, leader_yaw, agent_id):
        cos_yaw = np.cos(leader_yaw)
        sin_yaw = np.sin(leader_yaw)
        R = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        offset_vector = np.array(self.formation_offset.get(agent_id))
        return np.dot(R, offset_vector)