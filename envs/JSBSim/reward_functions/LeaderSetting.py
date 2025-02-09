import math

import numpy as np
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c
from ..utils.utils import get_AO_TA_R, LLA2NEU, get_root_dir


class CollisionReward(BaseRewardFunction):
    """
    CollisionReward
    Achieve reward when the following event happens:
    - Shot down by missile: -200
    - Crash accidentally: -200
    - Shoot down other aircraft: +200
    """
    def __init__(self, config):
        super().__init__(config)
        self.state_var = [
            c.position_long_gc_deg,  # 0. lontitude  (unit: °)
            c.position_lat_geod_deg,  # 1. latitude   (unit: °)
            c.position_h_sl_m,  # 2. altitude   (unit: m)
            c.attitude_roll_rad,  # 3. roll       (unit: rad)
            c.attitude_pitch_rad,  # 4. pitch      (unit: rad)
            c.attitude_heading_true_rad,  # 5. yaw        (unit: rad)
            c.velocities_v_north_mps,  # 6. v_north    (unit: m/s)
            c.velocities_v_east_mps,  # 7. v_east     (unit: m/s)
            c.velocities_v_down_mps,  # 8. v_down     (unit: m/s)
            c.velocities_u_mps,  # 9. v_body_x   (unit: m/s)
            c.velocities_v_mps,  # 10. v_body_y  (unit: m/s)
            c.velocities_w_mps,  # 11. v_body_z  (unit: m/s)
            c.velocities_vc_mps,  # 12. vc        (unit: m/s)
            c.accelerations_n_pilot_x_norm,  # 13. a_north   (unit: G)
            c.accelerations_n_pilot_y_norm,  # 14. a_east    (unit: G)
            c.accelerations_n_pilot_z_norm,  # 15. a_down    (unit: G)
        ]

    def get_reward(self, task, env, agent_id):
        """
        Reward is the sum of all the events.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        reward = 0
        if env.agents[agent_id].is_shotdown:
            reward -= 200
        elif env.agents[agent_id].is_crash:
            reward -= 200
        for missile in env.agents[agent_id].launch_missiles:
            if missile.is_success:
                reward += 200
        ego_state = np.array(env.agents[agent_id].get_property_values(self.state_var))

        for aircraft in env.agents[agent_id].partners:
            air_state = np.array(aircraft.get_property_values(self.state_var))
            dis=  LLA2NEU(*air_state[:3], *ego_state[:3])
            for d in dis:
                if d<15:
                    reward -= 200
                    env.agents[agent_id].crash()
        return self._process(reward, agent_id)
