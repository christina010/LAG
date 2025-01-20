import numpy as np
from wandb import agent
from .reward_function_base import BaseRewardFunction
from ..utils.utils import get_AO_TA_R


class PostureReward(BaseRewardFunction):
    """
    PostureReward 是一个姿态奖励函数,用于评估飞机的姿态和位置是否有利。总奖励由两部分组成:

    1. Orientation (朝向奖励):
    - 鼓励飞机朝向敌方 (AO角越小越好)
    其中:
    - AO (Angle-Off): 己方机头指向与敌方位置连线的夹角

    2. Range (距离奖励):
    - 鼓励与敌方保持适当距离(默认目标距离为3km)
    - 过远会受到惩罚
    - 不同版本的距离奖励函数有不同的衰减曲线

    最终奖励 = Orientation奖励 * Range奖励

    注意:
    - 仅支持1v1对战场景
    - 可以通过配置选择不同版本的orientation和range奖励函数
    """
    def __init__(self, config):
        super().__init__(config)
        # 从配置中读取使用哪个版本的朝向和距离奖励函数
        self.orientation_version = getattr(self.config, f'{self.__class__.__name__}_orientation_version', 'v2')
        self.range_version = getattr(self.config, f'{self.__class__.__name__}_range_version', 'v3')
        # 目标距离,默认3km
        self.target_dist = getattr(self.config, f'{self.__class__.__name__}_target_dist', 0.0)

        # 获取对应版本的奖励计算函数
        self.orientation_fn = self.get_orientation_function(self.orientation_version)
        self.range_fn = self.get_range_funtion(self.range_version)
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_orn', '_range']]

    def get_reward(self, task, env, agent_id):
        """
        计算姿态奖励。基于当前时刻的AO(Angle-Off)和R(Range)。

        参数:
            task: 任务实例
            env: 环境实例
            agent_id: 智能体ID

        返回:
            float: 计算得到的奖励值
        """
        new_reward = 0
        # 获取己方位置和速度信息: (north, east, down, vn, ve, vd)
        ego_feature = np.hstack([env.agents[agent_id].get_position(),
                                 env.agents[agent_id].get_velocity()])
        # 对每个敌方目标计算奖励
        for enm in env.agents[agent_id].enemies:
            # 获取敌方位置和速度信息
            enm_feature = np.hstack([enm.get_position(),
                                    enm.get_velocity()])
            # 计算AO、TA角度和距离
            AO, _, R = get_AO_TA_R(ego_feature, enm_feature)
            # 分别计算朝向奖励和距离奖励,并相乘
            orientation_reward = self.orientation_fn(AO)
            range_reward = self.range_fn(R / 1000)  # 转换为km
            new_reward += orientation_reward * range_reward
        return self._process(new_reward, agent_id, (orientation_reward, range_reward))

    def get_orientation_function(self, version):
        """
        获取不同版本的朝向奖励函数。
        各版本的函数形式略有不同,但都遵循:
        - AO角越小,奖励越大
        """
        if version == 'v0':
            return lambda AO: (1. - np.tanh(9 * (AO - np.pi / 9))) / 3. + 1 / 3. + 0.5
        elif version == 'v1':
            return lambda AO: (1. - np.tanh(2 * (AO - np.pi / 2))) / 2. + 0.5
        elif version == 'v2':
            return lambda AO: 1 / (50 * AO / np.pi + 2) + 1
        else:
            raise NotImplementedError(f"Unknown orientation function version: {version}")

    def get_range_funtion(self, version):
        """
        获取不同版本的距离奖励函数。
        各版本都遵循:
        - 距离接近目标距离时奖励最大
        - 距离过远时奖励衰减
        - 不同版本有不同的衰减曲线
        """
        if version == 'v0':
            return lambda R: np.exp(-(R - self.target_dist) ** 2 * 0.004) / (1. + np.exp(-(R - self.target_dist + 2) * 2))
        elif version == 'v1':
            return lambda R: np.clip(1.2 * np.min([np.exp(-(R - self.target_dist) * 0.21), 1]) /
                                     (1. + np.exp(-(R - self.target_dist + 1) * 0.8)), 0.3, 1)
        elif version == 'v2':
            return lambda R: max(np.clip(1.2 * np.min([np.exp(-(R - self.target_dist) * 0.21), 1]) /
                                         (1. + np.exp(-(R - self.target_dist + 1) * 0.8)), 0.3, 1), np.sign(7 - R))
        elif version == 'v3':
            return lambda R: 1 * (R < 5) + (R >= 5) * np.clip(-0.032 * R**2 + 0.284 * R + 0.38, 0, 1) + np.clip(np.exp(-0.16 * R), 0, 0.2)
        else:
            raise NotImplementedError(f"Unknown range function version: {version}")
