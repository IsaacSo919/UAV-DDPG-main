import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class UAVEnv(gym.Env):
    """
    UAV Mobile Edge Computing Environment.
    A drone serves as a flying MEC server for ground users,
    optimizing the trade-off between energy consumption and latency.
    """
    metadata = {"render_modes": ["human"]}

    # ─── Physical Constants (class-level, no 'self') ─────────
    height = 100
    ground_length = 100
    ground_width = 100
    bandwidth_nums = 1
    B = bandwidth_nums * 10 ** 6
    p_noisy_los = 10 ** (-13)
    p_noisy_nlos = 10 ** (-11)
    flight_speed = 50.0
    f_ue = 2e8
    f_uav = 1.2e9
    r = 10 ** (-27)
    s = 1000
    p_uplink = 0.1
    alpha0 = 1e-5
    T = 320
    t_fly = 1
    t_com = 7
    delta_t = t_fly + t_com
    v_ue = 1
    slot_num = int(T / delta_t)  # 40
    m_uav = 9.65
    M = 4

    action_bound = [-1, 1]
    action_dim = 4
    state_dim = 4 + M * 4  # 20

    def __init__(self):
        super().__init__()

        # ─── Gymnasium spaces (MUST be inside __init__) ──────
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim,), dtype=np.float32
        )

        # ─── Initialize mutable state ───────────────────────
        self.sum_task_size = 100 * 1048576
        self.e_battery_uav = 500000
        self.loc_uav = [50, 50]
        self.loc_ue_list = np.random.randint(0, 101, size=[self.M, 2])
        self.task_list = np.random.randint(2097153, 2621440, self.M)
        self.block_flag_list = np.random.randint(0, 2, self.M)
        self.state = self._build_state()

    def _build_state(self):
        """Construct the state vector from current attributes."""
        state = np.append(self.e_battery_uav, self.loc_uav)
        state = np.append(state, self.sum_task_size)
        state = np.append(state, np.ravel(self.loc_ue_list))
        state = np.append(state, self.task_list)
        state = np.append(state, self.block_flag_list)
        return state.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sum_task_size = 100 * 1048576
        self.e_battery_uav = 500000
        self.loc_uav = [50, 50]
        self.loc_ue_list = np.random.randint(0, 101, size=[self.M, 2])
        self._reset_step()
        self.state = self._build_state()
        return self.state, {}  # Gymnasium requires (obs, info)

    def _reset_step(self):
        self.task_list = np.random.randint(2621440, 3145729, self.M)
        self.block_flag_list = np.random.randint(0, 2, self.M)

    def _get_obs(self):
        self.state = self._build_state()
        return self.state

    def step(self, action):
        step_redo = False
        is_terminal = False
        offloading_ratio_change = False
        reset_dist = False

        # Map [-1,1] → [0,1]
        action = (action + 1) / 2

        comm_energy = self.compute_comm_energy(action)
        comp_energy = self.compute_comp_energy(action)
        prop_energy = self.compute_prop_energy(action)

        if action[0] == 1:
            ue_id = self.M - 1
        else:
            ue_id = int(self.M * action[0])

        theta = action[1] * np.pi * 2
        offloading_ratio = action[3]
        task_size = self.task_list[ue_id]
        block_flag = self.block_flag_list[ue_id]

        dis_fly = action[2] * self.flight_speed * self.t_fly
        e_fly = (dis_fly / self.t_fly) ** 2 * self.m_uav * self.t_fly * 0.5

        dx_uav = dis_fly * math.cos(theta)
        dy_uav = dis_fly * math.sin(theta)
        loc_uav_after_fly_x = self.loc_uav[0] + dx_uav
        loc_uav_after_fly_y = self.loc_uav[1] + dy_uav

        t_server = offloading_ratio * task_size / (self.f_uav / self.s)
        e_server = self.r * self.f_uav ** 3 * t_server

        if self.sum_task_size == 0:
            is_terminal = True
            reward = 0
        elif self.sum_task_size - self.task_list[ue_id] < 0:
            self.task_list = np.ones(self.M) * self.sum_task_size
            reward = 0
            step_redo = True
        elif (loc_uav_after_fly_x < 0 or loc_uav_after_fly_x > self.ground_width
              or loc_uav_after_fly_y < 0 or loc_uav_after_fly_y > self.ground_length):
            reset_dist = True
            delay = self.com_delay(self.loc_ue_list[ue_id], self.loc_uav,
                                   offloading_ratio, task_size, block_flag)
            reward = -delay
            self.e_battery_uav = self.e_battery_uav - e_server
            self._reset2(delay, self.loc_uav[0], self.loc_uav[1],
                         offloading_ratio, task_size, ue_id)
        elif self.e_battery_uav < e_fly or self.e_battery_uav - e_fly < e_server:
            delay = self.com_delay(
                self.loc_ue_list[ue_id],
                np.array([loc_uav_after_fly_x, loc_uav_after_fly_y]),
                0, task_size, block_flag)
            reward = -delay
            self._reset2(delay, loc_uav_after_fly_x, loc_uav_after_fly_y,
                         0, task_size, ue_id)
            offloading_ratio_change = True
        else:
            delay = self.com_delay(
                self.loc_ue_list[ue_id],
                np.array([loc_uav_after_fly_x, loc_uav_after_fly_y]),
                offloading_ratio, task_size, block_flag)
            reward = -delay
            self.e_battery_uav = self.e_battery_uav - e_fly - e_server
            self.loc_uav[0] = loc_uav_after_fly_x
            self.loc_uav[1] = loc_uav_after_fly_y
            self._reset2(delay, loc_uav_after_fly_x, loc_uav_after_fly_y,
                         offloading_ratio, task_size, ue_id)

        latency = self.com_delay(
            self.loc_ue_list[ue_id],
            np.array([loc_uav_after_fly_x, loc_uav_after_fly_y]),
            offloading_ratio, task_size, block_flag)

        return (self._get_obs(), comm_energy, comp_energy, prop_energy,
                latency, is_terminal, step_redo, offloading_ratio_change,
                reset_dist)

    def _reset2(self, delay, x, y, offloading_ratio, task_size, ue_id):
        self.sum_task_size -= self.task_list[ue_id]
        for i in range(self.M):
            tmp = np.random.rand(2)
            theta_ue = tmp[0] * np.pi * 2
            dis_ue = tmp[1] * self.delta_t * self.v_ue
            self.loc_ue_list[i][0] = self.loc_ue_list[i][0] + math.cos(theta_ue) * dis_ue
            self.loc_ue_list[i][1] = self.loc_ue_list[i][1] + math.sin(theta_ue) * dis_ue
            self.loc_ue_list[i] = np.clip(self.loc_ue_list[i], 0, self.ground_width)
        self._reset_step()

        with open('output.txt', 'a') as f:
            f.write(f"\nUE-{ue_id}, task size: {int(task_size)}, "
                    f"offloading ratio:{offloading_ratio:.2f}")
            f.write(f"\ndelay:{delay:.2f}")
            f.write(f"\nUAV hover loc:[{x:.2f}, {y:.2f}]")

    def com_delay(self, loc_ue, loc_uav, offloading_ratio, task_size, block_flag):
        dx = loc_uav[0] - loc_ue[0]
        dy = loc_uav[1] - loc_ue[1]
        dh = self.height
        dist_uav_ue = np.sqrt(dx * dx + dy * dy + dh * dh)
        p_noise = self.p_noisy_nlos if block_flag == 1 else self.p_noisy_los
        g_uav_ue = abs(self.alpha0 / dist_uav_ue ** 2)
        trans_rate = self.B * math.log2(1 + self.p_uplink * g_uav_ue / p_noise)
        t_tr = offloading_ratio * task_size / trans_rate
        t_edge_com = offloading_ratio * task_size / (self.f_uav / self.s)
        t_local_com = (1 - offloading_ratio) * task_size / (self.f_ue / self.s)
        if t_tr < 0 or t_edge_com < 0 or t_local_com < 0:
            raise ValueError("Negative delay computed!")
        return max(t_tr + t_edge_com, t_local_com)

    def compute_reward(self, comm_energy, comp_energy, prop_energy, latency):
        alpha = 1.0
        beta = 1.0
        total_energy = (comm_energy + comp_energy + prop_energy) / 60000
        total_latency = latency / 7.5
        return -(alpha * total_energy + beta * total_latency)

    def compute_comm_energy(self, action):
        return 0.5 * 0.1 * 1  # 0.05 J

    def compute_comp_energy(self, action):
        kappa = 1e-27
        return kappa * 1000 * 2e6 * (1.2e9 ** 2)

    def compute_prop_energy(self, action):
        gamma1, gamma2 = 0.5, 0.1
        v, a, g = 50, 2, 9.81
        return (gamma1 * v**3) + (gamma2 / v) * (1 + (a**2) / (g**2))