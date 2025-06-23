from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
import math
import mujoco
import time
import numpy as np
from PIL import Image
from pathlib import Path

#nome branch reward_con_tempo
DEFAULT_CAMERA_CONFIG = {
    "azimuth": 90.0,
    "distance": 3.0,
    "elevation": -25.0,
    "lookat": np.array([0., 0., 0.]),
    "fixedcamid": 0,
    "trackbodyid": -1,
    "type": 2,
}


class Go1MujocoEnv(MujocoEnv):
    """Custom Environment that follows gym interface."""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(self, ctrl_type="torque",domain=None, point=None, **kwargs):
        model_path = Path(f"./unitree_go1/scene_{ctrl_type}.xml")
        MujocoEnv.__init__(
            self,
            model_path=model_path.absolute().as_posix(),
            frame_skip=10, 
            observation_space=None,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": 60,
        }
        self.environment = domain
        self.point_type = point
        self._distance_window = []
        self._distance_window_size = 100
        self.max_distance=math.sqrt(2)*10
        self._last_render_time = -1.0
        self._max_episode_time_sec = 100000
        self.start_episode=time.perf_counter()
        self._step = 0
        self.half_x, self.half_y = 40.0, 40.0      
        self.max_z = 7                     
        self.img = np.array(Image.open('unitree_go1/assets/bhutanlake.png'))
        self.nrow, self.ncol = self.img.shape
        self.direction= self.calc_direction()
        self._base_body_mass     = self.model.body_mass.copy()
        self._base_body_inertia  = self.model.body_inertia.copy()
        self._base_actuator_gear = self.model.actuator_gear[:, 0].copy()
        self._velocity_scale = np.array([2.0, 2.0, 0.25])
        if point=="fixed":
            self.reward_weights = {
                "progress":1.5,
                "orientation":2,
                "time_eff":1,
                "survival":1,
                "death":1,
                "progress_post":3,
                "orientation_post":2,
                "time_eff_post":2.5,
                "survival_post":1,
                "death_post":1}
        else:
            self.reward_weights = {
                "progress":3,
                "orientation":0.5,
                "time_eff":0.5,
                "survival":0.5,
                "death":2,
                "progress_post":5,
                "orientation_post":0.5,
                "time_eff_post":1.5,
                "survival_post":0.5,
                "death_post":2}
        # Metrics used to determine if the episode should be terminated
        self._healthy_z_range = (0.14, 0.65)
        self._healthy_pitch_range = (-np.deg2rad(15), np.deg2rad(15))
        self._healthy_roll_range = (-np.deg2rad(15), np.deg2rad(15))
        self._reset_noise_scale = 0
        # Action: 12 torque values
        self._last_action = np.zeros(12)
        self.objective_point=self.random_point()
        self.relative_direction=self.calc_relative_direction(self.direction)
        self.initial_distance_to_goal=np.linalg.norm(self.objective_point-np.array([0,0]))
        self.distance_to_goal=np.linalg.norm(self.objective_point-np.array([0,0]))
        self._clip_obs_threshold = 100.0
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._get_obs().shape[0],), dtype=np.float64
        )


        # Feet site names to index mapping
        # https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-site
        # https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjtobj
        feet_site = [
            "FR",
            "FL",
            "RR",
            "RL",
        ]
        self._feet_site_name_to_id = {
            f: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        }

        self._main_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY.value, "trunk"
        )

        

    def step(self, action):
        old_rel_direction=self.relative_direction
        old_position=np.array(self.state_vector()[0:2])
        self._step += 1
        self.do_simulation(action, self.frame_skip)
        self.distance_to_goal = np.linalg.norm(self.objective_point - self.data.qpos[0:2])
        self._distance_window.append(self.distance_to_goal)
        if len(self._distance_window) > self._distance_window_size:
            self._distance_window.pop(0)
        self.direction=self.calc_direction()
        self.relative_direction=self.calc_relative_direction(self.direction)
        observation = self._get_obs()
        reward, reward_info = self._calc_reward(old_position,old_rel_direction)
        terminated = not self.is_healthy[0] or self.reached
        if terminated:
            delta = self.initial_distance_to_goal - self.distance_to_goal
            #print(f"delta_dist: {delta:.4f} | start: {self.initial_distance_to_goal:.4f} | now: {self.distance_to_goal:.4f} | percentage: {delta/self.initial_distance_to_goal:.4f}")
        truncated = self._step >= (self._max_episode_time_sec / self.dt)
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            **reward_info,
        }

        if self.render_mode == "human" and (self.data.time - self._last_render_time) > (
            1.0 / self.metadata["render_fps"]
        ):
            self.render()
            self._last_render_time = self.data.time

        self._last_action = action
        

        return observation, reward, terminated, truncated, info

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        min_z=min_z+self.get_maps_z(state[0],state[1])+0.001
        is_healthy = np.isfinite(state).all() and min_z <= state[2] 
        min_roll, max_roll = self._healthy_roll_range
        is_healthy = is_healthy and min_roll <= state[4] <= max_roll
        min_pitch, max_pitch = self._healthy_pitch_range
        is_healthy = is_healthy and min_pitch <= state[5] <= max_pitch
        stillness = False
        if len(self._distance_window) == self._distance_window_size:
            progress = self._distance_window[0] - self._distance_window[-1]
            is_healthy  = is_healthy and (progress >0.2)
            if progress<0.2:
                stillness= True
                is_healthy = False
        return is_healthy, stillness


    @property
    def reached(self):
        reached=np.linalg.norm(self.objective_point-self.data.qpos[0:2], ord=2)<0.5
        return reached


    



    def state_vector(self):
        base = super().state_vector()
        return np.concatenate([base, np.array([self.relative_direction]), np.array([self.distance_to_goal])])


    def random_point(self):
        if self.point_type ==  "random":
            x = np.random.uniform(-self.half_x/8, self.half_x/8)
            y = np.random.uniform(-self.half_y/8, self.half_y/8)
            return [x,y]  
        else:
            return [10,10] 

    def _calc_reward(self, old_position, old_rel_direction):
        objective=np.array(self.objective_point)
        old_distance= np.linalg.norm(objective - old_position)
        new_distance=self.distance_to_goal
        progress=old_distance-new_distance
        orientation_reward=np.cos(self.relative_direction)+(np.cos(self.relative_direction)-np.cos(old_rel_direction))
        time_eff=self.calc_vel_objective()
        survival = 0.1 if self.is_healthy else 0.0
        death_penalty = -10.0 if not self.is_healthy[0] else 0.0
        if abs(self.relative_direction) > 0.2:
            reward = self.reward_weights["progress"] * progress + self.reward_weights["orientation"] * orientation_reward + self.reward_weights["time_eff"]*time_eff + self.reward_weights["survival"]*survival + self.reward_weights["death"]*death_penalty
        else:
            reward = self.reward_weights["progress_post"] * progress + self.reward_weights["orientation_post"] * orientation_reward + self.reward_weights["time_eff_post"]*time_eff + self.reward_weights["survival_post"]*survival + self.reward_weights["death_post"]*death_penalty
            reward += self.reward_joint_motion()
            reward += 0.001 * np.linalg.norm(self.data.qvel)
        reward = reward + 100*self.reached
        reward = reward -200*self.is_healthy[1]
         
        reward_info = {
                    "progress": progress,
                    "orientation_reward": orientation_reward,
                    "reward_survive": survival,
                    "time_eff": time_eff
        }
        return reward, reward_info


    def _get_obs(self):
        position = self.data.qpos[7:].flatten() - self.model.key_qpos[0, 7:]
        velocity = self.data.qvel.flatten()
        velocity[:3] *= self._velocity_scale
        last_action = self._last_action
        curr_obs = np.concatenate((position, velocity, last_action)).clip(
            -self._clip_obs_threshold, self._clip_obs_threshold
        )
        terrain_window = self.sample_terrain_ahead(window_size=(9,9)).flatten()
        curr_obs= np.concatenate([curr_obs, np.array([self.relative_direction]), np.array([self.distance_to_goal/self.max_distance]), terrain_window])
        return curr_obs
        
    def sample_terrain_ahead(self, window_size):
        robot_pos = self.data.qpos[:3]
        heading = self.calc_direction()
        side = np.array([-heading[1], heading[0]])
    
        points = []
        for dx in np.linspace(0.1, 1.0, window_size[0]):
            for dy in np.linspace(-0.3, 0.3, window_size[1]):
                offset = dx * heading + dy * side
                sample_x = robot_pos[0] + offset[0]
                sample_y = robot_pos[1] + offset[1]
                height = self.get_maps_z(sample_x, sample_y)
                rel_height = height - robot_pos[2]
                points.append(rel_height)
    
        return np.array(points).reshape(window_size)

    def reward_joint_motion(self):
        joint_vels = self.data.qvel[6:18]
        forward_joint_ids = [1, 2, 4, 5, 11]
        motion_reward = np.sum(np.abs(joint_vels[forward_joint_ids]))
        return 0.01 * motion_reward


    
    def reset_model(self):
        if self.environment=="source":
            self.restore_physical_props()
            self.random_mass()
        self.data.qpos[:] = self.model.key_qpos[0] + self.np_random.uniform(
            low=-self._reset_noise_scale,
            high=self._reset_noise_scale,
            size=self.model.nq,
        )
                
        self.data.ctrl[:] = self.model.key_ctrl[
            0
        ] + self._reset_noise_scale * self.np_random.standard_normal(
            *self.data.ctrl.shape
        )
        self.objective_point = self.random_point() 
        self.initial_distance_to_goal=np.linalg.norm(self.objective_point-np.array([0,0]))
        self.distance_to_goal=np.linalg.norm(self.objective_point-np.array([0,0]))
        self.direction= self.calc_direction()
        self.relative_direction=self.calc_relative_direction(self.direction)
        x, y = self.objective_point
        z=self.get_maps_z(x,y)
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker_body")
        self.model.body_pos[body_id] = [x, y, z+3]
        self._step = 0
        self._last_action = np.zeros(12)
        self._last_render_time = -1.0

        observation = self._get_obs()
        self.start_episode=time.perf_counter()
        self._distance_window=[]
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }

    def get_maps_z(self,X,Y):
        u = (X + self.half_x) / (2*self.half_x)
        v = (Y + self.half_y) / (2*self.half_y)
        j = int(round(u * (self.ncol - 1)))
        i = int(round(v * (self.nrow - 1)))
        i = max(0, min(self.nrow-1, i))
        j = max(0, min(self.ncol-1, j))
        p = self.img[i, j]
        h_rel = p / 65535.0
        return h_rel * self.max_z
    def calc_direction(self): 
        mat = np.zeros(9, dtype=np.float64)
        quat = self.data.qpos[3:7].astype(np.float64)
        mujoco.mju_quat2Mat(mat, quat)
        x_axis_world = np.array([mat[0], mat[3], mat[6]])   
        direction_xy = x_axis_world[:2]
        norm=np.linalg.norm(direction_xy)
        if norm<1e-8:
            return np.array([1.0, 0.0])
        direction_xy /= norm

        return direction_xy
    def calc_relative_direction(self,direction):
        vec_to_target = self.objective_point - self.data.qpos[0:2]
        norm = np.linalg.norm(vec_to_target)
        if norm < 1e-8:
            return 0.0  
        vec_to_target /= norm
        dot_val = float(np.dot(direction, vec_to_target))
        cross_z = direction[0] * vec_to_target[1] - direction[1] * vec_to_target[0]
        theta = math.atan2(cross_z, dot_val)
        return theta
    def calc_vel_objective(self):
        goal_vec = self.objective_point - self.data.qpos[:2]        
        goal_dir = goal_vec / (np.linalg.norm(goal_vec) + 1e-8)
        vel_xy = self.data.qvel[:2]
        v_toward_goal = np.dot(vel_xy, goal_dir)
        return v_toward_goal
    def random_mass(self):
        factor=np.random.uniform(0.8,1.2)
        self.model.body_mass[:]    *= factor
        self.model.body_inertia[:] *= factor ** (5/3)
        self.model.actuator_gear[:, 0] *= factor
        mujoco.mj_forward(self.model, self.data)
    def restore_physical_props(self):
        self.model.body_mass[:]        = self._base_body_mass
        self.model.body_inertia[:]     = self._base_body_inertia
        self.model.actuator_gear[:, 0] = self._base_actuator_gear
        mujoco.mj_forward(self.model, self.data)