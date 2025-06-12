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

    def __init__(self, ctrl_type="torque", **kwargs):
        model_path = Path(f"./unitree_go1/scene_{ctrl_type}.xml")
        MujocoEnv.__init__(
            self,
            model_path=model_path.absolute().as_posix(),
            frame_skip=10,  # Perform an action every 10 frames (dt(=0.002) * 10 = 0.02 seconds -> 50hz action rate)
            observation_space=None,  # Manually set afterwards
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        # Update metadata to include the render FPS
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": 60,
        }
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
        


        # Weights for the reward and cost functions
        self.reward_weights = {
            "linear_vel_tracking": 2.0,  # Was 1.0
            "angular_vel_tracking": 0.5,
            "healthy": 0.0,  # was 0.05
            "feet_airtime": 1.0,
        }
        self.cost_weights = {
            "torque": 0.0002,
            "vertical_vel": 0.5,  # Was 1.0
            "xy_angular_vel": 0.02,  # Was 0.05
            "action_rate": 0.01,
            "joint_limit": 10.0,
        }

        # vx (m/s), vy (m/s), wz (rad/s)
        self._desired_velocity_min = np.array([-0.5, -0.6, -0.6])
        self._desired_velocity_max = np.array([1.5, 0.6, 0.6])
        self._desired_velocity = self._sample_desired_vel()
        self._velocity_scale = np.array([2.0, 2.0, 0.25])
        self._tracking_velocity_sigma = 0.25

        # Metrics used to determine if the episode should be terminated
        self._healthy_z_range = (0.19, 0.65)
        self._healthy_pitch_range = (-np.deg2rad(15), np.deg2rad(15))
        self._healthy_roll_range = (-np.deg2rad(15), np.deg2rad(15))

        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4)
        self._cfrc_ext_feet_indices = [4, 7, 10, 13]  # 4:FR, 7:FL, 10:RR, 13:RL

        # Non-penalized degrees of freedom range of the control joints
        dof_position_limit_multiplier = 0.9  # The % of the range that is not penalized
        ctrl_range_offset = (
            0.5
            * (1 - dof_position_limit_multiplier)
            * (
                self.model.actuator_ctrlrange[:, 1]
                - self.model.actuator_ctrlrange[:, 0]
            )
        )
        # First value is the root joint, so we ignore it
        self._soft_joint_range = np.copy(self.model.jnt_range[1:])
        self._soft_joint_range[:, 0] += ctrl_range_offset
        self._soft_joint_range[:, 1] -= ctrl_range_offset

        self._reset_noise_scale = 0

        # Action: 12 torque values
        self._last_action = np.zeros(12)
        self.objective_point=self.random_point()
        self.relative_direction=self.calc_relative_direction(self.direction)
        self.distance=np.linalg.norm(self.objective_point-np.array([0,0]))
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
        now=time.perf_counter()
        self.direction=self.calc_direction()
        self.relative_direction=self.calc_relative_direction(self.direction)
        reached = self.reached
        if reached:
            print("OOOOOOOOOOOOOOOOO")
        time_diff=now-self.start_episode
        observation = self._get_obs()
        reward, reward_info = self._calc_reward(action,old_position,time_diff,old_rel_direction)
        # TODO: Consider terminating if knees touch the ground
        terminated = not self.is_healthy[0] or self.reached
        #if terminated and distance_to_goal>1:
            #reward=reward-100

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
        #print(">>> Dopo un passo:", " pos (x,y):", self.data.qpos[0:2], " z:", self.data.qpos[2], " is_healthy:", self.is_healthy, " reached:", self.reached, " terminated:", terminated)

        return observation, reward, terminated, truncated, info

    
    def calc_leg_spread_penalty(self):
        qpos_joints = self.data.qpos[7:]
        penalty = 0.0
    
        # Assume 12 joints: 3 per leg (hip_abduction, hip_rotation, knee)
        # Hip abduction joints are at indices: 0, 3, 6, 9 (one per leg)
        abduction_indices = [0, 3, 6, 9]
    
        for i in abduction_indices:
            q = qpos_joints[i]
            # Penalize if too far from 0 (neutral straight position)
            penalty += q ** 2  # squared to penalize larger deviations more
    
        return -0.1 * penalty  # negative to penalize in reward

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
    
    @property
    def feet_contact_forces(self):
        feet_contact_forces = self.data.cfrc_ext[self._cfrc_ext_feet_indices]
        return np.linalg.norm(feet_contact_forces, axis=1)

    ######### Positive Reward functions #########
    @property
    def linear_velocity_tracking_reward(self):
        vel_sqr_error = np.sum(
            np.square(self._desired_velocity[:2] - self.data.qvel[:2])
        )
        return np.exp(-vel_sqr_error / self._tracking_velocity_sigma)

    @property
    def angular_velocity_tracking_reward(self):
        vel_sqr_error = (self._desired_velocity[2] - self.data.qvel[5]) ** 2
        return np.exp(-vel_sqr_error / self._tracking_velocity_sigma)

    @property
    def heading_tracking_reward(self):
        # TODO: qpos[3:7] are the quaternion values
        pass

    @property
    def feet_air_time_reward(self):
        """Award strides depending on their duration only when the feet makes contact with the ground"""
        feet_contact_force_mag = self.feet_contact_forces
        curr_contact = feet_contact_force_mag > 0.1
        contact_filter = np.logical_or(curr_contact, self._last_contacts)
        self._last_contacts = curr_contact

        # if feet_air_time is > 0 (feet was in the air) and contact_filter detects a contact with the ground
        # then it is the first contact of this stride
        first_contact = (self._feet_air_time > 0.0) * contact_filter
        self._feet_air_time += self.dt

        # Award the feets that have just finished their stride (first step with contact)
        air_time_reward = np.sum((self._feet_air_time - 0.5) * first_contact)
        # No award if the desired velocity is very low (i.e. robot should remain stationary and feet shouldn't move)
        air_time_reward *= np.linalg.norm(self._desired_velocity[:2]) > 0.1

        # zero-out the air time for the feet that have just made contact (i.e. contact_filter==1)
        self._feet_air_time *= ~contact_filter

        return air_time_reward
    
    @property
    def healthy_reward(self):
        return self.is_healthy

    ######### Negative Reward functions #########
    @property  # TODO: Not used
    def feet_contact_forces_cost(self):
        return np.sum(
            (self.feet_contact_forces - self._max_contact_force).clip(min=0.0)
        )

    @property  # TODO: Not used. Values are also quaternion!
    def non_flat_base_cost(self):
        # Penalize the robot for not being flat on the ground
        return np.sum(np.square(self.data.qpos[4:6]))

    @property
    def joint_limit_cost(self):
        # Penalize the robot for joints exceeding the soft control range
        out_of_range = (self._soft_joint_range[:, 0] - self.data.qpos[7:]).clip(
            min=0.0
        ) + (self.data.qpos[7:] - self._soft_joint_range[:, 1]).clip(min=0.0)
        return np.sum(out_of_range)

    @property
    def torque_cost(self):
        # Last 12 values are the motor torques
        return np.sum(np.square(self.data.qfrc_actuator[-12:]))

    @property
    def vertical_velocity_cost(self):
        return self.data.qvel[2] ** 2

    @property
    def xy_angular_velocity_cost(self):
        return np.sum(np.square(self.data.qvel[3:5]))

    def action_rate_cost(self, action):
        return np.sum(np.square(self._last_action - action))
    
    def state_vector(self):
        base = super().state_vector()
        return np.concatenate([base, np.array([self.relative_direction]), np.array([self.distance_to_goal])])


    def random_point(self):
        x = np.random.uniform(-self.half_x/8, self.half_x/8)
        y = np.random.uniform(-self.half_y/8, self.half_y/8) #was /4 every /10
        
        return [3,3]

    def _calc_reward(self, action, old_position,time_diff,old_rel_direction):
        objective=np.array(self.objective_point)
        old_distance= np.linalg.norm(objective - old_position)
        new_position= np.array(self.state_vector()[0:2])
        new_distance=self.distance_to_goal
        progress=old_distance-new_distance
        orientation_reward=np.cos(self.relative_direction)+(np.cos(self.relative_direction)-np.cos(old_rel_direction))
        #orientation_reward = 2 * -abs(self.relative_direction)
        #yaw_rate_penalty = -0.05 * abs(self.data.qvel[5])
        #time_eff=(self.distance-new_distance)/max(time_diff,1e-6)
        time_eff=self.calc_vel_objective()
        survival = 0.1 if self.is_healthy else 0.0
        death_penalty = -10.0 if not self.is_healthy[0] else 0.0
        if abs(self.relative_direction) > 0.2:
            reward = progress + 2 * orientation_reward + time_eff + survival + death_penalty
        else:
            reward = 3 * progress + orientation_reward + 2.5 * time_eff + survival + death_penalty
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
        #reward += 0.001*self.calc_leg_spread_penalty() #was*1
       
        return reward, reward_info


    def _get_obs(self):
        # The first three indices are the global x,y,z position of the trunk of the robot
        # The second four are the quaternion representing the orientation of the robot
        # The above seven values are ignored since they are privileged information
        # The remaining 12 values are the joint positions
        # The joint positions are relative to the starting position
        position = self.data.qpos[7:].flatten() - self.model.key_qpos[0, 7:]

        # The first three values are the global linear velocity of the robot
        # The second three are the angular velocity of the robot
        # The remaining 12 values are the joint velocities
        velocity = self.data.qvel.flatten()
        velocity[:3] *= self._velocity_scale

        last_action = self._last_action

        curr_obs = np.concatenate((position, velocity, last_action)).clip(
            -self._clip_obs_threshold, self._clip_obs_threshold
        )

        terrain_window = self.sample_terrain_ahead(window_size=(9,9)).flatten()  # 5x5 grid

        curr_obs= np.concatenate([curr_obs, np.array([self.relative_direction]), np.array([self.distance_to_goal/self.max_distance]), terrain_window])
        return curr_obs
        
    def sample_terrain_ahead(self, window_size):
        robot_pos = self.data.qpos[:3]           # [x, y, z]
        heading = self.calc_direction()          # 2D unit vector [x, y]
        side = np.array([-heading[1], heading[0]])  # perpendicular vector
    
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
        joint_vels = self.data.qvel[6:18]  # 12 motor joints
        forward_joint_ids = [1, 2, 4, 5, 11]
        motion_reward = np.sum(np.abs(joint_vels[forward_joint_ids]))
        return 0.01 * motion_reward  # scale as needed


    
    def reset_model(self):
        # Reset the position and control values with noise
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
        self.objective_point = self.random_point()  # If you want a new one each episode
        self.distance=np.linalg.norm(self.objective_point-np.array([0,0]))
        self.distance_to_goal=np.linalg.norm(self.objective_point-np.array([0,0]))
        self.direction= self.calc_direction()
        self.relative_direction=self.calc_relative_direction(self.direction)
        x, y = self.objective_point
        z=self.get_maps_z(x,y)
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker_body")
        self.model.body_pos[body_id] = [x, y, z+3]
       

        # Reset the variables and sample a new desired velocity
        self._desired_velocity = self._sample_desired_vel()
        self._step = 0
        self._last_action = np.zeros(12)
        self._feet_air_time = np.zeros(4)
        self._last_contacts = np.zeros(4)
        self._last_render_time = -1.0

        observation = self._get_obs()
        #print(">>> Dopo reset:", " pos (x,y):", self.data.qpos[0:2], " z:", self.data.qpos[2], " is_healthy:", self.is_healthy, " reached:", self.reached, " objective_point:", self.objective_point)
        self.start_episode=time.perf_counter()
        self._distance_window=[]
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }

    def _sample_desired_vel(self):
        desired_vel = np.random.default_rng().uniform(
            low=self._desired_velocity_min, high=self._desired_velocity_max
        )
        return np.array([1.0, 0, 0.0])  # TODO: Train with randomized desired_vel
    def get_maps_z(self,X,Y):
        u = (X + self.half_x) / (2*self.half_x)
        v = (Y + self.half_y) / (2*self.half_y)
        # 2) trova il pixel più vicino (round anziché floor)
        j = int(round(u * (self.ncol - 1)))
        i = int(round(v * (self.nrow - 1)))
        # clamp sugli indici
        i = max(0, min(self.nrow-1, i))
        j = max(0, min(self.ncol-1, j))
        # 3) leggi il valore di grigio e scala
        p = self.img[i, j]          # 0–255
        h_rel = p / 65535.0      # 0–1
        return h_rel * self.max_z
    def calc_direction(self): 
        mat = np.zeros(9, dtype=np.float64)  # vettore piatto
        quat = self.data.qpos[3:7].astype(np.float64)
        mujoco.mju_quat2Mat(mat, quat)
        x_axis_world = np.array([mat[0], mat[3], mat[6]])   
        direction_xy = x_axis_world[:2]
        norm=np.linalg.norm(direction_xy)
        if norm<1e-8:
            return np.array([1.0, 0.0])
        direction_xy /= norm

        return direction_xy
    #def calc_relative_direction(self,direction):
    #    rel_direction=self.objective_point-self.data.qpos[0:2]
    #    rel_direction/=np.linalg.norm(rel_direction)
    #    rel_direction=np.dot(direction, rel_direction)
    #    return rel_direction
    def calc_relative_direction(self,direction):
        vec_to_target = self.objective_point - self.data.qpos[0:2]
        norm = np.linalg.norm(vec_to_target)
        if norm < 1e-8:
            return 0.0  # target sovrapposto alla posizione del robot: angolo=0

        # 2. Lo normalizziamo
        vec_to_target /= norm

        # 3. Calcoliamo dot e cross (in 2D) tra forward_dir e vec_to_target
        #    - dot = |a||b| cosθ  (ma qui |a|=|b|=1)
        #    - cross_z = a_x * b_y - a_y * b_x
        dot_val = float(np.dot(direction, vec_to_target))
        cross_z = direction[0] * vec_to_target[1] - direction[1] * vec_to_target[0]

        # 4. Angolo signed = atan2(cross, dot) ∈ (−π, +π]
        theta = math.atan2(cross_z, dot_val)
        return theta
    def calc_vel_objective(self):
        goal_vec = self.objective_point - self.data.qpos[:2]        
        goal_dir = goal_vec / (np.linalg.norm(goal_vec) + 1e-8)
        vel_xy = self.data.qvel[:2]
        v_toward_goal = np.dot(vel_xy, goal_dir)
        return v_toward_goal
