import magnum as mn
import numpy as np
from habitat import logger
from habitat.core.registry import registry

from habitat_extensions.utils import art_utils

from ..task import RearrangeEpisode, RearrangeTask
from ..task_utils import (
    check_start_state,
    compute_region_goals_v1,
    compute_start_state,
    sample_noisy_start_state,
)
from .pick_task import RearrangePickTask


@registry.register_task(name="RearrangePlaceTask-v0")
class RearrangePlaceTask(RearrangePickTask):
    def initialize(self, episode: RearrangeEpisode):
        start_state = None  # (start_pos, start_ori)
        sim_state = self._sim.get_state()  # snapshot

        n_targets = len(self._sim.targets)
        if "TARGET_INDEX" in self._config:
            tgt_indices = [self._config.TARGET_INDEX]
        else:
            tgt_indices = self.np_random.permutation(n_targets)

        # ---------------------------------------------------------------------------- #
        # Sample a collision-free start state
        # ---------------------------------------------------------------------------- #
        for tgt_idx in tgt_indices:
            self._set_target(tgt_idx)
            self.place_goal = np.array(
                self.tgt_T.translation, dtype=np.float32
            )

            if self._has_cache_start_state(episode.episode_id):
                (
                    start_state,
                    start_ee_pos,
                    start_sim_state,
                ) = self._get_cache_start_state(episode.episode_id)
                if start_state is not None:
                    self._initialize_ee_pos(start_ee_pos)
                    self._sim.set_state(start_sim_state, include_robot=False)
            else:
                self._initialize_target_receptacle(episode)
                start_state = self.sample_start_state(episode)
                self._set_cache_start_state(episode.episode_id, start_state)

            if start_state is not None:
                break
            else:
                logger.warning(
                    "Episode {}: fail to sample a valid start state for {}({})".format(
                        episode.episode_id, self.tgt_obj.handle, self.tgt_idx
                    )
                )
                self._sim.set_state(sim_state)  # recover from snapshot

        # -------------------------------------------------------------------------- #
        # Remove validation
        # -------------------------------------------------------------------------- #
        if start_state is None:
            self._set_target(tgt_indices[0])
            self.place_goal = np.array(
                self.tgt_T.translation, dtype=np.float32
            )
            self._initialize_target_receptacle(episode)

            start_state = self.sample_start_state(episode, no_validation=True)
            # raise RuntimeError(
            logger.warning(
                "Episode {}({}): sample a start state without validation".format(
                    episode.episode_id, episode.scene_id
                )
            )

        if self._config.START_STATE is None:
            self._sim.robot.base_pos = start_state[0]
            self._sim.robot.base_ori = start_state[1]
        else:
            self._sim.robot.base_pos = self._config.START_STATE[0]
            self._sim.robot.base_ori = self._config.START_STATE[1]
        # self._sim.robot.open_gripper()
        # self._sim.gripper.snap_to_obj(self.tgt_obj)
        if self._config.STATE_INFO is not None:
            state = self._config.STATE_INFO
            self._sim.set_state(state, include_robot=False)

            self._sim.robot.sim_obj.clear_joint_states()
            self._sim.robot.sim_obj.transformation = state["robot_state"]["T"]
            self._sim.robot.sim_obj.joint_positions = state["robot_state"]["qpos"]
            self._sim.robot.sim_obj.joint_velocities = state["robot_state"]["qvel"]
            for joint_id, (motor_id, new_jms) in state["robot_state"]["motors"].items():
                assert joint_id in self._sim.robot.joint_motors, joint_id
                motor_id, jms = self._sim.robot.joint_motors[joint_id]
                for k, v in new_jms.items():
                    if k in ['motor_type']: continue
                    setattr(jms, k, v)
                self._sim.robot.sim_obj.update_joint_motor(motor_id, jms)
            self._sim.robot._sync_joint_motors()

            self._set_target(tgt_idx)
            self._sim.gripper.desnap(True)  # desnap anyway
            if state["grasped_obj"] is not None:
                self._sim.gripper.snap_to_obj(self._sim.rigid_objs[state["grasped_obj"]])
            # elif state["grasped_marker"] is not None:
            #     self._sim.gripper.snap_to_marker(state["grasped_marker"])
            self.place_goal = np.array(
                self.tgt_T.translation, dtype=np.float32
            )
            # self._sim.set_state(self._config.STATE_INFO)
            # self._set_target(tgt_idx)

        self._sim.internal_step_by_time(0.1)

    def _set_target(self, index):
        super()._set_target(index)
        self.tgt_receptacle_info = self._goal_receptacles[self.tgt_idx]

    def sample_start_state(
        self,
        episode,
        max_trials=20,
        verbose=False,
        no_validation=False,
    ):
        # Generate a initial start pos
        start_pos, _ = compute_start_state(
            self._sim, self.place_goal, init_start_pos=self.init_start_pos
        )

        # Skip if not in the largest island
        if not self._sim.is_at_larget_island(start_pos):
            logger.warning(
                "Episode {}: start_pos({}) is not at the largest island ".format(
                    episode.episode_id, self.tgt_idx
                )
            )
            return None

        pos_noise = self._config.get("BASE_NOISE", 0.05)
        ori_noise = self._config.get("BASE_ANGLE_NOISE", 0.15)

        for i in range(max_trials):
            # Avoid extreme end-effector positions by resampling each time
            self._initialize_ee_pos()

            start_state = sample_noisy_start_state(
                self._sim,
                start_pos,
                self.place_goal,
                pos_noise=pos_noise,
                ori_noise=ori_noise,
                pos_noise_thresh=2 * pos_noise,
                ori_noise_thresh=2 * ori_noise,
                max_trials=10,
                verbose=verbose,
                rng=self.np_random,
            )
            if start_state is None:
                continue
            if no_validation:
                return start_state
            max_ik_error = 0.14
            is_valid = check_start_state(
                self._sim,
                self,
                *start_state,
                task_type="place",
                max_ik_error=max_ik_error,
                max_collision_force=0.0,
                verbose=verbose,
            )
            if is_valid:
                if verbose:
                    print(f"Find a valid start state at {i}-th trial")
                return start_state

    def render(self, mode):
        if self._has_target_in_container():
            pos, ori = compute_start_state(
                self._sim, self.place_goal, self.init_start_pos
            )
            self._sim.visualize_arrow("init_start_pos", pos, ori)

        self._sim.set_object_bb_draw(True, self.tgt_obj.object_id)
        self._sim.visualize_target(self.tgt_idx)
        ret = self._sim.render(mode)
        self._sim.set_object_bb_draw(False, self.tgt_obj.object_id)
        return ret


@registry.register_task(name="RearrangePlaceTask-v1")
class RearrangePlaceTaskV1(RearrangePlaceTask):
    def sample_start_state(
        self, episode, max_trials=20, verbose=False, no_validation=False
    ):
        if self._has_cache_start_positions(episode.episode_id):
            start_positions = self._get_cache_start_positions(
                episode.episode_id
            )
        else:
            start_pos, _ = compute_start_state(self._sim, self.place_goal)
            height = start_pos[1]
            T = mn.Matrix4.translation(self.place_goal)
            # start_positions = compute_start_positions_from_map_v1(
            start_positions = compute_region_goals_v1(
                self._sim,
                T,
                region=None,
                radius=self._config.START_REGION_SIZE,
                height=height,
                max_radius=self._config.MAX_REGION_SIZE,
                debug=False,
            )
            self._set_cache_start_positions(
                episode.episode_id, start_positions
            )

        # print(len(start_positions))
        if start_positions is None or len(start_positions) == 0:
            logger.warning(
                "Episode {} ({}): Unable to find any navigable point around the {}-th target given the map.".format(
                    episode.episode_id, episode.scene_id, self.tgt_idx
                )
            )
            return None

        pos_noise = self._config.get("BASE_NOISE", 0.05)
        ori_noise = self._config.get("BASE_ANGLE_NOISE", 0.15)

        for i in range(max_trials):
            # Avoid extreme end-effector positions by resampling each time
            self._initialize_ee_pos()

            ind = self.np_random.choice(len(start_positions))
            start_state = sample_noisy_start_state(
                self._sim,
                start_positions[ind],
                self.place_goal,
                pos_noise=pos_noise,
                ori_noise=ori_noise,
                pos_noise_thresh=2 * pos_noise,
                ori_noise_thresh=2 * ori_noise,
                max_trials=10,
                verbose=verbose,
                rng=self.np_random,
            )
            if start_state is None:
                continue

            if check_start_state(
                self._sim,
                self,
                *start_state,
                task_type="place",
                max_collision_force=0.0,
                verbose=verbose,
            ):
                if verbose:
                    print(f"Find a valid start state at {i}-th trial")
                return start_state
