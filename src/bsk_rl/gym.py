"""Gymnasium and PettingZoo environments for satellite tasking problems."""

import functools
import logging
import os
from copy import deepcopy
from time import time_ns
from typing import Any, Callable, Generic, Iterable, Optional, TypeVar, Union

import numpy as np
from gymnasium import Env, spaces
from pettingzoo.utils.env import AgentID, ParallelEnv

from bsk_rl.comm import CommunicationMethod, NoCommunication
from bsk_rl.data import GlobalReward, NoReward
from bsk_rl.data.composition import ComposedReward
from bsk_rl.sats import Satellite
from bsk_rl.scene import Scenario
from bsk_rl.sim import Simulator
from bsk_rl.sim.world import WorldModel
from bsk_rl.utils import logging_config, vizard

logger = logging.getLogger(__name__)


SatObs = TypeVar("SatObs")
SatAct = TypeVar("SatAct")
MultiSatObs = tuple[SatObs, ...]
MultiSatAct = Iterable[SatAct]
SatArgRandomizer = Callable[[list[Satellite]], dict[Satellite, dict[str, Any]]]

NO_ACTION = int(2**31) - 1


def is_no_action(action):
    """Check if the action is a no-action placeholder."""
    if action is None:
        return True
    if isinstance(action, (int, np.integer)) and action == NO_ACTION:
        return True
    if isinstance(action, (np.ndarray, list, tuple)) and np.allclose(action, NO_ACTION):
        return True
    return False


def no_action_like(action):
    """Generate an action that is the same type and shape as the no-action placeholder."""
    if isinstance(action, (int, np.integer)):
        return NO_ACTION
    return action * 0 + NO_ACTION  # Aggressively try to convert while retaining type


class GeneralSatelliteTasking(Env, Generic[SatObs, SatAct]):
    def __init__(
        self,
        satellites: Union[Satellite, list[Satellite]],
        scenario: Optional[Scenario] = None,
        rewarder: Optional[Union[GlobalReward, list[GlobalReward]]] = None,
        world_type: Optional[type[WorldModel]] = None,
        world_args: Optional[dict[str, Any]] = None,
        communicator: Optional[CommunicationMethod] = None,
        sat_arg_randomizer: Optional[SatArgRandomizer] = None,
        sim_rate: float = 1.0,
        max_step_duration: float = 1e9,
        failure_penalty: float = -1.0,
        time_limit: Union[float, Callable] = float("inf"),
        terminate_on_time_limit: bool = False,
        generate_obs_retasking_only: bool = False,
        dtype: Optional[np.dtype] = None,
        log_level: Union[int, str] = logging.WARNING,
        log_dir: Optional[str] = None,
        vizard_dir: Optional[str] = None,
        vizard_settings: Optional[dict[str, Any]] = None,
        render_mode=None,
    ) -> None:
        """A `Gymnasium <https://gymnasium.farama.org>`_ environment adaptable to a wide range satellite tasking problems.

        These problems involve satellite(s) being tasked to complete tasks and maintain
        aliveness. These tasks often include rewards for data collection. The environment
        can be configured for any collection of satellites, including heterogenous
        constellations. Other configurable aspects are the scenario (e.g.
        imaging targets), data collection and recording, and intersatellite
        communication of data.

        The state space is a tuple containing the state of each satellite. Actions are
        assigned as a tuple of actions, one per satellite.

        Args:
            satellites: Satellite(s) to be simulated. See :ref:`bsk_rl.sats`.
            scenario: Environment the satellite is acting in; contains information
                about targets, etc. See :ref:`bsk_rl.scene`.
            rewarder: Handles recording and rewarding for data collection towards
                objectives. Can be a single rewarder or a tuple of multiple rewarders.
                See :ref:`bsk_rl.data`.
            communicator: Manages communication between satellites. See :ref:`bsk_rl.comm`.
            sat_arg_randomizer: For correlated randomization of satellites arguments. Should
                be a function that takes a list of satellites and returns a dictionary that
                maps satellites to dictionaries of satellite model arguments to be overridden.
            world_type: Type of Basilisk world model to be constructed.
            world_args: Arguments for :class:`~bsk_rl.sim.world.WorldModel` construction.
                Should be in the form of a dictionary with keys corresponding to the
                arguments of the constructor and values that are either the desired value
                or a function that takes no arguments and returns a randomized value.
            sim_rate: [s] Rate for model simulation.
            max_step_duration: [s] Maximum time to propagate sim at a step. If
                satellites are using variable interval actions, the actual step duration
                will be less than or equal to this value. It is preferable to set durations
                in the actions themselves.
            failure_penalty: Reward for satellite failure. Should be nonpositive.
            time_limit: [s] Time at which to truncate the simulation. Can also be a function
                that takes no arguments and returns a float. This function will be called
                every time the environment is reset to randomize the time limit.
            terminate_on_time_limit: Send terminations signal time_limit instead of just
                truncation.
            generate_obs_retasking_only: If True, only generate observations for satellites
                that require retasking. All other satellites will receive an observation of
                zeros.
            dtype: Data type for satellite observations. If None, the data type specified
                in the satellite.
            log_level: Logging level for the environment. Default is ``WARNING``.
            log_dir: Directory to write logs to in addition to the console.
            vizard_dir: Path to save Vizard visualization files. If None, no Vizard-related
                modules will be imported.
            vizard_settings: Settings for Vizard visualization. Set in ``vizIstance.settings``.
                Additionally, the key ``vizard_rate`` can be set to the rate at which Vizard updates.
                The key ``use_simple_earth`` can be set to use a lower detail Earth shader
                that may help viewing ground locations. Other settings can be found
                `in the Basilisk documentation <https://avslab.github.io/basilisk/Vizard/vizardAdvanced/vizardSettings.html#id1>`_.
            render_mode: Unused.
        """
        self.seed = None
        self._configure_logging(log_level, log_dir)
        if vizard_dir is not None:
            vizard.VIZARD_PATH = vizard_dir
        if vizard_settings is not None and vizard_dir is None:
            logger.warning(
                "Vizard settings provided but Vizard is not enabled. Ignoring settings."
            )
        self.vizard_settings = vizard_settings if vizard_settings is not None else {}

        if isinstance(satellites, Satellite):
            satellites = [satellites]
        self.satellites = deepcopy(satellites)

        self.dtype = dtype
        while True:
            for satellite in self.satellites:
                if [sat.name for sat in self.satellites].count(satellite.name) > 1:
                    for i, sat_rename in enumerate(
                        [sat for sat in self.satellites if sat.name == satellite.name]
                    ):
                        new_name = f"{sat_rename.name}_{i}"
                        logger.warning(
                            f"Renaming satellite {sat_rename.name} to {new_name}"
                        )
                        sat_rename.name = new_name

                # Update satellite observation dtypes
                if self.dtype is not None:
                    satellite.observation_builder.dtype = self.dtype

            # Check if all satellite names are unique
            sat_names = [sat.name for sat in self.satellites]
            if len(sat_names) == len(set(sat_names)):
                break

        self.simulator: Simulator

        if sat_arg_randomizer is None:
            sat_arg_randomizer = lambda sats: {}
        self.sat_arg_randomizer = sat_arg_randomizer

        if scenario is None:
            scenario = Scenario()

        if world_type is None:
            world_type = self._minimum_world_model()
        self.world_type = world_type
        if world_args is None:
            world_args = self.world_type.default_world_args()
        self.world_args_generator = self.world_type.default_world_args(**world_args)

        self.scenario = deepcopy(scenario)
        self.scenario.link_satellites(self.satellites)

        rewarder = deepcopy(rewarder)
        if rewarder is None:
            rewarder = NoReward()
        if (
            isinstance(rewarder, Iterable)
            and not type(rewarder).__name__ == "MagicMock"
        ):
            rewarder = ComposedReward(*rewarder)
        self.rewarder = rewarder
        self.rewarder.link_scenario(self.scenario)

        if communicator is None:
            communicator = NoCommunication()
        self.communicator = deepcopy(communicator)
        self.communicator.link_satellites(self.satellites)

        self.sim_rate = sim_rate
        self.max_step_duration = max_step_duration
        self.failure_penalty = failure_penalty
        if self.failure_penalty > 0:
            logger.warn("Failure penalty should be nonpositive")
        if callable(time_limit):
            self.time_limit_generator = time_limit
        else:
            self.time_limit_generator = lambda: time_limit
        self.terminate_on_time_limit = terminate_on_time_limit
        self.latest_step_duration = 0.0
        self.render_mode = render_mode
        self.generate_obs_retasking_only = generate_obs_retasking_only
        # Reward soft-clip scale (can be tuned from wrappers if needed)
        self.reward_clip_scale = 3000.0
        # Optional curriculum config (set by env wrapper)
        self.curriculum_config = None
        self._curriculum_state = {
            "episode_idx": 0,
            "stage": 0,
            "stable_count": 0,
            "applied_for_episode": False,
            "total_steps": 0,
            "episode_steps": 0,
            "last_step_applied": 0,
            "difficulty": 0.0,
            "base_params": None,
        }
        self._curriculum_stats = {
            "per_stage": {},
            "last_episode_logged": -1,
        }
        self._last_episode_done = False

    def _minimum_world_model(self) -> type[WorldModel]:
        """Determine the minimum world model required by the satellites."""
        types = set(
            sum(
                [satellite.dyn_type._requires_world() for satellite in self.satellites],
                [],
            )
        )
        if len(types) == 1:
            return list(types)[0]
        for test_type in types:
            if all([issubclass(test_type, other_type) for other_type in types]):
                return test_type

        # Else compose all types into a new class
        class MinimumEnv(*types):
            pass

        return MinimumEnv

    def get_satellite(self, name: str) -> "Satellite":
        """Get a satellite by name.

        Args:
            name: Name of the satellite to retrieve.

        Returns:
            The satellite object with the specified name.
        """
        for sat in self.satellites:
            if sat.name == name:
                return sat
        raise ValueError(f"Satellite with name '{name}' not found.")

    def _configure_logging(self, log_level, log_dir=None):
        """Configure bsk_rl logger with console and optional file output.
        
        Args:
            log_level: Logging level (DEBUG/INFO/WARNING/ERROR)
            log_dir: Optional file path for file logging
        """
        if isinstance(log_level, str):
            log_level = log_level.upper()
        logger = logging.getLogger("bsk_rl")
        logger.setLevel(log_level)
        # 避免重复输出：不向 root 传播，由 bsk_rl 自己的 handler 负责输出
        logger.propagate = False

        # Ensure each process has its own logger to avoid conflicts when printing
        # sim timestamps. Running multiple environments in the same process in
        # parallel will cause logging times to be incorrectly reported.
        warn_new_env = False
        for handler in logger.handlers:
            if hasattr(handler, 'filters') and handler.filters and hasattr(handler.filters[0], 'proc_id'):
                if handler.filters[0].proc_id == os.getpid():
                    logger.handlers.remove(handler)
                    warn_new_env = True

        ch = logging.StreamHandler()
        ch.setFormatter(logging_config.SimFormatter(color_output=True))
        ch.addFilter(logging_config.ContextFilter(env=self, proc_id=os.getpid()))
        logger.addHandler(ch)
        if warn_new_env:
            logger.warning(
                f"Creating logger for new env on PID={os.getpid()}. "
                "Old environments in process may now log times incorrectly."
            )

        if log_dir is not None:
            fh = logging.FileHandler(log_dir)
            fh.setFormatter(logging_config.SimFormatter(color_output=False))
            fh.addFilter(logging_config.ContextFilter(env=self, proc_id=os.getpid()))
            logger.addHandler(fh)

    def _generate_world_args(self) -> None:
        """Instantiate world_args from any randomizers in provided world_args."""
        self.world_args = {
            k: v if not callable(v) else v()
            for k, v in self.world_args_generator.items()
        }

    def _randomize_time_limit(self) -> None:
        time_limit = self.time_limit_generator()
        time_limit = np.ceil(time_limit / self.sim_rate) * self.sim_rate
        self.time_limit = time_limit

    def reset(
        self,
        seed: Optional[int] = None,
        options=None,
    ) -> tuple[MultiSatObs, dict[str, Any]]:
        """Reconstruct the simulator and reset the scenario.

        Satellite and world arguments get randomized on reset, if :class:`~bsk_rl.GeneralSatelliteTasking` ``.world_args``
        or :class:`~bsk_rl.sats.Satellite` ``.sat_args`` includes randomization functions.

        Certain classes in ``bsk_rl`` have a ``reset_pre_sim_init`` and/or ``reset_post_sim_init``
        method. These methods are respectively called before and after the new Basilisk
        :class:`~bsk_rl.sim.Simulator` is created. These allow for reset actions that
        feed into the underlying simulation and those that are dependent on the underlying
        simulation to be performed.

        Args:
            seed: Gymnasium environment seed.
            options: Unused.

        Returns:
            observation, info
        """
        # Explicitly delete the Basilisk simulation before creating a new one.
        self.delete_simulator()

        if seed is None:
            seed = time_ns() % 2**32
        logger.info(f"Resetting environment with seed={seed}")
        if not hasattr(self, "_reset_counter"):
            self._reset_counter = 0
        self._reset_counter += 1
        self.seed = seed
        super().reset(seed=self.seed)
        np.random.seed(self.seed)

        self._randomize_time_limit()

        self.scenario.reset_overwrite_previous()
        self.rewarder.reset_overwrite_previous()
        self.communicator.reset_overwrite_previous()
        for i, satellite in enumerate(self.satellites):
            satellite.reset_overwrite_previous()
            satellite.create_vizard_data(color=vizard.get_color(i))
        self.latest_step_duration = 0.0

        self._generate_world_args()
        overrides = self.sat_arg_randomizer(self.satellites)
        for satellite in self.satellites:
            sat_overrides = overrides.get(satellite, {})
            satellite.generate_sat_args(
                utc_init=self.world_args["utc_init"], **sat_overrides
            )

        self.scenario.utc_init = self.world_args["utc_init"]

        self.scenario.reset_pre_sim_init()
        self.rewarder.reset_pre_sim_init()
        self.communicator.reset_pre_sim_init()

        for satellite in self.satellites:
            self.rewarder.create_data_store(satellite)
            self.rewarder.data += satellite.data_store.data
            satellite.reset_pre_sim_init()

        self.simulator = Simulator(
            self.satellites,
            self.world_type,
            self.world_args,
            sim_rate=self.sim_rate,
            max_step_duration=self.max_step_duration,
            time_limit=self.time_limit,
        )
        self.simulator.setup_vizard(**self.vizard_settings)

        self.scenario.reset_during_sim_init()
        self.rewarder.reset_during_sim_init()
        self.communicator.reset_during_sim_init()

        self.simulator.finish_init()

        self.scenario.reset_post_sim_init()
        self.rewarder.reset_post_sim_init()
        self.communicator.reset_post_sim_init()

        for satellite in self.satellites:
            satellite.reset_post_sim_init()
            satellite.data_store.update_from_logs()

        # === 初始化 Episode 级别的累积变量 ===
        # 这些变量会在整个回合中累积，用于计算 episode-level 指标
        self._episode_completed_tasks = 0
        self._episode_expired_tasks = 0
        self._episode_processed_data = 0.0
        self._episode_offloaded_data = 0.0
        self._episode_energy_consumed = 0.0
        
        # ---学术级指标累积器 ---
        self._ep_stats = {
            "total_hops": 0,
            "all_completed_task_ids": set(),  # ✅ 所有完成任务 ID（包括本地和协作）
            "remote_task_ids": set(),      # ✅ 使用 set 统计唯一协作完成任务
            "completed_hop_task_ids": set(),  # ✅ hop_count>0 的完成任务
            "offloaded_task_ids": set(),   # ✅ 实际发生数据卸载的任务（后续汇总）
            "all_expired_task_ids": set(),   # ✅ 所有过期任务 ID
            "collab_failed_ids": set(),    # ✅ 使用 set 统计唯一协作失败任务
            "latencies": [],  # 存储所有完成任务的时延 [s]
            "energy_compute": 0.0,
            "energy_comm": 0.0,
            "sat_processed_data": {sat.name: 0.0 for sat in self.satellites}, # 用于 Jain's Index
            "completed_data_total": 0.0,  # ✅ 成功完成任务的数据量 [bits]，用于能效计算
            
            # 🆕 论文风格指标（用于 Result 模块画图）
            # A. 价值密度：平均完成任务优先级
            "completed_priorities": [],  # 存储所有完成任务的优先级
            
            # B. 卸载效率：尝试次数 vs 成功率
            "offload_attempts": 0,       # 卸载尝试总次数（包括失败的）
            "offload_successes": 0,      # 卸载成功次数

            # Level-2 路由诊断
            "level2_decisions": 0,       # Level-2 路由决策总次数
            "level2_self_selected": 0,   # Level-2 选择本地处理次数

            # C. 时延分解 (Latency Breakdown)
            "latency_transmission": [],  # 传输时延 [s]
            "latency_queuing": [],       # 排队时延 [s]  
            "latency_computing": [],     # 计算时延 [s]

            # Level-1 分配统计（用于诊断 alpha_sats 是否被压低）
            "alpha_ud_sum": 0.0,
            "alpha_cloud_sum": 0.0,
            "alpha_sats_sum": 0.0,
            "alpha_sat_local_sum": 0.0,
            "alpha_sat_offload_sum": 0.0,
            "alpha_ud_sum_sq": 0.0,
            "alpha_cloud_sum_sq": 0.0,
            "alpha_sats_sum_sq": 0.0,
            "alpha_sat_local_sum_sq": 0.0,
            "alpha_sat_offload_sum_sq": 0.0,
            "alpha_steps": 0,

            # 卸载统计（用于诊断 offloaded_data / relay_contribution）
            "offloaded_data_bits": 0.0,
            "relay_offloaded_bits": 0.0,
            "first_hop_offloaded_bits": 0.0,
            "prev_offloaded_total": {sat.name: 0.0 for sat in self.satellites},
            "prev_relay_offloaded_total": {sat.name: 0.0 for sat in self.satellites},
            "prev_first_hop_offloaded_total": {sat.name: 0.0 for sat in self.satellites},

            # 奖励分解统计（用于判断完成奖励是否被成本淹没）
            "reward_completion_sum": 0.0,
            "reward_cost_sum": 0.0,
            "reward_total_sum": 0.0,
            # 🆕 奖励分布统计（用于判断裁剪是否过强）
            "reward_raw_list": [],
            "reward_clip_list": [],
        }

        if hasattr(self, "_curriculum_state"):
            self._curriculum_state["episode_idx"] = (
                self._curriculum_state.get("episode_idx", 0) + 1
            )
            self._curriculum_state["applied_for_episode"] = False
            self._curriculum_state["episode_steps"] = 0
        # ✅ Function-mode curriculum: apply at episode start (so ep-1 is included)
        cfg = getattr(self, "curriculum_config", None)
        if isinstance(cfg, dict) and cfg.get("enable", False):
            mode = str(cfg.get("mode", cfg.get("curriculum_mode", "staged"))).lower()
            if mode in ("function", "continuous") and hasattr(
                self, "_curriculum_state"
            ):
                axis = str(cfg.get("axis", "episodes")).lower()
                warmup_episodes = int(cfg.get("warmup_episodes", 0))
                episode_idx = int(self._curriculum_state.get("episode_idx", 0))
                if warmup_episodes <= 0 or episode_idx > warmup_episodes:
                    difficulty = self._compute_curriculum_difficulty(cfg)
                    self._apply_curriculum_function(cfg, float(difficulty))
                    self._curriculum_state["difficulty"] = float(difficulty)
                    if axis == "steps":
                        self._curriculum_state["last_step_applied"] = int(
                            self._curriculum_state.get("total_steps", 0)
                        )
                # Prevent duplicate apply at episode end for this episode
                self._curriculum_state["applied_for_episode"] = True
        self._last_episode_done = False
        self._death_snapshot_logged = False

        # Ensure neighbor lists are initialized before the first observation.
        try:
            self._refresh_neighbor_lists()
            self._apply_neighbor_snapshot()
        except Exception as exc:
            logger.debug(f"[LOS] neighbor refresh during reset failed: {exc}")

        observation = self._get_obs()
        info = self._get_info()
        logger.info("Environment reset (count=%s)", self._reset_counter)
        return observation, info

    def delete_simulator(self):
        """Delete Basilisk objects.

        Only the simulator contains strong references to BSK models, so deleting it
        will delete all Basilisk objects. Enable debug-level logging to verify that the
        simulator, FSW, dynamics, and world models are all deleted on reset.
        """
        try:
            del self.simulator
        except AttributeError:
            pass

    def _get_obs(self) -> MultiSatObs:
        """Compose satellite observations into a single observation.

        Returns:
            tuple: Joint observation
        """
        if self.generate_obs_retasking_only:
            return tuple(
                (
                    satellite.get_obs()
                    if satellite.requires_retasking
                    else satellite.observation_space.low * 0
                )
                for satellite in self.satellites
            )
        else:
            return tuple(satellite.get_obs() for satellite in self.satellites)

    def _apply_neighbor_snapshot(self) -> None:
        for satellite in self.satellites:
            if not hasattr(satellite, "action_builder") or not hasattr(
                satellite.action_builder, "action_spec"
            ):
                continue
            snapshot = getattr(satellite, "_neighbor_snapshot", None)
            if snapshot is None:
                continue
            for act in satellite.action_builder.action_spec:
                if not hasattr(act, "neighbor_satellites") or not hasattr(
                    act, "add_neighbor"
                ):
                    continue
                act.neighbor_satellites.clear()
                max_neighbors = getattr(act, "max_neighbors", 4)
                for neighbor in list(snapshot)[:max_neighbors]:
                    act.add_neighbor(neighbor)

    def _get_fixed_neighbor_list(self, sat_index: int, max_neighbors: int):
        if not getattr(self, "fixed_topology", False):
            return None
        n_planes = getattr(self, "topology_n_planes", None)
        sats_per_plane = getattr(self, "topology_sats_per_plane", None)
        if not n_planes or not sats_per_plane:
            return None
        if max_neighbors < 4:
            return None
        total = n_planes * sats_per_plane
        if total != len(self.satellites):
            return None

        plane_idx = sat_index // sats_per_plane
        slot_idx = sat_index % sats_per_plane
        front_slot = (slot_idx + 1) % sats_per_plane
        back_slot = (slot_idx - 1 + sats_per_plane) % sats_per_plane
        right_plane = (plane_idx + 1) % n_planes
        left_plane = (plane_idx - 1 + n_planes) % n_planes

        indices = [
            plane_idx * sats_per_plane + front_slot,
            plane_idx * sats_per_plane + back_slot,
            left_plane * sats_per_plane + slot_idx,
            right_plane * sats_per_plane + slot_idx,
        ]
        used = {sat_index}
        fixed = []
        for idx in indices:
            if idx in used or idx < 0 or idx >= len(self.satellites):
                fixed.append(None)
                continue
            used.add(idx)
            fixed.append(self.satellites[idx])
        if max_neighbors > len(fixed):
            fixed.extend([None] * (max_neighbors - len(fixed)))
        return fixed

    def _refresh_neighbor_lists(self) -> None:
        max_diag_prints = getattr(self, "_los_diag_max_prints", 3)
        diag_count = getattr(self, "_los_diag_print_count", 0)
        diag_active = diag_count < max_diag_prints
        diag_total = 0
        diag_with_logs = 0
        diag_no_logs = 0
        diag_with_samples = 0
        diag_no_samples = 0
        diag_zero_visible = 0
        diag_visible_counts = []
        if not hasattr(self, "_vis_stats_interval"):
            self._vis_stats_interval = 50
        if not hasattr(self, "_vis_stats_calls"):
            self._vis_stats_calls = 0
        if not hasattr(self, "_vis_hist"):
            self._vis_hist = {"unknown": 0}

        communicator = getattr(self, "communicator", None)
        has_los_logs = communicator is not None and hasattr(communicator, "los_logs")

        for sat_index, satellite in enumerate(self.satellites):
            if not hasattr(satellite, "action_builder") or not hasattr(
                satellite.action_builder, "action_spec"
            ):
                continue

            all_other_sats = [s for s in self.satellites if s != satellite]
            visible_candidates = None
            saw_any_sample = False
            logs = {}

            if has_los_logs:
                logs = communicator.los_logs.get(satellite, {})
                if logs:
                    try:
                        visible_candidates = []
                        for s, los_logger in logs.items():
                            has_access = getattr(los_logger, "hasAccess", None)
                            if has_access is None:
                                continue
                            if isinstance(has_access, (list, tuple, np.ndarray)):
                                if len(has_access) == 0:
                                    continue
                                saw_any_sample = True
                                access = any(has_access)
                            else:
                                saw_any_sample = True
                                access = bool(has_access)
                            if access:
                                visible_candidates.append(s)
                        if not saw_any_sample:
                            visible_candidates = None
                    except Exception as exc:
                        if hasattr(satellite, "logger"):
                            satellite.logger.debug(f"[LOS] log parse failed: {exc}")
                        visible_candidates = None

            if diag_active:
                diag_total += 1
                if not has_los_logs:
                    pass
                elif logs:
                    diag_with_logs += 1
                else:
                    diag_no_logs += 1
                if saw_any_sample:
                    diag_with_samples += 1
                    visible_count = (
                        len(visible_candidates) if visible_candidates is not None else 0
                    )
                    diag_visible_counts.append(visible_count)
                    if visible_count == 0:
                        diag_zero_visible += 1
                else:
                    diag_no_samples += 1

            stats_recorded = False

            for act in satellite.action_builder.action_spec:
                if not hasattr(act, "neighbor_satellites") or not hasattr(
                    act, "add_neighbor"
                ):
                    continue
                act.neighbor_satellites.clear()

                max_neighbors = getattr(act, "max_neighbors", 4)
                fixed_neighbors = self._get_fixed_neighbor_list(
                    sat_index, max_neighbors
                )
                if fixed_neighbors is not None:
                    selected_neighbors = fixed_neighbors
                    if visible_candidates is None:
                        mask = [
                            1.0 if neighbor is not None else 0.0
                            for neighbor in selected_neighbors
                        ]
                    else:
                        mask = [
                            1.0 if neighbor in visible_candidates else 0.0
                            for neighbor in selected_neighbors
                        ]
                    satellite._neighbor_snapshot = list(selected_neighbors)
                    satellite._neighbor_mask_snapshot = mask
                    if not stats_recorded:
                        if saw_any_sample:
                            visible_count = int(sum(1 for m in mask if m > 0.0))
                            self._vis_hist[str(visible_count)] = (
                                self._vis_hist.get(str(visible_count), 0) + 1
                            )
                        else:
                            self._vis_hist["unknown"] = (
                                self._vis_hist.get("unknown", 0) + 1
                            )
                        stats_recorded = True
                    for neighbor in selected_neighbors:
                        act.add_neighbor(neighbor)
                    continue

                candidate_pool = (
                    visible_candidates
                    if visible_candidates is not None
                    else all_other_sats
                )

                def get_distance(other_sat):
                    try:
                        if hasattr(satellite, "dynamics") and hasattr(
                            other_sat, "dynamics"
                        ):
                            r_self = satellite.dynamics.r_BN_N
                            r_other = other_sat.dynamics.r_BN_N
                            return np.linalg.norm(r_self - r_other)
                    except Exception:
                        pass
                    return float("inf")

                sorted_by_distance = sorted(candidate_pool, key=get_distance)
                selected_neighbors = sorted_by_distance[:max_neighbors]
                satellite._neighbor_snapshot = list(selected_neighbors)
                satellite._neighbor_mask_snapshot = [1.0] * len(selected_neighbors)
                if not stats_recorded:
                    if saw_any_sample:
                        visible_count = len(selected_neighbors)
                        self._vis_hist[str(visible_count)] = (
                            self._vis_hist.get(str(visible_count), 0) + 1
                        )
                    else:
                        self._vis_hist["unknown"] = (
                            self._vis_hist.get("unknown", 0) + 1
                        )
                    stats_recorded = True
                for neighbor in selected_neighbors:
                    act.add_neighbor(neighbor)

        if diag_active:
            sim_time = getattr(self.simulator, "sim_time", 0.0)
            if not has_los_logs:
                diag_msg = (
                    f"[LOS DIAG] t={sim_time:.2f} los_logs unavailable; "
                    "skip LoS visibility."
                )
                logger.warning("%s", diag_msg)
            else:
                if diag_visible_counts:
                    min_vis = min(diag_visible_counts)
                    max_vis = max(diag_visible_counts)
                    avg_vis = sum(diag_visible_counts) / len(diag_visible_counts)
                else:
                    min_vis = max_vis = avg_vis = 0.0
                diag_msg = (
                    f"[LOS DIAG] t={sim_time:.2f} sats={diag_total} logs={diag_with_logs} "
                    f"no_logs={diag_no_logs} samples={diag_with_samples} "
                    f"no_samples={diag_no_samples} zero_visible={diag_zero_visible} "
                    f"visible(min/avg/max)={min_vis:.1f}/{avg_vis:.2f}/{max_vis:.1f}"
                )
                logger.warning("%s", diag_msg)
            self._los_diag_print_count = diag_count + 1

        self._vis_stats_calls += 1
        if self._vis_stats_calls % self._vis_stats_interval == 0:
            hist = self._vis_hist
            unknown = int(hist.get("unknown", 0))
            known = int(sum(v for k, v in hist.items() if k != "unknown"))
            if known > 0:
                keys = sorted(int(k) for k in hist.keys() if k != "unknown")
                avg_vis = sum(k * hist[str(k)] for k in keys) / known
                ge2 = sum(hist[str(k)] for k in keys if k >= 2)
                ratio_ge2 = ge2 / known
                dist = ", ".join(f"{k}:{hist[str(k)]}" for k in keys)
                logger.warning(
                    "[VIS STATS] samples=%d unknown=%d avg=%.2f ge2=%.2f dist={%s}",
                    known,
                    unknown,
                    avg_vis,
                    ratio_ge2,
                    dist,
                )
            self._vis_hist = {"unknown": 0}

    def _get_info(self) -> dict[str, Any]:
        """Compose satellite info and compute episode-level metrics.
        
        This method implements a Consumer Pattern for buffer-based statistics:
        - Reads completed_tasks_buffer and expired_tasks_buffer from each satellite
        - Accumulates statistics into self._ep_stats
        - Clears buffers after reading to prevent double counting
        
        Episode Metrics (ep_xxx) - Logged to WandB:
        ============================================
        
        A. System Performance (系统效能):
        - ep_completion_rate: 任务成功率 = completed / (completed + expired)
        - ep_completed: 已完成任务总数 (Episode累积)
        - ep_expired: 已过期任务总数 (Episode累积)
        
        B. QoS (服务质量):
        - ep_avg_latency: 平均端到端时延 [s]
        - ep_latency_p95: 95分位时延 [s] - 用于SLA保证
        - ep_latency_max: 最大时延 [s]
        
        C. Resource Efficiency (资源效率):
        - ep_energy_efficiency: 能效比 [bits/J]
        - ep_energy_comm_ratio: 通信能耗占比 (暂时预留接口)
        
        D. Collaboration (协作特性 - 证明MARL优势):
        - ep_load_balance_jain: Jain公平指数 [1/n, 1] - 越大越均衡
        - ep_avg_hops: 平均跳数 - 协作深度
        - ep_collab_contribution_ratio: 协作贡献率 = 异地完成数 / 总完成数
        - ep_collab_failure_ratio: 协作失败率 = 协作过期数 / 总过期数
        
        Internal Metrics (_xxx) - Debug Only:
        =====================================
        - _step_energy: 本Step能耗增量 [J]
        - d_ts: 本Step仿真时长 [s]
        """
        info: dict[str, Any] = {
            satellite.name: {"requires_retasking": satellite.requires_retasking}
            for satellite in self.satellites
        }

        reward_components = None
        rewarder = self.rewarder
        if hasattr(rewarder, "last_reward_components"):
            reward_components = rewarder.last_reward_components
        elif hasattr(rewarder, "rewarders"):
            for sub_rewarder in rewarder.rewarders:
                if hasattr(sub_rewarder, "last_reward_components"):
                    reward_components = sub_rewarder.last_reward_components
                    break
        
        # --- 1. 增量更新 Episode 统计量 (Consumer Pattern) ---
        step_completed_count = 0
        step_energy_total = 0.0
        
        for sat in self.satellites:
            # A. 处理完成任务缓冲区
            if hasattr(sat, 'completed_tasks_buffer'):
                
                for task in sat.completed_tasks_buffer:
                    # 累积基础统计
                    hop_count = getattr(task, 'hop_count', 0)
                    access_sat = getattr(task, 'access_satellite', '')
                    task_id = getattr(task, 'task_id', id(task))  # 获取任务ID
                    
                    # ✅ 添加到所有完成任务集合
                    self._ep_stats["all_completed_task_ids"].add(task_id)
                    
                    self._ep_stats["total_hops"] += hop_count
                    # ✅ 使用 set.add() 确保每个任务只统计一次
                    if hop_count > 0 or (access_sat and access_sat != sat.name):
                        self._ep_stats["remote_task_ids"].add(task_id)
                    if hop_count > 0:
                        self._ep_stats["completed_hop_task_ids"].add(task_id)
                    
                    # 记录时延分布 (用于 P95 计算)
                    # ✅ 直接从任务对象计算时延，而不是查询 slice_registry
                    # T_total = max(T_UD, T_SAT, T_CLOUD)
                    t_ud = getattr(task, 't_ud_path', 0.0)
                    t_sat = getattr(task, 't_sat_path', 0.0)
                    t_cloud = getattr(task, 't_cloud_path', 0.0)
                    task_latency = max(t_ud, t_sat, t_cloud)
                    
                    if task_latency > 0:
                        self._ep_stats["latencies"].append(task_latency)
                    
                    # 🆕 论文风格指标收集
                    # A. 价值密度：记录完成任务的优先级
                    task_priority = getattr(task, 'priority', 0.5)
                    self._ep_stats["completed_priorities"].append(task_priority)
                    
                    # B. 卸载效率：成功统计在 episode 末尾按“真实数据卸载”计算
                    
                    # C. 时延分解 (Latency Breakdown)
                    # 从任务中提取各分量时延
                    t_tx = getattr(task, 't_tx_total', 0.0)  # 传输时延
                    t_queue = getattr(task, 't_queue', 0.0)  # 排队时延
                    t_compute = getattr(task, 't_compute', 0.0)  # 计算时延
                    
                    if t_tx > 0 or t_queue > 0 or t_compute > 0:
                        self._ep_stats["latency_transmission"].append(t_tx)
                        self._ep_stats["latency_queuing"].append(t_queue)
                        self._ep_stats["latency_computing"].append(t_compute)
                    
                    # ✅ 累积成功完成任务的数据量（用于能效计算）
                    # 使用 original_data_size 而不是 data_size（后者可能被部分消耗）
                    completed_data = getattr(task, 'original_data_size', 0.0)
                    self._ep_stats["completed_data_total"] += completed_data
                    
                    step_completed_count += 1
                
                # ✅ 关键：读取后清空缓冲区，防止重复计算或与 step 不对齐
                sat.completed_tasks_buffer.clear()
            
            # B. 处理处理数据量 (用于 Jain's Index)
            # 使用增量：当前总处理量 - 上次记录的处理量 (需要维护上次状态？较为复杂)
            # 简化方案：直接使用 sat.processed_data_total (它是累积值)，每步更新到 _ep_stats
            current_processed = getattr(sat, 'processed_data_total', 0.0)
            self._ep_stats["sat_processed_data"][sat.name] = current_processed
            
            # C. 能耗拆解 (如果卫星支持细分)
            # 这里我们需要本步的能耗增量。sat.data_store.new_data 是本步增量。
            # 如果没有细分字段，暂时全部算作 Compute 或按比例估算
            if hasattr(sat, 'data_store'):
                new_data = getattr(sat.data_store, 'new_data', None)
                if new_data is not None:
                    step_energy = getattr(new_data, 'energy_consumed', 0.0)
                    if not (np.isnan(step_energy) or np.isinf(step_energy)):
                        step_energy_total += step_energy
                        # 尝试获取细分能耗 (需要 Satellite 类支持，暂时预留接口)
                        self._ep_stats["energy_compute"] += step_energy  # 默认全归计算
                    energy_tx = getattr(new_data, "energy_tx", 0.0) or 0.0
                    energy_rx = getattr(new_data, "energy_rx", 0.0) or 0.0
                    energy_comm = energy_tx + energy_rx
                    if not (np.isnan(energy_comm) or np.isinf(energy_comm)):
                        self._ep_stats["energy_comm"] += energy_comm
            
            # D. 🆕 卸载尝试统计（论文指标）
            offload_attempts = getattr(sat, 'offload_attempts_this_step', 0)
            self._ep_stats["offload_attempts"] += offload_attempts

            # E. 🆕 Level-2 路由自选比例统计（诊断用）
            level2_decisions = getattr(sat, "level2_decisions_this_step", 0)
            level2_self_selected = getattr(sat, "level2_self_selected_this_step", 0)
            self._ep_stats["level2_decisions"] += level2_decisions
            self._ep_stats["level2_self_selected"] += level2_self_selected

            # F. 队列级切片诊断
            info[sat.name]["step_slice_queue_len"] = float(
                len(getattr(sat, "slice_queue", []))
            )
            info[sat.name]["step_slice_queue_data_gbits"] = float(
                sum(
                    getattr(task, "data_size", 0.0)
                    for task in getattr(sat, "slice_queue", [])
                )
                / 1e9
            )
            info[sat.name]["step_new_slice_count"] = float(
                getattr(sat, "new_slice_count_this_step", 0)
            )
            info[sat.name]["step_new_slice_data_gbits"] = float(
                getattr(sat, "new_slice_data_this_step", 0.0) / 1e9
            )
            cand = float(getattr(sat, "level2_candidate_slices_this_step", 0))
            proc = float(getattr(sat, "level2_processed_count_this_step", 0))
            info[sat.name]["step_level2_effective_ratio"] = (
                proc / cand if cand > 0 else 0.0
            )

            # E. Level-1 分配比例（用于诊断 alpha_sats 是否长期偏低）
            # P2 Fix: 只在 set_action 被调用后（_alpha_updated=True）才累加，
            # 避免非 retasking 步重复读取旧值导致 episode 均值被"拉平"。
            level1_ratios = getattr(sat, "last_level1_ratios", None)
            alpha_updated = getattr(sat, "_alpha_updated", True)  # 向后兼容默认True
            if level1_ratios is not None and len(level1_ratios) >= 2:
                alpha_ud = float(level1_ratios[0])
                alpha_cloud = float(level1_ratios[1])
                alpha_sats = np.asarray(level1_ratios[2:], dtype=np.float64)
                alpha_sats_sum = float(np.sum(alpha_sats)) if alpha_sats.size > 0 else 0.0
                alpha_sats_mean = float(np.mean(alpha_sats)) if alpha_sats.size > 0 else 0.0
                info[sat.name]["step_alpha_ud"] = alpha_ud
                info[sat.name]["step_alpha_cloud"] = alpha_cloud
                info[sat.name]["step_alpha_sats_sum"] = alpha_sats_sum
                info[sat.name]["step_alpha_sats_mean"] = alpha_sats_mean
                if alpha_updated:
                    self._ep_stats["alpha_ud_sum"] += alpha_ud
                    self._ep_stats["alpha_cloud_sum"] += alpha_cloud
                    self._ep_stats["alpha_sats_sum"] += alpha_sats_sum
                    self._ep_stats["alpha_ud_sum_sq"] += alpha_ud * alpha_ud
                    self._ep_stats["alpha_cloud_sum_sq"] += alpha_cloud * alpha_cloud
                    self._ep_stats["alpha_sats_sum_sq"] += alpha_sats_sum * alpha_sats_sum
                    self._ep_stats["alpha_steps"] += 1
                    sat._alpha_updated = False  # 重置标记，等待下次 set_action
                is_hier = hasattr(sat, "process_slice_queue_hier")
                if is_hier and len(level1_ratios) >= 4:
                    alpha_sat_local = float(level1_ratios[2])
                    alpha_sat_offload = float(level1_ratios[3])
                    info[sat.name]["step_alpha_sat_local"] = alpha_sat_local
                    info[sat.name]["step_alpha_sat_offload"] = alpha_sat_offload
                    if alpha_updated:
                        self._ep_stats["alpha_sat_local_sum"] += alpha_sat_local
                        self._ep_stats["alpha_sat_offload_sum"] += alpha_sat_offload
                        self._ep_stats["alpha_sat_local_sum_sq"] += (
                            alpha_sat_local * alpha_sat_local
                        )
                        self._ep_stats["alpha_sat_offload_sum_sq"] += (
                            alpha_sat_offload * alpha_sat_offload
                        )

            # F. 卸载链路统计（含首跳与中继）
            current_offloaded = getattr(sat, "offloaded_data_total", 0.0)
            prev_offloaded = self._ep_stats["prev_offloaded_total"].get(sat.name, 0.0)
            step_offloaded = max(current_offloaded - prev_offloaded, 0.0)
            self._ep_stats["prev_offloaded_total"][sat.name] = current_offloaded

            current_first_hop = getattr(sat, "first_hop_offloaded", 0.0)
            prev_first_hop = self._ep_stats["prev_first_hop_offloaded_total"].get(sat.name, 0.0)
            step_first_hop = max(current_first_hop - prev_first_hop, 0.0)
            self._ep_stats["prev_first_hop_offloaded_total"][sat.name] = current_first_hop

            current_relay = getattr(sat, "relay_offloaded", 0.0)
            prev_relay = self._ep_stats["prev_relay_offloaded_total"].get(sat.name, 0.0)
            step_relay = max(current_relay - prev_relay, 0.0)
            self._ep_stats["prev_relay_offloaded_total"][sat.name] = current_relay

            if step_offloaded > 0:
                relay_ratio = step_relay / step_offloaded
            else:
                relay_ratio = 0.0
            info[sat.name]["step_offloaded_gbits"] = step_offloaded / 1e9
            info[sat.name]["step_first_hop_offloaded_gbits"] = step_first_hop / 1e9
            info[sat.name]["step_relay_offloaded_gbits"] = step_relay / 1e9
            info[sat.name]["step_relay_contribution"] = relay_ratio
            self._ep_stats["offloaded_data_bits"] += step_offloaded
            self._ep_stats["first_hop_offloaded_bits"] += step_first_hop
            self._ep_stats["relay_offloaded_bits"] += step_relay

            # G. 奖励分解（用于判断完成奖励是否被成本淹没）
            if reward_components and sat.name in reward_components:
                comp = reward_components[sat.name]
                completion_reward = float(comp.get("completion_reward", 0.0))
                expired_penalty = float(comp.get("expired_penalty", 0.0))
                energy_cost = float(comp.get("energy_cost", 0.0))
                queue_penalty = float(comp.get("queue_penalty", 0.0))
                slice_drain_reward = float(comp.get("slice_drain_reward", 0.0))
                progress_reward = float(comp.get("progress_reward", 0.0))
                offload_instant_reward = float(comp.get("offload_instant_reward", 0.0))
                efficiency_reward = float(comp.get("efficiency_reward", 0.0))
                total_reward = float(comp.get("total_reward", 0.0))
                cost_total = expired_penalty + energy_cost + queue_penalty

                info[sat.name]["step_reward_completion"] = completion_reward
                info[sat.name]["step_reward_cost_total"] = cost_total
                info[sat.name]["step_reward_completion_minus_cost"] = completion_reward - cost_total
                info[sat.name]["step_reward_total"] = total_reward
                info[sat.name]["step_reward_progress"] = progress_reward
                info[sat.name]["step_reward_slice_drain"] = slice_drain_reward
                info[sat.name]["step_reward_offload_instant"] = offload_instant_reward
                info[sat.name]["step_reward_efficiency"] = efficiency_reward

                self._ep_stats["reward_completion_sum"] += completion_reward
                self._ep_stats["reward_cost_sum"] += cost_total
                self._ep_stats["reward_total_sum"] += total_reward
        
        # 更新 Episode 累积计数器
        self._episode_energy_consumed += step_energy_total
        self._episode_completed_tasks += step_completed_count
        
        # 统计过期任务 (同理处理 expired_buffer)
        step_expired_count = 0
        for sat in self.satellites:
            if hasattr(sat, 'expired_tasks_buffer'):
                for task in sat.expired_tasks_buffer:
                    step_expired_count += 1
                    task_id = getattr(task, 'task_id', id(task))  # 获取任务ID
                    
                    # ✅ 任务级失败：任意切片过期即视为任务过期
                    self._ep_stats["all_expired_task_ids"].add(task_id)
                    
                    # 检查是否是协作失败 (access_sat != sat or hop > 0)
                    access_sat = getattr(task, 'access_satellite', '')
                    is_collab = (access_sat and access_sat != sat.name) or getattr(task, 'hop_count', 0) > 0
                    if is_collab:
                        # ✅ 使用 set.add() 确保每个任务只统计一次
                        self._ep_stats["collab_failed_ids"].add(task_id)
                
                # 清空缓冲区
                sat.expired_tasks_buffer.clear()
        
        self._episode_expired_tasks += step_expired_count

        # ========================================
        # 计算基础指标 - 统一使用 _ep_stats 中的 set（确保分子分母一致）
        # ========================================
        # 指标含义与公式速查（论文可读性）：
        # - ep_completion_rate = completed / (completed + expired)
        #   任务成功率，衡量系统是否“能完成”。
        # - ep_energy_efficiency = completed_data_bits / total_energy
        #   能效比，衡量单位能耗完成的数据量（只算成功完成任务的数据量）。
        # - ep_load_balance_jain = (sum x)^2 / (n * sum x^2)
        #   Jain 公平指数，x 为各卫星处理数据量；越接近 1 越均衡。
        # - ep_avg_hops = total_hops / total_completed
        #   平均协作跳数，反映协作深度/网络开销。
        # - ep_collab_contribution = remote_completed / total_completed
        #   协作贡献率，异地完成任务占比（越高说明协作更充分）。
        # - ep_collab_failure_ratio = collab_failed / total_expired
        #   协作失败率，过期任务中因协作失败的占比。
        # - ep_reward_total_per_arrived = reward_total_sum / total_arrived
        #   归一化总奖励，削弱泊松到达波动对奖励曲线的影响。
        # 
        # 修复 Bug：之前 remote_count 用 _ep_stats 统计，但 total_completed 用卫星的
        # completed_task_ids 统计，导致数据源不一致，ratio > 1.0
        # 
        # 现在改为：统一使用 _ep_stats 中从 buffer 收集的任务 ID
        # 
        # ⚠️ 任务失败判定：任意切片过期即任务失败
        expired_ids = self._ep_stats["all_expired_task_ids"]
        completed_ids = self._ep_stats["all_completed_task_ids"] - expired_ids
        
        total_completed = len(completed_ids)
        total_expired = len(expired_ids)
        total_tasks = total_completed + total_expired
        
        # 1. System Performance
        info["ep_completion_rate"] = total_completed / total_tasks if total_tasks > 0 else 0.0
        
        # 2. QoS (Latency) - 使用 _ep_stats 中的 latencies 列表
        latencies = self._ep_stats["latencies"]
        if len(latencies) > 0:
            info["ep_avg_latency"] = float(np.mean(latencies))
            info["ep_latency_p95"] = float(np.percentile(latencies, 95))
            info["ep_latency_max"] = float(np.max(latencies))
        else:
            info["ep_avg_latency"] = 0.0
            info["ep_latency_p95"] = 0.0
            info["ep_latency_max"] = 0.0

        # 3. Efficiency (Energy)
        # ✅ 修复：使用「成功完成任务的数据量」而不是「处理过的数据量」
        # 这样能效指标与完成率正相关，避免「虚高能效」问题
        # - 之前：processed_data_total 包含所有处理过的数据（含未完成的）
        # - 现在：completed_data_total 只包含成功完成的任务数据量
        completed_data_bits = self._ep_stats["completed_data_total"]
        info["ep_energy_efficiency"] = (
            completed_data_bits / self._episode_energy_consumed 
            if self._episode_energy_consumed > 0 else 0.0
        )
        
        # 3.1 Energy Breakdown (Comm vs Comp)
        total_energy_breakdown = self._ep_stats["energy_compute"] + self._ep_stats["energy_comm"]
        info["ep_energy_comm_ratio"] = (
            self._ep_stats["energy_comm"] / total_energy_breakdown
            if total_energy_breakdown > 0 else 0.0
        )
        
        # 4. Collaboration Metrics
        # Jain's Fairness Index
        sat_loads = list(self._ep_stats["sat_processed_data"].values())
        if len(sat_loads) > 0 and sum(sat_loads) > 0:
            sum_x = sum(sat_loads)
            sum_x_sq = sum(x**2 for x in sat_loads)
            n = len(sat_loads)
            info["ep_load_balance_jain"] = (sum_x ** 2) / (n * sum_x_sq)
        else:
            info["ep_load_balance_jain"] = 1.0 # 空载视为均衡
            
        # Hop Metrics (使用 _ep_stats)
        info["ep_avg_hops"] = (
            self._ep_stats["total_hops"] / total_completed 
            if total_completed > 0 else 0.0
        )
        
        # Collab Contribution (使用 _ep_stats 中的 set 长度)
        # ✅ 现在是唯一任务数 / 唯一任务数，必然 <= 1
        remote_count = len(self._ep_stats["remote_task_ids"])
        info["ep_collab_contribution"] = (
            remote_count / total_completed
            if total_completed > 0 else 0.0
        )
        
        # Collab Failure Ratio (使用 _ep_stats 中的 set 长度)
        # 只统计任务级失败中的协作失败
        collab_failed_ids = self._ep_stats["collab_failed_ids"] & expired_ids
        collab_failed_count = len(collab_failed_ids)
        info["ep_collab_failure_ratio"] = (
            collab_failed_count / total_expired
            if total_expired > 0 else 0.0
        )
        
        # ========================================
        # 🆕 论文风格指标 (用于 Result 模块画图)
        # ========================================
        
        # A. 价值密度：平均完成任务优先级 (Avg Priority of Completed Tasks)
        # 证明 Agent 学会了优先保住 VIP 任务
        completed_priorities = self._ep_stats["completed_priorities"]
        info["ep_avg_completed_priority"] = (
            float(np.mean(completed_priorities)) if completed_priorities else 0.5
        )
        
        # B. 卸载效率：尝试次数 vs 成功率 (Quality over Quantity)
        # 证明 Agent 学会了"拒绝垃圾任务"，不做无效卸载
        first_hop_task_ids: set = set()
        relay_task_ids: set = set()
        result_relay_task_ids: set = set()
        for sat in self.satellites:
            first_hop_task_ids.update(getattr(sat, "offloaded_task_ids", set()))
            relay_task_ids.update(getattr(sat, "relay_task_ids", set()))
            result_relay_task_ids.update(
                getattr(sat, "result_relay_task_ids", set())
            )
        data_offload_task_ids = first_hop_task_ids | relay_task_ids
        self._ep_stats["offloaded_task_ids"] = data_offload_task_ids
        offload_successes = len(completed_ids & data_offload_task_ids)
        first_hop_successes = len(completed_ids & first_hop_task_ids)
        relay_offload_successes = len(completed_ids & relay_task_ids)
        result_relay_successes = len(completed_ids & result_relay_task_ids)
        result_relay_only_successes = len(
            (completed_ids & result_relay_task_ids) - data_offload_task_ids
        )
        offload_attempts = self._ep_stats["offload_attempts"]
        info["ep_offload_attempts"] = offload_attempts
        info["ep_offload_successes"] = offload_successes
        info["ep_offload_success_rate"] = (
            offload_successes / offload_attempts if offload_attempts > 0 else 0.0
        )
        level2_decisions = self._ep_stats["level2_decisions"]
        info["ep_level2_self_ratio"] = (
            self._ep_stats["level2_self_selected"] / level2_decisions
            if level2_decisions > 0 else 0.0
        )
        info["ep_first_hop_offload_successes"] = first_hop_successes
        info["ep_relay_offload_successes"] = relay_offload_successes
        info["ep_result_relay_successes"] = result_relay_successes
        info["ep_result_relay_only_successes"] = result_relay_only_successes

        shield_interventions_total = 0
        routing_execution_modes = []
        level2_condition_modes = []
        slice_queue_len_values = []
        slice_queue_data_values = []
        new_slice_count_values = []
        new_slice_data_values = []
        level2_effective_values = []
        for sat in self.satellites:
            routing_execution_modes.append(
                str(getattr(sat, "routing_execution_mode", "delayed"))
            )
            level2_condition_modes.append(
                str(getattr(self, "level2_condition_mode", "auto"))
            )
            slice_queue_len_values.append(float(len(getattr(sat, "slice_queue", []))))
            slice_queue_data_values.append(
                float(
                    sum(
                        getattr(task, "data_size", 0.0)
                        for task in getattr(sat, "slice_queue", [])
                    )
                    / 1e9
                )
            )
            new_slice_count_values.append(
                float(getattr(sat, "new_slice_count_this_step", 0))
            )
            new_slice_data_values.append(
                float(getattr(sat, "new_slice_data_this_step", 0.0) / 1e9)
            )
            cand = float(getattr(sat, "level2_candidate_slices_this_step", 0))
            proc = float(getattr(sat, "level2_processed_count_this_step", 0))
            level2_effective_values.append(proc / cand if cand > 0 else 0.0)
            action_builder = getattr(sat, "action_builder", None)
            action_spec = getattr(action_builder, "action_spec", None)
            if action_spec is None:
                continue
            for act in action_spec:
                shield_interventions_total += getattr(act, "shield_interventions", 0)
        info["ep_shield_interventions"] = shield_interventions_total
        info["ep_slice_queue_len"] = (
            float(np.mean(slice_queue_len_values)) if slice_queue_len_values else 0.0
        )
        info["ep_slice_queue_data_gbits"] = (
            float(np.mean(slice_queue_data_values)) if slice_queue_data_values else 0.0
        )
        info["ep_new_slice_count"] = (
            float(np.mean(new_slice_count_values)) if new_slice_count_values else 0.0
        )
        info["ep_new_slice_data_gbits"] = (
            float(np.mean(new_slice_data_values)) if new_slice_data_values else 0.0
        )
        info["ep_level2_effective_ratio"] = (
            float(np.mean(level2_effective_values)) if level2_effective_values else 0.0
        )
        info["ep_routing_execution_mode"] = (
            1.0
            if any(mode == "immediate" for mode in routing_execution_modes)
            else 0.0
        )
        info["ep_level2_condition_mode"] = (
            1.0
            if any(mode == "chain" for mode in level2_condition_modes)
            else 0.0
        )

        # Level-1 分配统计（Episode 平均）
        alpha_steps = self._ep_stats["alpha_steps"]
        def _calc_std(sum_val: float, sum_sq: float, steps: int) -> float:
            if steps <= 0:
                return 0.0
            mean_val = sum_val / steps
            var = max(sum_sq / steps - mean_val * mean_val, 0.0)
            return float(np.sqrt(var))

        info["ep_alpha_ud_mean"] = (
            self._ep_stats["alpha_ud_sum"] / alpha_steps if alpha_steps > 0 else 0.0
        )
        info["ep_alpha_cloud_mean"] = (
            self._ep_stats["alpha_cloud_sum"] / alpha_steps if alpha_steps > 0 else 0.0
        )
        info["ep_alpha_sats_mean"] = (
            self._ep_stats["alpha_sats_sum"] / alpha_steps if alpha_steps > 0 else 0.0
        )
        info["ep_alpha_sat_local_mean"] = (
            self._ep_stats["alpha_sat_local_sum"] / alpha_steps if alpha_steps > 0 else 0.0
        )
        info["ep_alpha_sat_offload_mean"] = (
            self._ep_stats["alpha_sat_offload_sum"] / alpha_steps if alpha_steps > 0 else 0.0
        )
        info["ep_alpha_ud_std"] = _calc_std(
            self._ep_stats["alpha_ud_sum"],
            self._ep_stats["alpha_ud_sum_sq"],
            alpha_steps,
        )
        info["ep_alpha_cloud_std"] = _calc_std(
            self._ep_stats["alpha_cloud_sum"],
            self._ep_stats["alpha_cloud_sum_sq"],
            alpha_steps,
        )
        info["ep_alpha_sats_std"] = _calc_std(
            self._ep_stats["alpha_sats_sum"],
            self._ep_stats["alpha_sats_sum_sq"],
            alpha_steps,
        )
        info["ep_alpha_sat_local_std"] = _calc_std(
            self._ep_stats["alpha_sat_local_sum"],
            self._ep_stats["alpha_sat_local_sum_sq"],
            alpha_steps,
        )
        info["ep_alpha_sat_offload_std"] = _calc_std(
            self._ep_stats["alpha_sat_offload_sum"],
            self._ep_stats["alpha_sat_offload_sum_sq"],
            alpha_steps,
        )

        # 卸载统计（Episode 累积与中继占比）
        offloaded_bits = self._ep_stats["offloaded_data_bits"]
        first_hop_bits = self._ep_stats["first_hop_offloaded_bits"]
        relay_bits = self._ep_stats["relay_offloaded_bits"]
        info["ep_offloaded_gbits"] = offloaded_bits / 1e9
        info["ep_first_hop_offloaded_gbits"] = first_hop_bits / 1e9
        info["ep_relay_offloaded_gbits"] = relay_bits / 1e9
        info["ep_relay_contribution_ratio"] = (
            relay_bits / offloaded_bits if offloaded_bits > 0 else 0.0
        )

        # 奖励分解（Episode 累积）
        completion_sum = self._ep_stats["reward_completion_sum"]
        cost_sum = self._ep_stats["reward_cost_sum"]
        info["ep_reward_completion_sum"] = completion_sum
        info["ep_reward_cost_sum"] = cost_sum
        info["ep_reward_completion_minus_cost"] = completion_sum - cost_sum

        # 奖励分布统计（原始 vs 裁剪）
        raw_list = self._ep_stats.get("reward_raw_list", [])
        clip_list = self._ep_stats.get("reward_clip_list", [])
        if raw_list:
            info["ep_reward_raw_mean"] = float(np.mean(raw_list))
            info["ep_reward_raw_p95"] = float(np.percentile(raw_list, 95))
            info["ep_reward_raw_max"] = float(np.max(raw_list))
        else:
            info["ep_reward_raw_mean"] = 0.0
            info["ep_reward_raw_p95"] = 0.0
            info["ep_reward_raw_max"] = 0.0
        if clip_list:
            info["ep_reward_clip_mean"] = float(np.mean(clip_list))
            info["ep_reward_clip_p95"] = float(np.percentile(clip_list, 95))
            info["ep_reward_clip_max"] = float(np.max(clip_list))
            clip_scale = float(getattr(self, "reward_clip_scale", 600.0))
            sat_count = sum(1 for v in clip_list if abs(v) >= 0.95 * clip_scale)
            info["ep_reward_clip_saturation"] = sat_count / max(len(clip_list), 1)
        else:
            info["ep_reward_clip_mean"] = 0.0
            info["ep_reward_clip_p95"] = 0.0
            info["ep_reward_clip_max"] = 0.0
            info["ep_reward_clip_saturation"] = 0.0
        
        # C. 时延分解 (Latency Breakdown)
        # 堆叠柱状图：展示传输/排队/计算时延的比例
        lat_tx = self._ep_stats["latency_transmission"]
        lat_queue = self._ep_stats["latency_queuing"]
        lat_comp = self._ep_stats["latency_computing"]
        
        info["ep_avg_latency_tx"] = float(np.mean(lat_tx)) if lat_tx else 0.0
        info["ep_avg_latency_queue"] = float(np.mean(lat_queue)) if lat_queue else 0.0
        info["ep_avg_latency_compute"] = float(np.mean(lat_comp)) if lat_comp else 0.0

        # --- Debug Info ---
        if step_completed_count > 0:
             logger.debug(f"[METRICS] Step completed: {step_completed_count}, Cumulative: {total_completed}, Hops: {self._ep_stats['total_hops']}")

        # 任务统计
        # 到达的总任务数 (从 scenario 获取)
        total_arrived = 0
        if hasattr(self.scenario, 'arrived_tasks'):
            total_arrived = len(self.scenario.arrived_tasks)
        elif hasattr(self.scenario, 'total_tasks_arrived'):
            total_arrived = self.scenario.total_tasks_arrived
        
        # 已处理的任务数 = 已完成 + 已过期（使用ID集合，确保唯一）
        total_processed = total_completed + total_expired
        
        # 未处理任务数 = 已到达 - 已处理
        # 这确保等式成立: ep_arrived = ep_completed + ep_expired + ep_pending
        pending_count = max(0, total_arrived - total_processed)
        
        # ⚠️ 调试：检查等式是否满足
        if total_processed > total_arrived:
            logger.warning(
                f"[METRICS BUG] Processed > Arrived! "
                f"arrived={total_arrived}, completed={total_completed}, "
                f"expired={total_expired}, processed={total_processed}, "
                f"pending={pending_count}"
            )
        
        # 任务池剩余 (尚未到达的)
        pool_remaining = 0
        if hasattr(self.scenario, 'task_pool'):
            pool_remaining = len(self.scenario.task_pool)

        # 保留原有的内部指标
        info["_step_energy"] = step_energy_total
        info["ep_completed"] = total_completed
        info["ep_expired"] = total_expired
        info["ep_arrived"] = total_arrived  # 已到达的总任务数
        info["ep_pending"] = pending_count  # 待处理任务数 = 已到达 - 已处理
        info["ep_pool_remaining"] = pool_remaining  # 任务池剩余
        info["d_ts"] = self.latest_step_duration
        # 归一化奖励：削弱泊松到达波动对奖励的影响
        reward_total_sum = self._ep_stats.get("reward_total_sum", 0.0)
        info["ep_reward_total_per_arrived"] = (
            reward_total_sum / total_arrived if total_arrived > 0 else 0.0
        )
        # 课程学习阶段（用于面板诊断）
        if hasattr(self, "_curriculum_state"):
            info["ep_curriculum_stage"] = float(
                self._curriculum_state.get("stage", 0)
            )
            info["ep_curriculum_difficulty"] = float(
                self._curriculum_state.get("difficulty", 0.0)
            )

        self._update_curriculum_stats(info)
        self._maybe_apply_curriculum(info)
        return info

    def _iter_rewarders(self) -> list[Any]:
        rewarder = self.rewarder
        if hasattr(rewarder, "rewarders"):
            return list(rewarder.rewarders)
        return [rewarder]

    def _apply_curriculum_updates(self, cfg: dict[str, Any], stage: int, completion: float) -> None:
        scenario = self.scenario
        if stage in (0, 1):
            if stage == 0:
                arrival_mult = float(
                    cfg.get("arrival_rate_mult_stage1", cfg.get("arrival_rate_mult", 1.0))
                )
                workload_mult = float(
                    cfg.get("workload_max_mult_stage1", cfg.get("workload_max_mult", 1.0))
                )
                delay_mult = float(
                    cfg.get("max_delay_max_mult_stage1", cfg.get("max_delay_max_mult", 1.0))
                )
                stage_name = "Stage-1"
            else:
                arrival_mult = float(
                    cfg.get(
                        "arrival_rate_mult_stage2",
                        cfg.get("arrival_rate_mult_stage1", cfg.get("arrival_rate_mult", 1.0)),
                    )
                )
                workload_mult = float(
                    cfg.get(
                        "workload_max_mult_stage2",
                        cfg.get("workload_max_mult_stage1", cfg.get("workload_max_mult", 1.0)),
                    )
                )
                delay_mult = float(
                    cfg.get(
                        "max_delay_max_mult_stage2",
                        cfg.get(
                            "max_delay_max_mult_stage1",
                            cfg.get("max_delay_max_mult", 1.0),
                        ),
                    )
                )
                stage_name = "Stage-2"

            if hasattr(scenario, "task_arrival_rate"):
                scenario.task_arrival_rate = max(0.0, scenario.task_arrival_rate * arrival_mult)
            if hasattr(scenario, "workload_range"):
                w_min, w_max = scenario.workload_range
                new_max = max(w_min, w_max * workload_mult)
                scenario.workload_range = (w_min, new_max)
            if hasattr(scenario, "max_delay_range"):
                d_min, d_max = scenario.max_delay_range
                new_max = max(d_min, d_max * delay_mult)
                scenario.max_delay_range = (d_min, new_max)

            logger.warning(
                "[Curriculum] %s difficulty up: completion=%.3f arrival=%.3f workload_max=%.3f max_delay_max=%.3f",
                stage_name,
                completion,
                getattr(scenario, "task_arrival_rate", -1.0),
                getattr(scenario, "workload_range", (-1.0, -1.0))[1],
                getattr(scenario, "max_delay_range", (-1.0, -1.0))[1],
            )
            return

        if stage == 2:
            penalty_mult = float(
                cfg.get("penalty_mult_stage3", cfg.get("penalty_mult", 1.0))
            )
            for rewarder in self._iter_rewarders():
                for attr in ("w_delay", "w_queue_penalty", "w_energy"):
                    if hasattr(rewarder, attr):
                        setattr(rewarder, attr, getattr(rewarder, attr) * penalty_mult)

            logger.warning(
                "[Curriculum] Stage-3 penalties up: completion=%.3f mult=%.3f",
                completion,
                penalty_mult,
            )

    def _compute_curriculum_difficulty(self, cfg: dict[str, Any]) -> float:
        state = getattr(self, "_curriculum_state", {})
        axis = str(cfg.get("axis", "episodes")).lower()
        warmup_episodes = int(cfg.get("warmup_episodes", 0))
        warmup_steps = int(cfg.get("warmup_steps", 0))
        total_episodes = int(cfg.get("total_episodes", 1))
        total_steps = int(cfg.get("total_steps", 1))
        if axis == "steps":
            progress_value = int(state.get("total_steps", 0)) - warmup_steps
            denom = max(total_steps, 1)
        else:
            progress_value = int(state.get("episode_idx", 0)) - warmup_episodes
            denom = max(total_episodes, 1)
        progress = max(0.0, float(progress_value) / float(denom))
        progress = min(progress, 1.0)

        schedule = str(cfg.get("schedule", "linear")).lower()
        if schedule == "sqrt":
            progress = float(np.sqrt(progress))
        elif schedule == "sigmoid":
            k = float(cfg.get("schedule_k", 6.0))
            progress = float(1.0 / (1.0 + np.exp(-k * (progress - 0.5))))

        diff_min = float(cfg.get("difficulty_min", 0.0))
        diff_max = float(cfg.get("difficulty_max", 1.0))
        difficulty = diff_min + (diff_max - diff_min) * progress
        return float(np.clip(difficulty, min(diff_min, diff_max), max(diff_min, diff_max)))

    def _apply_curriculum_function(self, cfg: dict[str, Any], difficulty: float) -> None:
        scenario = self.scenario
        state = getattr(self, "_curriculum_state", {})
        base = state.get("base_params")
        if not isinstance(base, dict):
            base = {
                "task_arrival_rate": getattr(scenario, "task_arrival_rate", None),
                "workload_range": getattr(scenario, "workload_range", None),
                "max_delay_range": getattr(scenario, "max_delay_range", None),
            }
            state["base_params"] = base

        def _lerp(start: float, end: float) -> float:
            return float(start + (end - start) * difficulty)

        arrival_mult = _lerp(
            float(cfg.get("arrival_rate_mult_start", 1.0)),
            float(cfg.get("arrival_rate_mult_end", 1.0)),
        )
        if base.get("task_arrival_rate") is not None and hasattr(
            scenario, "task_arrival_rate"
        ):
            scenario.task_arrival_rate = max(0.0, base["task_arrival_rate"] * arrival_mult)

        workload_range = base.get("workload_range")
        if (
            workload_range is not None
            and hasattr(scenario, "workload_range")
            and len(workload_range) == 2
        ):
            w_min, w_max = workload_range
            workload_mult = _lerp(
                float(cfg.get("workload_max_mult_start", 1.0)),
                float(cfg.get("workload_max_mult_end", 1.0)),
            )
            scenario.workload_range = (w_min, max(w_min, w_max * workload_mult))

        delay_range = base.get("max_delay_range")
        if (
            delay_range is not None
            and hasattr(scenario, "max_delay_range")
            and len(delay_range) == 2
        ):
            d_min, d_max = delay_range
            delay_mult = _lerp(
                float(cfg.get("max_delay_max_mult_start", 1.0)),
                float(cfg.get("max_delay_max_mult_end", 1.0)),
            )
            scenario.max_delay_range = (d_min, max(d_min, d_max * delay_mult))

        if bool(cfg.get("adjust_penalties", False)):
            penalty_mult = _lerp(
                float(cfg.get("penalty_mult_start", 1.0)),
                float(cfg.get("penalty_mult_end", 1.0)),
            )
            base_penalties = state.get("base_penalties")
            if not isinstance(base_penalties, list):
                base_penalties = []
                for rewarder in self._iter_rewarders():
                    entry = {"rewarder": rewarder}
                    for attr in ("w_delay", "w_queue_penalty", "w_energy"):
                        if hasattr(rewarder, attr):
                            entry[attr] = getattr(rewarder, attr)
                    base_penalties.append(entry)
                state["base_penalties"] = base_penalties
            for entry in base_penalties:
                rewarder = entry.get("rewarder")
                if rewarder is None:
                    continue
                for attr in ("w_delay", "w_queue_penalty", "w_energy"):
                    if attr in entry:
                        setattr(rewarder, attr, entry[attr] * penalty_mult)

        logger.warning(
            "[Curriculum-F] difficulty=%.3f arrival=%.3f workload_max=%.3f max_delay_max=%.3f",
            difficulty,
            getattr(scenario, "task_arrival_rate", -1.0),
            getattr(scenario, "workload_range", (-1.0, -1.0))[1],
            getattr(scenario, "max_delay_range", (-1.0, -1.0))[1],
        )

    def _maybe_apply_curriculum(self, info: dict[str, Any]) -> None:
        cfg = getattr(self, "curriculum_config", None)
        if not cfg or not cfg.get("enable", False):
            return
        state = getattr(self, "_curriculum_state", {})

        mode = str(cfg.get("mode", cfg.get("curriculum_mode", "staged"))).lower()
        axis = str(cfg.get("axis", "episodes")).lower()
        if mode in ("function", "continuous") and axis == "steps":
            interval = int(cfg.get("update_interval_steps", 1))
            interval = max(interval, 1)
            total_steps = int(state.get("total_steps", 0))
            last_applied = int(state.get("last_step_applied", 0))
            if total_steps - last_applied >= interval:
                difficulty = self._compute_curriculum_difficulty(cfg)
                self._apply_curriculum_function(cfg, float(difficulty))
                state["difficulty"] = float(difficulty)
                state["last_step_applied"] = total_steps
            return

        if not getattr(self, "_last_episode_done", False):
            return

        if state.get("applied_for_episode"):
            return

        mode = str(cfg.get("mode", cfg.get("curriculum_mode", "staged"))).lower()
        if mode in ("function", "continuous"):
            warmup_episodes = int(cfg.get("warmup_episodes", 0))
            episode_idx = int(state.get("episode_idx", 0))
            if warmup_episodes > 0 and episode_idx <= warmup_episodes:
                state["applied_for_episode"] = True
                return

            difficulty = self._compute_curriculum_difficulty(cfg)
            self._apply_curriculum_function(cfg, float(difficulty))
            state["difficulty"] = float(difficulty)
            state["applied_for_episode"] = True
            return

        warmup_episodes = int(cfg.get("warmup_episodes", 0))
        episode_idx = int(state.get("episode_idx", 0))
        if warmup_episodes > 0 and episode_idx <= warmup_episodes:
            state["applied_for_episode"] = True
            return

        completion = info.get("ep_completion_rate")
        if completion is None:
            state["applied_for_episode"] = True
            return

        thresholds = cfg.get("thresholds", [])
        if not thresholds:
            state["applied_for_episode"] = True
            return

        max_stages = int(cfg.get("max_stages", len(thresholds)))
        stage = int(state.get("stage", 0))
        if stage >= max_stages or stage >= len(thresholds):
            state["applied_for_episode"] = True
            return

        patience = int(cfg.get("patience", 1))
        if completion >= float(thresholds[stage]):
            state["stable_count"] = int(state.get("stable_count", 0)) + 1
        else:
            state["stable_count"] = 0

        if state["stable_count"] >= max(patience, 1):
            self._apply_curriculum_updates(cfg, stage, float(completion))
            state["stage"] = stage + 1
            state["stable_count"] = 0
            table = self._format_curriculum_stage_table()
            if table:
                logger.info(table)

        state["applied_for_episode"] = True

    def _get_reward(self):
        """Return a scalar reward for the step.
        
        奖励归一化策略：
        - 除以本步到达任务数（+ 平滑项），消除泊松到达随机性
        - 使相同策略在不同到达量下获得稳定奖励
        - 让 Critic 更容易学习 value function
        """
        reward = sum(self.reward_dict.values())
        for satellite in self.satellites:
            alive = satellite.is_alive(log_failure=True)
            if not alive:
                if not hasattr(self, "_death_snapshot_logged"):
                    self._death_snapshot_logged = False
                if not self._death_snapshot_logged:
                    self._death_snapshot_logged = True
                    dyn = getattr(satellite, "dynamics", None)
                    try:
                        battery_charge = (
                            dyn.battery_charge if dyn is not None else None
                        )
                    except Exception:
                        battery_charge = None
                    try:
                        battery_frac = (
                            dyn.battery_charge_fraction if dyn is not None else None
                        )
                    except Exception:
                        battery_frac = None
                    try:
                        r_norm = (
                            float(np.linalg.norm(dyn.r_BN_N))
                            if dyn is not None and hasattr(dyn, "r_BN_N")
                            else None
                        )
                    except Exception:
                        r_norm = None
                    try:
                        wheel_speeds = (
                            dyn.wheel_speeds if dyn is not None else None
                        )
                    except Exception:
                        wheel_speeds = None
                    max_wheel_speed = (
                        getattr(dyn, "maxWheelSpeed", None) if dyn is not None else None
                    )
                    min_orbital_radius = (
                        getattr(dyn, "min_orbital_radius", None)
                        if dyn is not None
                        else None
                    )
                    logger.warning(
                        "[DEATH SNAPSHOT] sat=%s t=%.1fs battery=%s battery_frac=%s "
                        "r_norm=%s min_orbital_radius=%s wheel_speeds=%s maxWheelSpeed=%s",
                        satellite.name,
                        self.simulator.sim_time,
                        battery_charge,
                        battery_frac,
                        r_norm,
                        min_orbital_radius,
                        wheel_speeds,
                        max_wheel_speed,
                    )
                reward += self.failure_penalty
        
        # ✅ NaN/Inf 保护：防止异常奖励值传播到策略梯度
        if np.isnan(reward) or np.isinf(reward):
            logger.error(f"[REWARD BUG] reward={reward}, resetting to 0.0")
            reward = 0.0

        # 记录裁剪前的原始奖励（用于诊断尺度）
        raw_reward = float(reward)
        if hasattr(self, "_ep_stats") and isinstance(self._ep_stats, dict):
            self._ep_stats.setdefault("reward_raw_list", []).append(raw_reward)
        
        # ✅ 步级别奖励归一化：消除泊松到达的随机性
        # 策略：reward / max(step_arrived, 1) 
        # 这样奖励变成"每任务平均奖励"，消除到达数量波动
        if getattr(self, "reward_normalize_by_arrival", True):
            step_arrived = getattr(self, "_step_arrived_tasks", 0)
            # 🛠️ 简化归一化：直接除以到达数（最小为1避免除零）
            # 不再使用 smooth_factor，让奖励信号更清晰
            normalize_factor = max(step_arrived, 1)
            reward = reward / normalize_factor
            # 记录归一化因子用于诊断
            if hasattr(self, "_ep_stats") and isinstance(self._ep_stats, dict):
                self._ep_stats.setdefault("reward_norm_factors", []).append(normalize_factor)
        
        # ✅ 软裁剪（可选）：归一化后奖励已稳定，默认关闭
        # 若需要启用，设置 reward_soft_clip_enable: true
        if getattr(self, "reward_soft_clip_enable", False):
            soft_clip_scale = float(getattr(self, "reward_clip_scale", 50.0))
            reward = soft_clip_scale * np.tanh(reward / soft_clip_scale)

        # 记录最终奖励（用于诊断）
        if hasattr(self, "_ep_stats") and isinstance(self._ep_stats, dict):
            self._ep_stats.setdefault("reward_final_list", []).append(float(reward))

        # 奖励缩放已移至 TorchRL Transform (common.py 中的 RewardScaling)
        # 不在此处手动缩放，避免重复
        
        return reward

    def _get_terminated(self) -> bool:
        """Return the terminated flag for the step. 截止条件成立或任务完成并且任务池空 
            is_alive()         → 物理层死亡（不可恢复，直接终止）
            is_truncated()     → 资源层耗尽（软约束，可配置阈值）
            is_terminated()    → 任务层耗尽（硬约束，任务空）
        """
        if self.terminate_on_time_limit and self._get_truncated():
            return True
        else:
            return not all(
                satellite.is_alive() and not self.rewarder.is_terminated(satellite)
                for satellite in self.satellites
            )

    def _update_curriculum_stats(self, info: dict[str, Any]) -> None:
        if not getattr(self, "_last_episode_done", False):
            return
        state = getattr(self, "_curriculum_state", {})
        stats = getattr(self, "_curriculum_stats", None)
        if stats is None:
            self._curriculum_stats = {"per_stage": {}, "last_episode_logged": -1}
            stats = self._curriculum_stats
        episode_idx = int(state.get("episode_idx", 0))
        if stats.get("last_episode_logged", -1) == episode_idx:
            return
        stage = int(state.get("stage", 0))
        per_stage = stats["per_stage"].setdefault(
            stage,
            {
                "episodes": 0,
                "arrived_sum": 0.0,
                "reward_per_arrived_sum": 0.0,
            },
        )
        per_stage["episodes"] += 1
        per_stage["arrived_sum"] += float(info.get("ep_arrived", 0.0))
        per_stage["reward_per_arrived_sum"] += float(
            info.get("ep_reward_total_per_arrived", 0.0)
        )
        stats["last_episode_logged"] = episode_idx

    def _format_curriculum_stage_table(self) -> str:
        stats = getattr(self, "_curriculum_stats", {}).get("per_stage", {})
        if not stats:
            return ""
        lines = [
            "[Curriculum] Stage summary (episode mean):",
            "stage | episodes | arrived_avg | reward_per_arrived_avg",
            "----- | -------- | ----------- | ----------------------",
        ]
        for stage in sorted(stats.keys()):
            row = stats[stage]
            episodes = row.get("episodes", 0)
            arrived_avg = (
                row.get("arrived_sum", 0.0) / episodes if episodes > 0 else 0.0
            )
            reward_avg = (
                row.get("reward_per_arrived_sum", 0.0) / episodes
                if episodes > 0
                else 0.0
            )
            lines.append(
                f"{stage:>5} | {episodes:>8} | {arrived_avg:>11.3f} | {reward_avg:>22.6f}"
            )
        return "\n".join(lines)

    def _update_curriculum_step_counters(self) -> None:
        state = getattr(self, "_curriculum_state", None)
        if not isinstance(state, dict):
            return
        state["total_steps"] = int(state.get("total_steps", 0)) + 1
        state["episode_steps"] = int(state.get("episode_steps", 0)) + 1

    def _get_truncated(self) -> bool:
        """Return the truncated flag for the step. 回合时间到达或所有卫星电量全耗尽"""
        time_limit_reached = self.simulator.sim_time >= self.time_limit
        resource_depleted = all(  # 所有卫星都耗尽才截断
            self.rewarder.is_truncated(satellite) for satellite in self.satellites
        )
        
        if time_limit_reached:
            logger.info(
                f"[TRUNCATED] Time limit reached: {self.simulator.sim_time:.1f}s / {self.time_limit}s"
            )
        elif resource_depleted:
            logger.info("[TRUNCATED] Resource depleted (battery/queue overflow)")
        
        return time_limit_reached or resource_depleted

    @property
    def action_space(self) -> spaces.Space[MultiSatAct]:
        """Compose satellite action spaces into a tuple.

        Returns:
            Joint action space
        """
        return spaces.Tuple((satellite.action_space for satellite in self.satellites))

    @property
    def observation_space(self) -> spaces.Space[MultiSatObs]:
        """Compose satellite observation spaces into a tuple.

        Note: calls ``reset()``, which can be expensive, to determine observation size.

        Returns:
            Joint observation space
        """
        try:
            self.simulator
        except AttributeError:
            logger.info("Calling env.reset() to get observation space")
            self.reset(seed=self.seed)
        return spaces.Tuple(
            [satellite.observation_space for satellite in self.satellites]
        )

    def _step(self, actions: MultiSatAct) -> None:
        logger.debug("Stepping environment with actions: %s", actions)
        
        # 处理任务到达(泊松过程) - 如果scenario是STINTaskScenario
        self._step_arrived_tasks = 0  # 初始化本步到达任务数
        if hasattr(self.scenario, 'process_task_arrivals'):
            step_duration = self.latest_step_duration if hasattr(self, 'latest_step_duration') else self.max_step_duration
            arrivals = self.scenario.process_task_arrivals(step_duration)
            self._step_arrived_tasks = arrivals  # 保存用于奖励归一化
            if arrivals > 0:
                logger.debug(f"Poisson arrival: {arrivals} tasks arrived this step")
        
        # 关键：将到达的任务分配给卫星（基于可见性检查）
        if hasattr(self.scenario, 'assign_tasks_to_satellites'):
            self.scenario.assign_tasks_to_satellites()

        
        # Reset per-step counters (avoid accumulation across steps)
        for satellite in self.satellites:
            if hasattr(satellite, "offload_attempts_this_step"):
                satellite.offload_attempts_this_step = 0
            if hasattr(satellite, "level2_decisions_this_step"):
                satellite.level2_decisions_this_step = 0
            if hasattr(satellite, "level2_self_selected_this_step"):
                satellite.level2_self_selected_this_step = 0
            if hasattr(satellite, "new_slice_count_this_step"):
                satellite.new_slice_count_this_step = 0
            if hasattr(satellite, "new_slice_data_this_step"):
                satellite.new_slice_data_this_step = 0.0
            if hasattr(satellite, "level2_processed_count_this_step"):
                satellite.level2_processed_count_this_step = 0
            if hasattr(satellite, "level2_candidate_slices_this_step"):
                satellite.level2_candidate_slices_this_step = 0
            if hasattr(satellite, "sent_data_this_step"):
                satellite.sent_data_this_step = 0.0
            if hasattr(satellite, "received_data_this_step"):
                satellite.received_data_this_step = 0.0
            if hasattr(satellite, "tx_time_this_step"):
                satellite.tx_time_this_step = 0.0
            if hasattr(satellite, "rx_time_this_step"):
                satellite.rx_time_this_step = 0.0

        if len(actions) != len(self.satellites):
            raise ValueError("There must be the same number of actions and satellites")
        # Ensure neighbor lists are available before routing decisions.
        needs_neighbors = False
        for satellite in self.satellites:
            if not hasattr(satellite, "action_builder") or not hasattr(
                satellite.action_builder, "action_spec"
            ):
                continue
            for act in satellite.action_builder.action_spec:
                if hasattr(act, "neighbor_satellites") and len(act.neighbor_satellites) == 0:
                    needs_neighbors = True
                    break
            if needs_neighbors:
                break
        if needs_neighbors:
            self._refresh_neighbor_lists()
            self._apply_neighbor_snapshot()
        for satellite, action in zip(self.satellites, actions):
            satellite.info = []  # reset satellite info log

            if not is_no_action(action):
                satellite.requires_retasking = False
                satellite.set_action(action)
            if not satellite.is_alive():
                satellite.requires_retasking = False
            else:
                # STIN/ComputationSatellite 使用 task_queue 机制，不需要传统的 retasking
                if satellite.requires_retasking and not hasattr(satellite, 'task_queue'):
                    satellite.logger.warning(
                        f"Requires retasking but received no task."
                    )


        previous_time = self.simulator.sim_time  # should these be recorded in simulator
        self.simulator.run()
        self.latest_step_duration = self.simulator.sim_time - previous_time

        # Refresh neighbor lists using the latest LOS logs before they are cleared.
        self._refresh_neighbor_lists()
        self._apply_neighbor_snapshot()

        # 关键: 执行本地计算处理 (处理 task_queue 中的任务)
        for satellite in self.satellites:
            if hasattr(satellite, 'execute_local_compute'):
                satellite.execute_local_compute(self.latest_step_duration)
        
        # ✅ 修复: 执行结果接力处理 (处理 result_queue 中的结果)
        for satellite in self.satellites:
            if hasattr(satellite, 'execute_result_relay'):
                satellite.execute_result_relay(self.latest_step_duration)

        new_data = {
            satellite.name: satellite.data_store.update_from_logs()
            for satellite in self.satellites
        }
        self.reward_dict = self.rewarder.reward(new_data)

        self.communicator.communicate()

        for satellite in self.satellites:
            if satellite.requires_retasking:
                satellite.logger.debug(f"Satellite {satellite.name} requires retasking")

    def step(
        self, actions: MultiSatAct
    ) -> tuple[MultiSatObs, float, bool, bool, dict[str, Any]]:
        """Propagate the simulation, update information, and get rewards.

        Args:
            actions: Joint action for satellites

        Returns:
            observation, reward, terminated, truncated, info
        """
        logger.debug("=== STARTING STEP ===")
        self._step(actions)
        self._update_curriculum_step_counters()

        observation = self._get_obs()
        reward = self._get_reward()
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        self._last_episode_done = bool(terminated or truncated)
        info = self._get_info()
        logger.debug(f"Step reward: {reward}")
        
        # Episode结束时输出详细统计
        if terminated or truncated:
            total_completed = sum(
                sat.completed_tasks_count for sat in self.satellites 
                if hasattr(sat, 'completed_tasks_count')
            )
            total_expired = sum(
                sat.expired_tasks_count for sat in self.satellites 
                if hasattr(sat, 'expired_tasks_count')
            )
            
            if hasattr(self.scenario, 'task_pool'):
                pool_remaining = len(self.scenario.task_pool)
                total_arrived = len(self.scenario.arrived_tasks) if hasattr(self.scenario, 'arrived_tasks') else 0
                
                logger.info(
                    f"[Episode End] Sim_time: {self.simulator.sim_time:.1f}s | "
                    f"Terminated: {terminated} | Truncated: {truncated} | "
                    f"Completed: {total_completed} | Expired: {total_expired} | "
                    f"Pool remaining: {pool_remaining} | Total arrived: {total_arrived}"
                )
            else:
                logger.info(
                    f"[Episode End] Terminated: {terminated} | Truncated: {truncated} | "
                    f"Completed: {total_completed} | Expired: {total_expired}"
                )
        else:
            logger.debug(f"Episode terminated: {terminated}")
            logger.debug(f"Episode truncated: {truncated}")
        logger.debug("Step info: %s", info)
        logger.debug("Step observation: %s", observation)
        return observation, reward, terminated, truncated, info

    def render(self) -> None:  # pragma: no cover
        """No rendering implemented."""
        return None

    def close(self) -> None:
        """Try to cleanly delete everything."""
        if hasattr(self, "simulator") and self.simulator is not None:
            del self.simulator


class SatelliteTasking(GeneralSatelliteTasking, Generic[SatObs, SatAct]):
    def __init__(self, satellite: Satellite, *args, **kwargs) -> None:
        """A special case of :class:`GeneralSatelliteTasking` for one satellite.

        For compatibility with standard training APIs, actions and observations are
        directly exposed for the single satellite as opposed to being wrapped in a
        tuple.

        Args:
            satellite: Satellite to be simulated.
            *args: Passed to :class:`GeneralSatelliteTasking`.
            **kwargs: Passed to :class:`GeneralSatelliteTasking`.
        """
        super().__init__(satellites=satellite, *args, **kwargs)
        if not len(self.satellites) == 1:
            raise ValueError(
                "SatelliteTasking must be initialized with a single satellite."
            )

    @property
    def action_space(self) -> spaces.Space[SatAct]:
        """Return the single satellite action space."""
        return self.satellite.action_space

    @property
    def observation_space(self) -> spaces.Box:
        """Return the single satellite observation space."""
        super().observation_space
        return self.satellite.observation_space

    @property
    def satellite(self) -> Satellite:
        """Satellite being tasked."""
        return self.satellites[0]

    def step(self, action) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Task the satellite with a single action."""
        return super().step([action])

    def _get_obs(self) -> Any:
        return self.satellite.get_obs()

    def _get_info(self) -> dict[str, Any]:
        info = super()._get_info()
        for k, v in info[self.satellite.name].items():
            info[k] = v
        del info[self.satellite.name]
        return info


class ConstellationTasking(
    GeneralSatelliteTasking, ParallelEnv, Generic[SatObs, SatAct, AgentID]
):
    def __init__(
        self,
        *args,
        meta_agent_groupings: Optional[dict[AgentID, list[str]]] = None,
        only_retask_idle_meta_agent_members: bool = False,
        **kwargs,
    ) -> None:
        """Implements the `PettingZoo <https://pettingzoo.farama.org>`_ parallel API for the :class:`GeneralSatelliteTasking` environment.

        Args:
            *args: Passed to :class:`GeneralSatelliteTasking`.
            meta_agent_groupings: A dictionary mapping agent names to lists of satellite names.
            only_retask_idle_meta_agent_members: If True, only satellites in a meta agent
                that require retasking will receive actions. Other actions in the meta
                agent output will be ignored. This may also be useful to control in the
                training pipeline.
            **kwargs: Passed to :class:`GeneralSatelliteTasking`.
        """
        super().__init__(*args, **kwargs)

        self.only_retask_idle_meta_agent_members = only_retask_idle_meta_agent_members

        if meta_agent_groupings is None:
            meta_agent_groupings = {}

        sats_in_meta_agents = sum(meta_agent_groupings.values(), [])
        for sat in self.satellites:
            if sat.name not in sats_in_meta_agents:
                meta_agent_groupings[sat.name] = [sat.name]

        self.meta_agent_groupings: dict[AgentID, list[Satellite]] = {
            name: [self.get_satellite(member) for member in members]
            for name, members in meta_agent_groupings.items()
        }

    def _validate_meta_agent_groupings(self):
        """Validate that meta agent groupings consist of similar action spaces."""
        for name, members in self.meta_agent_groupings.items():
            if len(members) == 0:
                raise ValueError(f"Meta agent '{name}' has no members.")
            action_space_type = type(members[0].action_space)
            for member in members:
                assert isinstance(member.action_space, action_space_type), (
                    f"Meta agent '{name}' has members with different action space types."
                )
                assert isinstance(member.observation_space, spaces.Box), (
                    f"Only Box observation spaces are supported for meta agents, "
                    f"but member '{member.name}' has {type(member.observation_space)}."
                )

    def reset(
        self, seed: int | None = None, options=None
    ) -> tuple[MultiSatObs, dict[str, Any]]:
        """Reset the environment and return PettingZoo Parallel API format."""
        self.newly_dead = []
        self._agents_last_compute_time = None
        return super().reset(seed, options)

    @property
    def agents(self) -> list[AgentID]:
        """Agents currently in the environment."""
        if (
            self._agents_last_compute_time is None
            or self._agents_last_compute_time != self.simulator.sim_time
        ):
            truncated = super()._get_truncated()
            agents = [
                agent
                for agent, satellites in self.meta_agent_groupings.items()
                if all(satellite.is_alive() for satellite in satellites)
                and not truncated
            ]
            self._agents_last_compute_time = self.simulator.sim_time
            self._agents_cache = agents
            return agents
        else:
            return self._agents_cache

    @property
    def num_agents(self) -> int:
        """Number of agents currently in the environment."""
        return len(self.agents)

    @property
    def possible_agents(self) -> list[AgentID]:
        """Return the list of all possible agents."""
        return list(self.meta_agent_groupings.keys())

    @property
    def max_num_agents(self) -> int:
        """Maximum number of agents possible in the environment."""
        return len(self.possible_agents)

    @property
    def previously_dead(self) -> list[AgentID]:
        """Return the list of agents that died at least one step ago."""
        return list(set(self.possible_agents) - set(self.agents) - set(self.newly_dead))

    @property
    def observation_spaces(self) -> dict[AgentID, spaces.Box]:
        """Return the observation space for each agent."""
        super().observation_space
        self._validate_meta_agent_groupings()

        obs_spaces = {}
        for agent, satellites in self.meta_agent_groupings.items():
            if len(satellites) == 1:
                obs_spaces[agent] = satellites[0].observation_space
            else:
                dtype = (
                    self.dtype if self.dtype else satellites[0].observation_space.dtype
                )
                low = np.concatenate(
                    [sat.observation_space.low for sat in satellites]
                )
                high = np.concatenate(
                    [sat.observation_space.high for sat in satellites]
                )
                if dtype is not None:
                    low = low.astype(dtype, copy=False)
                    high = high.astype(dtype, copy=False)
                obs_spaces[agent] = spaces.Box(
                    low=low,
                    high=high,
                    dtype=dtype,
                )
        return obs_spaces

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID) -> spaces.Space[SatObs]:
        """Return the observation space for a certain agent."""
        return self.observation_spaces[agent]

    @property
    def action_spaces(self) -> dict[AgentID, spaces.Space[SatAct]]:
        """Return the action space for each agent."""
        act_spaces = {}
        for agent, satellites in self.meta_agent_groupings.items():
            if len(satellites) == 1:
                act_spaces[agent] = satellites[0].action_space
            else:
                if isinstance(satellites[0].action_space, spaces.Discrete):
                    act_spaces[agent] = spaces.MultiDiscrete(
                        [sat.action_space.n for sat in satellites]
                    )
                elif isinstance(satellites[0].action_space, spaces.Box):
                    dtype = (
                        self.dtype if self.dtype else satellites[0].action_space.dtype
                    )
                    low = np.concatenate([sat.action_space.low for sat in satellites])
                    high = np.concatenate([sat.action_space.high for sat in satellites])
                    if dtype is not None:
                        low = low.astype(dtype, copy=False)
                        high = high.astype(dtype, copy=False)
                    act_spaces[agent] = spaces.Box(
                        low=low,
                        high=high,
                        dtype=dtype,
                    )
        return act_spaces

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID) -> spaces.Space[SatAct]:
        """Return the action space for a certain agent."""
        return self.action_spaces[agent]

    def _requires_retasking(self, agent: AgentID) -> bool:
        """Check if the agent requires retasking."""
        return any(
            satellite.requires_retasking
            for satellite in self.meta_agent_groupings[agent]
        )

    def _get_obs(self) -> dict[AgentID, SatObs]:
        """Format the observation per the PettingZoo Parallel API."""
        obs = {}
        for agent, satellites in self.meta_agent_groupings.items():
            # 始终为所有 agents 生成观测
            # 对于 dead agents，返回零向量以避免 TorchRL KeyError
            if agent in self.previously_dead:
                # 返回零向量观测
                agent_obs = [
                    satellite.observation_space.low * 0 for satellite in satellites
                ]
            elif self.generate_obs_retasking_only and not self._requires_retasking(agent):
                agent_obs = [
                    satellite.observation_space.low * 0 for satellite in satellites
                ]
            else:
                agent_obs = [satellite.get_obs() for satellite in satellites]

            if len(agent_obs) == 1:
                obs[agent] = agent_obs[0]
            else:
                obs[agent] = np.concatenate(agent_obs)

        return obs

    def state(self) -> np.ndarray:
        """Return the global state for centralized training (CTDE).

        Returns a concatenation of all agents' observations, providing
        a global view appropriate for centralized critics (e.g., MAPPO, QMIX).

        Returns:
            np.ndarray: Global state with shape [n_agents * obs_dim]
        """
        use_self_only = bool(getattr(self, "state_use_self_only", False))
        self_only_dim = getattr(self, "state_self_only_dim", None)
        env_tail_dim = int(getattr(self, "state_self_only_env_tail_dim", 0) or 0)
        state_aggregation_mode = str(
            getattr(self, "state_aggregation_mode", "concat")
        ).lower()
        state_aggregation_topk = int(getattr(self, "state_aggregation_topk", 0) or 0)

        def _maybe_slice(obs: np.ndarray) -> np.ndarray:
            if not use_self_only or self_only_dim is None:
                return obs
            arr = np.asarray(obs)
            if arr.ndim == 0:
                return arr
            if arr.shape[-1] >= self_only_dim:
                head = arr[..., : self_only_dim]
                if env_tail_dim > 0 and arr.shape[-1] >= self_only_dim + env_tail_dim:
                    tail = arr[..., -env_tail_dim:]
                    return np.concatenate([head, tail], axis=-1)
                return head
            return arr

        all_obs = []
        for agent in self.possible_agents:
            satellites = self.meta_agent_groupings[agent]
            if agent in self.previously_dead:
                # Dead agents contribute zero observations
                agent_obs = [
                    _maybe_slice(np.zeros_like(satellite.observation_space.low))
                    for satellite in satellites
                ]
            else:
                agent_obs = [_maybe_slice(satellite.get_obs()) for satellite in satellites]

            if len(agent_obs) == 1:
                all_obs.append(agent_obs[0])
            else:
                all_obs.append(np.concatenate(agent_obs))

        if not all_obs:
            return np.asarray([], dtype=self.dtype if self.dtype else np.float32)

        if state_aggregation_mode == "concat":
            return np.concatenate(all_obs)

        state_mat = np.stack(all_obs, axis=0)
        if state_aggregation_mode == "mean":
            return state_mat.mean(axis=0)
        if state_aggregation_mode == "mean_std":
            return np.concatenate([state_mat.mean(axis=0), state_mat.std(axis=0)], axis=0)
        if state_aggregation_mode == "topk_mean":
            k = max(1, min(state_aggregation_topk, state_mat.shape[0]))
            idx = np.argsort(np.linalg.norm(state_mat, axis=1))[-k:]
            return state_mat[idx].mean(axis=0)

        # Unknown mode fallback.
        return np.concatenate(all_obs)

    @property
    def state_space(self) -> spaces.Box:
        """Return the global state space for centralized training.

        Returns:
            spaces.Box: State space with shape [n_agents * obs_dim]
        """
        use_self_only = bool(getattr(self, "state_use_self_only", False))
        self_only_dim = getattr(self, "state_self_only_dim", None)
        env_tail_dim = int(getattr(self, "state_self_only_env_tail_dim", 0) or 0)
        state_aggregation_mode = str(
            getattr(self, "state_aggregation_mode", "concat")
        ).lower()

        def _maybe_slice_space(arr: np.ndarray) -> np.ndarray:
            if not use_self_only or self_only_dim is None:
                return arr
            arr = np.asarray(arr)
            if arr.ndim == 0:
                return arr
            if arr.shape[-1] >= self_only_dim:
                head = arr[..., : self_only_dim]
                if env_tail_dim > 0 and arr.shape[-1] >= self_only_dim + env_tail_dim:
                    tail = arr[..., -env_tail_dim:]
                    return np.concatenate([head, tail], axis=-1)
                return head
            return arr

        # Concatenate all observation spaces
        all_lows = []
        all_highs = []
        for agent in self.possible_agents:
            obs_space = self.observation_space(agent)
            all_lows.append(_maybe_slice_space(obs_space.low))
            all_highs.append(_maybe_slice_space(obs_space.high))

        dtype = self.dtype if self.dtype else np.float32
        low_mat = np.stack(all_lows, axis=0)
        high_mat = np.stack(all_highs, axis=0)

        if state_aggregation_mode == "concat":
            low = np.concatenate(all_lows)
            high = np.concatenate(all_highs)
        elif state_aggregation_mode == "mean":
            low = low_mat.mean(axis=0)
            high = high_mat.mean(axis=0)
        elif state_aggregation_mode == "mean_std":
            low = np.concatenate([low_mat.mean(axis=0), np.zeros_like(low_mat.mean(axis=0))], axis=0)
            high = np.concatenate([high_mat.mean(axis=0), np.maximum(np.abs(high_mat), np.abs(low_mat)).max(axis=0)], axis=0)
        elif state_aggregation_mode == "topk_mean":
            low = low_mat.mean(axis=0)
            high = high_mat.mean(axis=0)
        else:
            low = np.concatenate(all_lows)
            high = np.concatenate(all_highs)
        if dtype is not None:
            low = low.astype(dtype, copy=False)
            high = high.astype(dtype, copy=False)
        return spaces.Box(
            low=low,
            high=high,
            dtype=dtype,
        )

    def _get_reward(self) -> dict[AgentID, float]:
        """Format the reward per the PettingZoo Parallel API."""
        satellite_rewards = {
            self.get_satellite(name): reward
            for name, reward in self.reward_dict.items()
        }
        for satellite in self.satellites:
            if not satellite.is_alive():
                if satellite in satellite_rewards:
                    satellite_rewards[satellite] += self.failure_penalty
                else:
                    satellite_rewards[satellite] = self.failure_penalty

        # 始终为所有 agents 返回 reward，包括 dead agents
        reward = {}
        for agent, sats in self.meta_agent_groupings.items():
            if agent in self.previously_dead:
                reward[agent] = 0.0  # Dead agents 返回 0
            else:
                reward[agent] = sum(satellite_rewards[sat] for sat in sats)

        return reward

    def _get_terminated(self) -> dict[AgentID, bool]:
        """Format terminations per the PettingZoo Parallel API."""
        # 始终为所有 agents 返回 terminated，包括 dead agents
        terminated = {}
        for agent, satellites in self.meta_agent_groupings.items():
            if agent in self.previously_dead:
                terminated[agent] = True  # Dead agents 视为已终止
            elif self.terminate_on_time_limit and super()._get_truncated():
                terminated[agent] = True
            else:
                terminated[agent] = any(
                    not sat.is_alive() or self.rewarder.is_terminated(sat)
                    for sat in satellites
                )
                # 🔍 调试: 找出第一个 step terminated 的原因
                if terminated[agent] and self.simulator.sim_time < 100:
                    for sat in satellites:
                        alive = sat.is_alive()
                        rewarder_term = self.rewarder.is_terminated(sat)
                        if not alive or rewarder_term:
                            logger.warning(
                                f"[TERM DEBUG] {agent} terminated at t={self.simulator.sim_time:.1f}s: "
                                f"is_alive={alive}, rewarder.is_terminated={rewarder_term}"
                            )
        return terminated

    def _get_truncated(self) -> dict[AgentID, bool]:
        """Format truncations per the PettingZoo Parallel API."""
        truncated_global = super()._get_truncated()
        # 始终为所有 agents 返回 truncated，包括 dead agents
        truncated = {}
        for agent, satellites in self.meta_agent_groupings.items():
            if agent in self.previously_dead:
                truncated[agent] = False  # Dead agents 不视为被截断
            else:
                truncated[agent] = truncated_global or any(
                    self.rewarder.is_truncated(sat) for sat in satellites
                )
        return truncated

    def _get_info(self) -> dict[AgentID, dict]:
        """Format info per the PettingZoo Parallel API."""
        info_per_sat = super()._get_info()

        # Group info by agent
        # 始终为所有 agents 返回 info，包括 dead agents，避免 TorchRL KeyError
        info = {}
        for agent, satellites in self.meta_agent_groupings.items():
            if agent in self.previously_dead:
                # Dead agents 返回默认 info
                info[agent] = {"requires_retasking": False}
            else:
                info[agent] = {
                    "requires_retasking": any(
                        info_per_sat[sat.name]["requires_retasking"]
                        for sat in satellites
                    )
                }
                if len(satellites) > 1:
                    for satellite in satellites:
                        info[agent][satellite.name] = info_per_sat[satellite.name]

        # Identify common info
        common = {
            k: v
            for k, v in info_per_sat.items()
            if k not in [sat.name for sat in self.satellites]
        }

        # Pass common info to all agents and to __common__
        for agent in info.keys():
            for k, v in common.items():
                info[agent][k] = v
        info["__common__"] = common

        return info

    def _decompose_meta_action(
        self, agent: AgentID, action: SatAct
    ) -> dict[Satellite, SatAct]:
        """Decompose a meta agent action into satellite actions."""
        sat_to_action_map = {}
        i = 0
        for satellite in self.meta_agent_groupings[agent]:
            action_len = satellite.action_space.shape
            if len(action_len) == 0:
                action_len = 1
            else:
                action_len = action_len[0]

            if isinstance(action, (list, tuple, np.ndarray)):
                if (
                    not self.only_retask_idle_meta_agent_members
                    or satellite.requires_retasking
                ):
                    if action_len == 1:
                        sat_to_action_map[satellite] = action[i]
                    else:
                        sat_to_action_map[satellite] = action[i : i + action_len]
                else:
                    sat_to_action_map[satellite] = None
                i += action_len
            else:
                sat_to_action_map[satellite] = action

        return sat_to_action_map

    def step(
        self,
        actions: dict[AgentID, SatAct],
    ) -> tuple[
        dict[AgentID, SatObs],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        """Step the environment and return PettingZoo Parallel API format."""
        logger.debug("=== STARTING STEP ===")

        previous_alive = self.agents

        sat_to_action_map = {}
        for agent, action in actions.items():
            if len(self.meta_agent_groupings[agent]) > 1:
                logger.debug(f"Decomposing action for meta agent {agent}")
            sat_to_action_map.update(self._decompose_meta_action(agent, action))

        action_vector = []
        for satellite in self.satellites:
            if satellite in sat_to_action_map:
                action_vector.append(sat_to_action_map[satellite])
            else:
                action_vector.append(None)
        self._step(action_vector)
        self._update_curriculum_step_counters()

        self.newly_dead = list(set(previous_alive) - set(self.agents))

        for agent in self.newly_dead:
            for satellite in self.meta_agent_groupings[agent]:
                for attr in [
                    "_timed_terminal_event_name",
                    "_image_event_name",
                ]:
                    event_name = getattr(satellite, attr, None)
                    if event_name is not None:
                        self.simulator.delete_event(event_name)

        observation = self._get_obs()
        reward = self._get_reward()
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        done_any = any(terminated.values()) or any(truncated.values())
        done_all = all(terminated.values()) or all(truncated.values())
        self._last_episode_done = bool(done_all)
        info = self._get_info()
        nonzero_reward = {k: v for k, v in reward.items() if v != 0}
        logger.debug(f"Step reward: {nonzero_reward}")
        if any(terminated.values()):
            terminated_true = [k for k, v in terminated.items() if v]
            logger.info(f"Episode terminated: {terminated_true}")
        if any(truncated.values()):
            truncated_true = [k for k, v in truncated.items() if v]
            logger.info(f"Episode truncated: {truncated_true}")
        logger.debug("Step info: %s", info)
        logger.debug("Step observation: %s", observation)
        return observation, reward, terminated, truncated, info


__all__ = []
