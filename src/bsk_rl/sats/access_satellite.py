"""Satellites are the agents in the environment."""

import bisect
import logging
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Union

import numpy as np
from Basilisk.utilities import macros
from Basilisk.simulation import simpleStorageUnit, simpleBattery

from scipy.optimize import minimize_scalar, root_scalar

from bsk_rl.sats.satellite import Satellite
from bsk_rl.scene.targets import Target
from bsk_rl.sim import dyn, fsw
from bsk_rl.utils import vizard
from bsk_rl.utils.functional import valid_func_name
from bsk_rl.utils.orbital import elevation

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.data.unique_image_data import UniqueImageStore

logger = logging.getLogger(__name__)

SatObs = Any
SatAct = Any

'''
具体卫星类被设计为不感知决策接口的物理实体，其 action_space 与 observation_space 的定义被刻意外置到 Builder 层，以实现动力学建模与决策接口构造的解耦。这种设计符合控制系统与强化学习中的分层建模原则，并显著提升了系统的可扩展性与复用性。
'''
class AccessSatellite(Satellite):
    """Satellite that detects access opportunities for ground locations.
    探测地面站的接入机会，几何可见性与时间窗口的“信息生成器（information generator）”，而不是“决策执行体（decision executor）”。
    """

    def __init__(
        self,
        *args,
        generation_duration: float = 600.0,
        initial_generation_duration: Optional[float] = None,
        max_generation_duration_beyond_initial: Optional[float] = float("inf"),
        **kwargs,
    ) -> None:
        """Satellite that detects access opportunities for ground locations.
        它不执行动作，只负责 计算时间窗口。它维护一份“日程表”，告诉 Agent 在什么时间段内，卫星与地面点（目标或地面站）是几何可见的。
        This satellite can be used to computes access opportunities for ground locations
        such as imaging targets or ground stations. The satellite will calculate upcoming
        opportunities for each location and order the opportunities by close time.
        Opportunities are calculated based on a per-location minimum elevation angle.

        Args:
            args: Passed through to :class:`Satellite` constructor.
            generation_duration: [s] Duration to calculate additional opportunities for
                when the simulation time reaches the current calculation time. If
                `None`, generate opportunities for the simulation `time_limit` unless
                the simulation is infinite.
            initial_generation_duration: [s] Period to calculate opportunities for on
                environment reset.
            max_generation_duration_beyond_initial: [s] Maximum time to calculate opportunities
                beyond the initial generation duration.
            kwargs: Passed through to :class:`Satellite` constructor.
        """
        super().__init__(*args, **kwargs)
        self.generation_duration = generation_duration
        self.initial_generation_duration = initial_generation_duration
        self.max_generation_duration_beyond_initial = (
            max_generation_duration_beyond_initial
        )
        self.access_filter_functions = []
        self.add_access_filter(lambda opportunity: True)

    def reset_overwrite_previous(self) -> None:
        """Overwrite previous opportunities and locations."""
        super().reset_overwrite_previous()
        self.opportunities: list[dict] = []
        self.window_calculation_time = 0
        self.locations_for_access_checking: list[dict[str, Any]] = []

    def add_location_for_access_checking(
        self,
        object: Any,
        r_LP_P: np.ndarray,
        min_elev: float,
        type: str,
        start_time: float = 0.0,
    ) -> None:
        """Add a location to be included in opportunity calculations.

        .. warning::
            The added location will only be considered in future calls to
            :class:`~AccessSatellite.calculate_additional_windows`; opportunities are not
            computed retroactively.“新添加的地面点（Location）不会自动补算‘过去’的可见窗口，它只会在‘未来’的计算周期中生效。”
            该方法只是把地面点的信息存储起来，等待后续的计算周期进行可见窗口的计算。
            如何避免这个问题？
            如果你必须在仿真中途动态添加目标，并且希望能立即看到它的窗口，你需要手动触发一次计算，或者在添加目标时重置计算时间（但这可能会导致重复计算，比较复杂）
        Args:
            object: Object for with to compute opportunities.
            r_LP_P: [m] Objects planet-fixed location.
            min_elev: [rad] Minimum elevation angle for access.
            type: Category of opportunity target provides.
            start_time: [s] Time at which to start calculating opportunities for this location.
        """
        location_dict = dict(r_LP_P=r_LP_P, min_elev=min_elev, type=type)
        location_dict[type] = object  # For backwards compatibility, prefer "object" key
        location_dict["object"] = object
        location_dict["start_time"] = start_time
        self.locations_for_access_checking.append(location_dict)

    def reset_post_sim_init(self) -> None:
        """Handle initial window calculations for new simulation.

        :meta private:
        """
        super().reset_post_sim_init()
        if self.initial_generation_duration is None:
            if self.simulator.time_limit == float("inf"):
                self.initial_generation_duration = 0
            else:
                self.initial_generation_duration = self.simulator.time_limit
        self.calculate_additional_windows(self.initial_generation_duration)

    def calculate_additional_windows(self, duration: float) -> None:
        """Use a multiroot finding method to evaluate imaging windows for each location.

        Args:
            duration: Time to calculate windows from end of previous window.
        """
        if duration <= 0:
            return

        calculation_start = self.window_calculation_time
        calculation_end = self.window_calculation_time + max(
            duration, self.trajectory.dt * 2, self.generation_duration
        )
        calculation_end = self.generation_duration * np.ceil(
            calculation_end / self.generation_duration
        )

        self.logger.info(
            "Finding opportunity windows from "
            f"{calculation_start:.2f} to "
            f"{calculation_end:.2f} seconds"
        )

        # Get discrete times and positions for next trajectory segment
        self.trajectory.extend_to(calculation_end)
        r_BP_P_interp = self.trajectory.r_BP_P
        window_calc_span = np.logical_and(
            r_BP_P_interp.x >= calculation_start - 1e-9,
            r_BP_P_interp.x <= calculation_end + 1e-9,
        )  # Account for floating point error in window_calculation_time
        times = r_BP_P_interp.x[window_calc_span]
        positions = r_BP_P_interp.y[window_calc_span]

        r_max = np.max(np.linalg.norm(positions, axis=-1))
        access_dist_thresh_multiplier = 1.1
        for location in self.locations_for_access_checking:
            start_idx = max(
                np.searchsorted(times, location["start_time"], side="right") - 1, 0
            )
            times_loc = times[start_idx:]
            positions_loc = positions[start_idx:]

            alt_est = r_max - np.linalg.norm(location["r_LP_P"])
            access_dist_threshold = (
                access_dist_thresh_multiplier * alt_est / np.sin(location["min_elev"])
            )
            candidate_windows = self._find_candidate_windows(
                location["r_LP_P"], times_loc, positions_loc, access_dist_threshold
            )

            for candidate_window in candidate_windows:
                roots = self._find_elevation_roots(
                    r_BP_P_interp,
                    location["r_LP_P"],
                    location["min_elev"],
                    candidate_window,
                )
                new_windows = self._refine_window(
                    roots, candidate_window, (times_loc[0], times_loc[-1])
                )
                for new_window in new_windows:
                    self._add_window(
                        location["object"],
                        new_window,
                        type=location["type"],
                        r_LP_P=location["r_LP_P"],
                        merge_time=times_loc[0],
                    )

        self.window_calculation_time = calculation_end

    @staticmethod
    def _find_elevation_roots(
        position_interp,
        location: np.ndarray,
        min_elev: float,
        window: tuple[float, float],
        min_duration: float = 0.1,
    ):
        """Find times where the elevation is equal to the minimum elevation.

        Finds exact times where the satellite's elevation relative to a target is
        equal to the minimum elevation.
        """

        def root_fn(t):
            return -(elevation(position_interp(t), location) - min_elev)

        elev_0, elev_1 = root_fn(window[0]), root_fn(window[1])

        if elev_0 < 0 and elev_1 < 0:
            logger.warning(
                "initial_generation_duration is shorter than the maximum window length; some windows may be neglected."
            )
            return []
        elif elev_0 < 0 or elev_1 < 0:
            return [root_scalar(root_fn, bracket=window).root]
        else:
            res = minimize_scalar(root_fn, bracket=window, tol=1e-4)
            if res.fun < 0:
                window_mid = res.x
                r_open = root_scalar(root_fn, bracket=(window[0], window_mid)).root
                r_close = root_scalar(root_fn, bracket=(window_mid, window[1])).root
                if r_close - r_open > min_duration:
                    return [r_open, r_close]

        return []

    @staticmethod
    def _find_candidate_windows(
        location: np.ndarray, times: np.ndarray, positions: np.ndarray, threshold: float
    ) -> list[tuple[float, float]]:
        """Find `times` where a window is plausible.

        i.e. where a `positions` point is within `threshold` of `location`. Too big of
        a dt in times may miss windows or produce bad results.
        """
        close_times = np.linalg.norm(positions - location, axis=1) < threshold
        close_indices = np.where(close_times)[0]
        groups = np.split(close_indices, np.where(np.diff(close_indices) != 1)[0] + 1)
        groups = [group for group in groups if len(group) > 0]
        candidate_windows = []
        for group in groups:
            t_start = times[max(0, group[0] - 1)]
            t_end = times[min(len(times) - 1, group[-1] + 1)]
            candidate_windows.append((t_start, t_end))
        return candidate_windows

    @staticmethod
    def _refine_window(
        endpoints: Iterable,
        candidate_window: tuple[float, float],
        computation_window: tuple[float, float],
    ) -> list[tuple[float, float]]:
        """Detect if an exact window has been truncated by a coarse window."""
        endpoints = list(endpoints)

        # Filter endpoints that are too close
        for i, endpoint in enumerate(endpoints[0:-1]):
            if abs(endpoint - endpoints[i + 1]) < 1e-6:
                endpoints[i] = None
        endpoints = [endpoint for endpoint in endpoints if endpoint is not None]

        # Find pairs
        if len(endpoints) % 2 == 1:
            if candidate_window[0] == computation_window[0]:
                endpoints.insert(0, computation_window[0])
            elif candidate_window[-1] == computation_window[-1]:
                endpoints.append(computation_window[-1])
            else:
                return []  # Temporary fix for rare issue.

        new_windows = []
        for t1, t2 in zip(endpoints[0::2], endpoints[1::2]):
            new_windows.append((t1, t2))

        return new_windows

    def _add_window(
        self,
        object: Any,
        new_window: tuple[float, float],
        type: str,
        r_LP_P: np.ndarray,
        merge_time: Optional[float] = None,
    ):
        """Add an opportunity window.

        Args:
            object: Object to add window for
            new_window: New window for target
            type: Type of window being added
            r_LP_P: Planet-fixed location of object
            merge_time: Time at which merges with existing windows will occur. If None,
                check all windows for merges.
        """
        if new_window[0] == merge_time or merge_time is None:
            for opportunity in self.opportunities:
                if (
                    opportunity["type"] == type
                    and opportunity["object"] == object
                    and opportunity["window"][1] == new_window[0]
                ):
                    opportunity["window"] = (opportunity["window"][0], new_window[1])
                    return
        bisect.insort(
            self.opportunities,
            {"object": object, "window": new_window, "type": type, "r_LP_P": r_LP_P},
            key=lambda x: x["window"][1],
        )

    @property
    def upcoming_opportunities(self) -> list[dict]:
        """Ordered list of opportunities that have not yet closed."""
        start = bisect.bisect_left(
            self.opportunities,
            self.simulator.sim_time + 1e-12,
            key=lambda x: x["window"][1],
        )
        upcoming = self.opportunities[start:]
        return upcoming

    def opportunities_dict(
        self,
        types: Optional[Union[str, list[str]]] = None,
        filter: Union[Optional[Callable], list] = None,
    ) -> dict[Any, list[tuple[float, float]]]:
        """Make dictionary of opportunities that maps objects to lists of windows.

        Args:
            types: Types of opportunities to include. If None, include all types.
            filter: Function that takes an opportunity dictionary and returns a boolean
                if the opportunity should be included in the output.
        """
        if isinstance(types, str):
            types = [types]

        if isinstance(filter, list):
            filter_list = filter
            filter = lambda opportunity: opportunity["object"] not in filter_list

        if filter is None:
            filter = self.default_access_filter

        windows = {}
        for opportunity in self.opportunities:
            type = opportunity["type"]
            if (types is None or type in types) and filter(opportunity):
                if opportunity["object"] not in windows:
                    windows[opportunity["object"]] = []
                windows[opportunity["object"]].append(opportunity["window"])
        return windows

    def upcoming_opportunities_dict(
        self,
        types: Optional[Union[str, list[str]]] = None,
        filter: Union[Optional[Callable], list] = None,
    ) -> dict[Any, list[tuple[float, float]]]:
        """Get dictionary of upcoming opportunities.

        Maps objects to lists of windows that have not yet closed.

        Args:
            types: Types of opportunities to include. If None, include all types.
            filter: Function that takes an opportunity dictionary and returns a boolean
                if the opportunity should be included in the output.
        """
        if isinstance(types, str):
            types = [types]

        if isinstance(filter, list):
            filter_list = filter
            filter = lambda opportunity: opportunity["object"] not in filter_list

        if filter is None:
            filter = self.default_access_filter

        windows = {}
        for opportunity in self.upcoming_opportunities:
            type = opportunity["type"]
            if (types is None or type in types) and filter(opportunity):
                if opportunity["object"] not in windows:
                    windows[opportunity["object"]] = []
                windows[opportunity["object"]].append(opportunity["window"])
        return windows

    def next_opportunities_dict(
        self,
        types: Optional[Union[str, list[str]]] = None,
        filter: Union[Optional[Callable], list] = None,
        min_needed: Optional[int] = None,
    ) -> dict[Any, tuple[float, float]]:
        """Make dictionary of opportunities that maps objects to the next open windows.

        Args:
            types: Types of opportunities to include. If None, include all types.
            filter: Function that takes an opportunity dictionary and returns a boolean
                if the opportunity should be included in the output.
            min_needed: Minimum number of opportunities to return. If None, return all
        """
        if isinstance(types, str):
            types = [types]

        if isinstance(filter, list):
            filter_list = filter
            filter = lambda opportunity: opportunity["object"] not in filter_list

        if filter is None:
            filter = self.default_access_filter

        next_windows = {}
        total_found = 0
        for opportunity in self.upcoming_opportunities:
            type = opportunity["type"]
            if (types is None or type in types) and filter(opportunity):
                if opportunity["object"] not in next_windows:
                    next_windows[opportunity["object"]] = opportunity["window"]
                    total_found += 1
            if min_needed is not None and total_found >= min_needed:
                break
        return next_windows

    def find_next_opportunities(
        self,
        n: int,
        pad: bool = True,
        max_lookahead: int = 100,
        types: Optional[Union[str, list[str]]] = None,
        filter: Union[Optional[Callable], list] = None,
    ) -> list[dict]:
        """Find the n nearest opportunities, sorted by window close time.

        Args:
            n: Number of opportunities to attempt to include.
            pad: If true, duplicates the last target if the number of opportunities
                found is less than n.
            max_lookahead: Maximum times to call calculate_additional_windows.
            types: Types of opportunities to include. If None, include all types.
            filter: Function that takes an opportunity dictionary and returns a boolean
                if the opportunity should be included in the output.

        Returns:
            ``n`` nearest opportunities, ordered
        """
        if isinstance(types, str):
            types = [types]

        if isinstance(filter, list):
            filter_list = filter
            filter = lambda opportunity: opportunity["object"] not in filter_list

        if filter is None:
            filter = self.default_access_filter

        if n == 0:
            return []

        for _ in range(max_lookahead):
            upcoming_opportunities = self.upcoming_opportunities
            next_opportunities = []
            for opportunity in upcoming_opportunities:
                type = opportunity["type"]
                if (types is None or type in types) and filter(opportunity):
                    next_opportunities.append(opportunity)

                if len(next_opportunities) >= n:
                    return next_opportunities
            if (
                self.window_calculation_time
                >= self.initial_generation_duration
                + self.max_generation_duration_beyond_initial
            ):
                break
            self.calculate_additional_windows(self.generation_duration)
        if pad and len(next_opportunities) >= 1:
            self.logger.info(
                f"Only {len(next_opportunities)} opportunities found, padding to {n}."
            )
            next_opportunities += [next_opportunities[-1]] * (
                n - len(next_opportunities)
            )
        else:
            raise RuntimeError(
                "No opportunities found! Use add_location_for_access_checking to add locations."
            )
        return next_opportunities

    def get_access_filter(self):
        """Deprecated function.

        :meta private:
        """
        raise DeprecationWarning(
            "get_access_filter is deprecated. Use add_access_filter and default_access_filter instead."
        )

    def add_access_filter(
        self,
        access_filter_fn: Callable,
        types: Optional[Union[str, list[str]]] = None,
        prepend: bool = False,
    ):
        """Add an access filter function to the list of access filters.

        Calls to :class:`~AccessSatellite.opportunities_dict`, :class:`~AccessSatellite.find_next_opportunities`,
        and similar functions will use the boolean AND of all access filter functions,
        unless otherwise specified.

        Access filters are used by various other aspects of the environment to limit
        which opportunities are considered based on the satellite's local knowledge of
        the environment.
        """
        if types is not None:
            if isinstance(types, str):
                types = [types]

            def access_filter_type_restricted(opportunity):
                return opportunity["type"] not in types or access_filter_fn(opportunity)

            to_add = access_filter_type_restricted
        else:
            to_add = access_filter_fn

        if prepend:
            self.access_filter_functions.insert(0, to_add)
        else:
            self.access_filter_functions.append(to_add)

    @property
    def default_access_filter(self):
        """Generate a default access filter function that combines all access filters.

        :meta private:
        """

        def access_filter(opportunity):
            for access_filter_fn in self.access_filter_functions:
                if not access_filter_fn(opportunity):
                    return False
            else:
                return True

        return access_filter
'''
    这些具体的卫星都没有对父类的动作和状态空间进行重写，因为他们本身并不参与决策，都是被动的“信息生成器”。真正定义“可观测量”和“可执行动作”的，是 Builder 层构建的 Observation / Action 对象。所以直接使用父类的即可，因为我们将修改逻辑放到了开头的类属性中，也就是 dyn_type 和 fsw_type 以及 action_builder_type 和 observation_builder_type。
'''

class ImagingSatellite(AccessSatellite):
    """Satellite with agile imaging capabilities."""

    buffer_name = "image_buffer"

    dyn_type = dyn.ImagingDynModel
    fsw_type = fsw.ImagingFSWModel

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Satellite with agile imaging capabilities.

        Stop the simulation when a target is imaged or missed so that time is not wasted
        on an inaccessible or already imaged target.
        """
        super().__init__(*args, **kwargs)
        self.fsw: ImagingSatellite.fsw_type
        self.dynamics: ImagingSatellite.dyn_type
        self.data_store: "UniqueImageStore"
        self.target_types = "target"
        self.latest_target = None

    @property
    def known_targets(self) -> list["Target"]:
        """List of known targets."""
        try:
            return self.data_store.data.known
        except AttributeError:
            return []

    def reset_overwrite_previous(self) -> None:
        """Overwrite statistics about previous episode."""
        super().reset_overwrite_previous()
        self._image_event_name = None
        self.imaged = 0
        self.missed = 0

    def reset_pre_sim_init(self) -> None:
        """Set the buffer parameters based on computed windows.

        :meta private:
        """
        super().reset_pre_sim_init()
        self.sat_args["bufferNames"] = [self.buffer_name]
        self.sat_args["transmitterNumBuffers"] = len(self.sat_args["bufferNames"])

    def _update_image_event(self, target: "Target") -> None:
        """Create a simulator event that terminates on imaging.

        Causes the simulation to stop when a target is imaged.

        Args:
            target: Target expected to be imaged
        """
        self._disable_image_event()

        self._image_event_name = valid_func_name(f"image_{self.name}_{target.id}")
        if self._image_event_name not in self.simulator.eventMap.keys():
            current_data_level = (
                self.dynamics.storageUnit.storageUnitDataOutMsg.read().storedData[0]
            )

            def side_effect(sim):
                self.logger.info(f"imaged {target}")
                self.imaged += 1
                self.requires_retasking = True
                self.remove_imaging_line()

            self.simulator.createNewEvent(
                self._image_event_name,
                macros.sec2nano(self.fsw.fsw_rate),
                True,
                conditionFunction=lambda sim: self.dynamics.storageUnit.storageUnitDataOutMsg.read().storedData[
                    0
                ]
                > current_data_level,
                actionFunction=side_effect,
                terminal=self.variable_interval,
            )
        else:
            self.simulator.eventMap[self._image_event_name].eventActive = True

    def _disable_image_event(self) -> None:
        """Turn off simulator termination due to this satellite's imaging checker."""
        if (
            self._image_event_name is not None
            and self._image_event_name in self.simulator.eventMap
        ):
            self.simulator.delete_event(self._image_event_name)

    def parse_target_selection(self, target_query: Union[int, Target, str]):
        """Identify a target from a query.

        Parses an upcoming target index, Target object, or target id.

        Args:
            target_query: Target upcoming index, object, or id.
        """
        if np.issubdtype(type(target_query), np.integer):
            target = self.find_next_opportunities(
                n=target_query + 1, types=self.target_types
            )[-1]["object"]
        elif isinstance(target_query, Target):
            target = target_query
        elif isinstance(target_query, str):
            try:
                target = [
                    target for target in self.known_targets if target.id == target_query
                ][0]
            except IndexError:
                raise ValueError(f"Target {target_query} not a known target!")
        else:
            raise TypeError(f"Invalid target_query! Cannot be a {type(target_query)}!")

        return target

    def enable_target_window(
        self, target: "Target", max_duration: Optional[float] = None
    ):
        """Enable a timed opportunity close event and a successfully imaged event.

        Args:
            target: Target to terminate the step on imaging or when out of range.
            max_duration: [s] Maximum duration to wait for imaging. If None, use the default
                behavior.
        """
        self._update_image_event(target)
        for opportunity in self.upcoming_opportunities:
            if opportunity["object"] == target:
                next_window = opportunity["window"]
                break
        self.logger.info(
            f"{target} window enabled: {next_window[0]:.1f} to {next_window[1]:.1f}"
        )
        if max_duration is None:
            max_duration = 1e9

        def side_effect(sim):
            if np.isclose(sim.sim_time, next_window[1], atol=1e-9):
                self.missed += 1
            self.remove_imaging_line()

        self.update_timed_terminal_event(
            min(next_window[1], self.simulator.sim_time + max_duration),
            info=f"for {target} window",
            extra_actions=side_effect,
        )

    def task_target_for_imaging(
        self, target: "Target", max_duration: Optional[float] = None
    ):
        """Task the satellite to image a target.

        Args:
            target: Selected target
            max_duration: [s] Maximum duration to wait for imaging. If None, wait until
                the end of the target's access window.
        """
        msg = f"{target} tasked for imaging"
        self.logger.info(msg)
        self.fsw.action_image(target.r_LP_P, self.buffer_name)
        self.enable_target_window(target, max_duration=max_duration)
        self.draw_imaging_line(target)
        self.latest_target = target

    @vizard.visualize
    def draw_imaging_line(
        self, target: "Target", vizSupport=None, vizInstance=None
    ) -> None:
        """Draw a line from the satellite to the target in vizard."""
        if not hasattr(self, "target_line"):
            vizSupport.createTargetLine(
                vizInstance,
                fromBodyName=self.name,
                toBodyName=target.name,
                lineColor=self.vizard_color,
            )
            self.target_line = vizSupport.targetLineList[-1]
        self.target_line.toBodyName = target.name
        vizSupport.updateTargetLineList(vizInstance)

    @vizard.visualize
    def remove_imaging_line(self, vizSupport=None, vizInstance=None):
        """Remove the imaging line from Vizard."""
        if hasattr(self, "target_line"):
            self.target_line.toBodyName = self.name
            vizSupport.updateTargetLineList(vizInstance)

# class ComputationSatellite(AccessSatellite):
    # """
    # STIN 计算节点卫星。
    # 继承自 AccessSatellite，自动拥有由 sat_args 配置的电池、存储、发射机和姿态控制。
    # 仅扩展 CPU 计算逻辑。
    # """
    # dyn_type = dyn.ComputationDynModel
    # fsw_type = fsw.ImagingFSWModel
    # # 1. 定义论文与物理环境对应的参数配置 (SMEC Config)
    # smec_config = {
    #     # --- 通信 (对应论文 B_v, p_u) ---
    #     "transmitterBaudRate": -20.0 * 1e6,  # -20 Mbps (负数=发送)
    #     "transmitterPowerDraw": -15.0,       # 15W 输入 -> ~5W 射频输出
    #     "transmitterNumBuffers": 50,         # 允许并行的传输缓冲区数量
        
    #     # --- 存储 (对应论文 d_t 及 buffer) ---
    #     "dataStorageCapacity": 1.0 * 1e9,    # 1 Gbit
        
    #     # --- 能源 ---
    #     "batteryStorageCapacity": 500000.0,  # 500 kJ (~140 Wh)
    #     "basePowerDraw": -10.0,              # 基础平台功耗 (修正 0.0W 的不合理设定)
    #     "instrumentPowerDraw": 0.0,          # 禁用默认的相机功耗，我们只用 CPU
        
    #     # --- [新增] 计算模块参数 (论文 F, c_t) ---
    #     # 这些参数基类不识别，但会保留在 sat_args 中供我们使用
    #     "cpuFrequency": 1.0 * 1e9,           # 1 GHz
    #     "cpuWorkload": 1000.0,               # 1000 cycles/bit
    #     "cpuPowerDraw": -20.0,               # CPU 满载功耗 20W
    # }

    # @classmethod
    # def default_sat_args(cls, **kwargs) -> dict[str, Any]:
    #     """Add SMEC defaults while keeping parent validation."""
    #     defaults = super().default_sat_args()
    #     defaults.update(cls.smec_config)

    #     for k, v in kwargs.items():
    #         if k not in defaults:
    #             raise KeyError(f"{k} not a valid key for sat_args")
    #         defaults[k] = v

    #     return defaults

    # def __init__(self, name, sat_args, *args, **kwargs):
    #     super().__init__(name, sat_args, *args, **kwargs)

    #     self.cpu_process_rate: float | None = None
    #     self.processed_data_total = 0.0
    #     self.offloaded_data_total = 0.0

    # def generate_sat_args(self, **kwargs) -> None:
    #     """Generate sat_args then cache计算速率 F/c_t."""
    #     super().generate_sat_args(**kwargs)
    #     self.cpu_process_rate = (
    #         self.sat_args["cpuFrequency"] / self.sat_args["cpuWorkload"]
    #     )

    # def reset_overwrite_previous(self) -> None:
    #     super().reset_overwrite_previous()
    #     self.processed_data_total = 0.0
    #     self.offloaded_data_total = 0.0
        

    # def execute_compute(self, duration: float) -> None:
    #     """
    #     [Action Backend] 执行本地计算。
    #     由 act.ComputeData 调用。
    #     """
    #     # 1. 物理层：开启 CPU 耗电 (通过 dynamics 访问我们自定义的 cpuPowerSink)
    #     # 注意：Action 调度器会在 duration 结束后自动让 step 停止，我们需要在停止时结算
    #     self.dynamics.cpuPowerSink.nodePowerOut = self.sat_args["cpuPowerDraw"]
        
    #     # 2. 逻辑层：计算预期处理量
    #     bits_to_process = self.cpu_process_rate * duration
        
    #     # 3. 定义回调：时间到后结算数据
    #     def _compute_finished(sim):
    #         # 获取当前存储量
    #         current_data = self.dynamics.storageUnit.storageLevel
    #         actual_processed = min(current_data, bits_to_process)
            
    #         # 扣除数据 (模拟被计算消耗)
    #         self.dynamics.storageUnit.storageLevel -= actual_processed
    #         self.processed_data_total += actual_processed
            
    #         # 关闭 CPU 耗电
    #         self.dynamics.cpuPowerSink.nodePowerOut = 0.0
            
    #         # 标记需要新决策
    #         self.requires_retasking = True
    #         self.logger.info(f"Computed {actual_processed/1e6:.2f} Mb")

    #     # 4. 注册终端事件
    #     self.update_timed_terminal_event(
    #         self.simulator.sim_time + duration,
    #         info="computing_task",
    #         extra_actions=_compute_finished
    #     )

    # def execute_offload(self, target_sat: "Satellite", duration: float) -> None:
    #     """
    #     [Action Backend] 执行任务卸载。
    #     由 act.OffloadData 调用。
    #     """
    #     # 1. 物理层：开启发射机耗电
    #     # 获取 transmitter 的功耗节点 (通常在 powerMonitor 中)
    #     # 假设基类逻辑：我们手动开启一个临时的 commLoad，或者直接假设 transmitter 开启
    #     # 这里为了简单，我们假设 AccessSatellite 的 transmitter 默认是开启待机的，
    #     # 我们这里不额外操作 PowerNode，只专注于数据流转。
    #     # (如果需要精确功耗，可以像 cpuPowerSink 那样再加一个 commPowerSink)
        
    #     # 2. 姿态控制：指向目标卫星 (复用 FSW 的 action_image)
    #     # 我们需要知道目标卫星的位置。AccessSatellite 会计算相对位置。
    #     # 这是一个高级功能：将目标卫星视为一个 Pointing Target
    #     # r_Target_P = target_sat.dynamics.r_BN_P
    #     # self.fsw.action_image(r_Target_P, "data_buffer") 
        
    #     # 3. 逻辑层：计算传输量
    #     tx_rate = abs(self.sat_args["transmitterBaudRate"])
    #     bits_to_send = tx_rate * duration
        
    #     def _offload_finished(sim):
    #         # 本地扣除
    #         current_data = self.dynamics.storageUnit.storageLevel
    #         actual_sent = min(current_data, bits_to_send)
            
    #         # 目标接收 (直接操作目标对象)
    #         # 注意：BSK-RL 允许多个卫星对象互访
    #         if hasattr(target_sat.dynamics, 'storageUnit'):
    #             space_left = target_sat.dynamics.storageUnit.storageCapacity - target_sat.dynamics.storageUnit.storageLevel
    #             actual_received = min(actual_sent, space_left)
                
    #             self.dynamics.storageUnit.storageLevel -= actual_received
    #             target_sat.dynamics.storageUnit.storageLevel += actual_received
                
    #             self.offloaded_data_total += actual_received
    #             self.logger.info(f"Offloaded {actual_received/1e6:.2f} Mb to {target_sat.name}")
            
    #         self.requires_retasking = True

    #     # 4. 注册终端事件
    #     self.update_timed_terminal_event(
    #         self.simulator.sim_time + duration,
    #         info=f"offload_to_{target_sat.name}",
    #         extra_actions=_offload_finished
    #     )
