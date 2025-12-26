"""Satellite action types can be used to add actions to an agent."""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from bsk_rl.utils.functional import AbstractClassProperty, Resetable

if TYPE_CHECKING:  # pragma: no cover
    from gymnasium import spaces

    from bsk_rl.sats import Satellite
    from bsk_rl.sim import Simulator

'''
逻辑链条如下：卫星的 action_spec 并不直接对应 gym 的动作空间，而是通过 ActionBuilder 进行中间转换。

    每个卫星（Satellite）包含一个 action_spec，用于描述该卫星支持的动作规格列表（Action 对象，处于定义态）。

    Action 是一个抽象基类，用于刻画具体动作的语义及其执行接口。

    ActionBuilder 是一个抽象构建器，负责管理 action_spec，并将其映射为 gym 兼容的动作空间，同时负责解析和分派智能体输出的动作。

    select_action_builder 根据 action_spec 中声明的 builder_type，选择并实例化合适的 ActionBuilder 子类。

    ActionBuilder 基于其持有的 action_spec 构建并暴露 gym 的 action_space，从而供智能体进行动作采样与输出。

    该设计实现了卫星动作语义与强化学习接口之间的解耦，提高了系统的可扩展性与可维护性。
'''
def select_action_builder(satellite: "Satellite") -> "ActionBuilder":
    """Identify the proper action builder based on a satellite's action spec.

    Args:
        satellite: Satellite to build actions for.

    Returns:
        action builder of the appropriate type
    """
    builder_types = [spec.builder_type for spec in satellite.action_spec]
    if all([builder_type == builder_types[0] for builder_type in builder_types]):
        return builder_types[0](satellite)
    else:
        raise NotImplementedError("Heterogenous action builders not supported.")


class ActionBuilder(ABC, Resetable):

    def __init__(self, satellite: "Satellite") -> None:
        """Base class for all action builders.

        Args:
            satellite: Satellite to build actions for.
        """
        self.satellite = satellite
        self.simulator: "Simulator"
        self.action_spec = deepcopy(self.satellite.action_spec)
        for act in self.action_spec:
            act.link_satellite(self.satellite)

    def reset_overwrite_previous(self) -> None:
        """Perform any once-per-episode setup."""
        for act in self.action_spec:
            act.reset_overwrite_previous()

    def reset_pre_sim_init(self) -> None:
        """Perform any once-per-episode setup."""
        for act in self.action_spec:
            act.reset_pre_sim_init()

    def reset_during_sim_init(self) -> None:
        """Perform any once-per-episode setup."""
        for act in self.action_spec:
            act.reset_during_sim_init()

    def reset_post_sim_init(self) -> None:
        """Perform any once-per-episode setup."""
        self.simulator = self.satellite.simulator  # already a proxy
        for act in self.action_spec:
            act.link_simulator(self.simulator)  # already a proxy
            act.reset_post_sim_init()

    @property
    @abstractmethod
    def action_space(self) -> "spaces.Space":
        """Return the action space."""
        pass

    @property
    @abstractmethod
    def action_description(self) -> Any:
        """Return a description of the action space."""
        pass

    @abstractmethod
    def set_action(self, action: Any) -> None:
        """Set the action to be taken."""
        pass


class Action(ABC, Resetable):
    builder_type: type[ActionBuilder] = AbstractClassProperty()  #: :meta private:

    def __init__(self, name: str = "act") -> None:
        """Base class for all actions.

        Args:
            name: Name of the action.
        """
        self.name = name
        self.satellite: "Satellite"
        self.simulator: "Simulator"

    def link_satellite(self, satellite: "Satellite") -> None:
        """Link the action to a satellite.

        Args:
            satellite: Satellite to link to

        :meta private:
        """
        self.satellite = satellite  # already a proxy

    def link_simulator(self, simulator: "Simulator") -> None:
        """Link the action to a simulator.

        Args:
            simulator: Simulator to link to

        :meta private:
        """
        self.simulator = simulator  # already a proxy

    @abstractmethod
    def set_action(self, action: Any) -> None:  # pragma: no cover
        """Execute code to perform an action."""
        pass


__doc_title__ = "Backend"
__all__ = ["ActionBuilder"]
