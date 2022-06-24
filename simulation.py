from dataclasses import dataclass, KW_ONLY
from utils import generate_skill_up_fn, EMPTY_MAT_COST, MAX_CRAFTING_LV


@dataclass(frozen=True)
class RunConfigs:
    _: KW_ONLY
    simulated_lv: int
    simulations: int


@dataclass(frozen=True)
class SimulationStep:
    level: int
    cost: int


class Simulation:
    def __init__(self, patterns: list, market: dict):
        self.cm_bank = {}
        self.patterns = \
            Simulation.augment_patterns_to_include_cost(patterns, market)
        self.probability_fn_mapping = \
            Simulation.map_probability_fns(self.patterns)
        # self.sum_costs = \
        #     Simulation.map_sum_costs(self.patterns, self.market)

    @staticmethod
    def map_probability_fns(patterns: dict) -> dict:
        return {pattern.item: generate_skill_up_fn(pattern.skill) for pattern in patterns}

    @staticmethod
    def augment_patterns_to_include_cost(patterns: dict, market: dict) -> dict:
        pw_cost = patterns.copy()
        for pat in pw_cost:
            mats = pat.materials
            for (name, _) in mats.items():
                unit_cost = market[name].market_value
                if unit_cost is None:
                    unit_cost = EMPTY_MAT_COST
                pat.cost[name] = unit_cost

        return pw_cost

    @staticmethod
    def map_sum_costs(patterns: dict, market: dict) -> dict:
        sum_cost_mapping = {}
        for pat in patterns:
            mats = pat.materials
            sum_cost = 0
            for (name, qty) in mats.items():
                unit_cost = market[name].market_value
                if unit_cost is None:
                    unit_cost = EMPTY_MAT_COST
                else:
                    unit_cost *= qty

                sum_cost += unit_cost

            sum_cost_mapping[pat.item] = sum_cost

        return sum_cost_mapping

    def step(self, level: int) -> SimulationStep:
        return SimulationStep(1, 1)

    def run_simulation(self, config: RunConfigs) -> None:
        for _ in range(config.simulations):
            current_lv = 0
            total_cost = 0
            while current_lv != MAX_CRAFTING_LV:
                step = self.step(current_lv)
                current_lv += step.level
                total_cost += step.cost
