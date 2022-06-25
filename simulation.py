from dataclasses import dataclass, KW_ONLY
from utils import generate_skill_up_fn, EMPTY_MAT_COST


@dataclass(frozen=True)
class RunConfigs:
    _: KW_ONLY
    sim_start_lv: int
    sim_end_lv: int
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

    def add_to_bank(self, name: str) -> None:
        self.cm_bank[name] += 1

    def pop_from_bank_if_exists(self, name: str) -> bool:
        val = self.cm_bank.get(name, 0)
        if val == 0:
            return False
        self.cm_bank[name] -= 1
        return True

    def step(self, level: int) -> SimulationStep:
        # If crafting material -> push to bank:
        # if not, check bank for a material that we've already
        # crafted -> if exists, pop and reduce cost to 0

        # Pre-compute probabilities across all available patterns
        probabilities = {}
        for (name, compute_prob) in self.probability_fn_mapping.items():
            probabilities[name] = compute_prob(level)

        # Filter out 0 probability patterns
        probabilities = dict(filter(lambda x: x[1] != 0.0, probabilities.items()))
        
        return SimulationStep(1, 1)

    def run_simulation(self, config: RunConfigs) -> None:
        for _ in range(config.simulations):
            current_lv = config.sim_start_lv
            total_cost = 0
            while current_lv != config.sim_end_lv:
                step = self.step(current_lv)
                current_lv += step.level
                total_cost += step.cost
