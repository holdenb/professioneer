import math
import random
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, KW_ONLY
from data.datamodels import CraftingPattern
from utils import compute_wci, generate_skill_up_fn, EMPTY_MAT_COST, CATEGORY_NAME_MATERIAL


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
    pattern_name: str


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
        return {name: generate_skill_up_fn(p.skill) for (name, p) in patterns.items()}

    @staticmethod
    def augment_patterns_to_include_cost(patterns: list , market: dict) -> dict:
        for pat in patterns:
            mats = pat.materials
            for (name, _) in mats.items():
                unit_cost = market[name].market_value
                if unit_cost is None:
                    unit_cost = EMPTY_MAT_COST
                pat.cost[name] = unit_cost

        # Convert to mapping before returning
        patterns = {p.item: p for p in patterns}

        return patterns

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
        if name not in self.cm_bank.keys():
            self.cm_bank[name] = 1
            return
        self.cm_bank[name] += 1

    def pop_materials_from_bank_if_exists(self, pattern: CraftingPattern) -> None:
        for (name, qty) in pattern.materials.items():
            ct_in_bank = self.cm_bank.get(name)
            if ct_in_bank is None:
                continue

            if ct_in_bank > qty:
                self.cm_bank[name] -= qty
            else:
                del self.cm_bank[name]

    def is_in_bank(self, name: str) -> bool:
        return name in self.cm_bank.keys()

    def compute_score(self, pattern: CraftingPattern, prob_crafting: int, bank_fn):
        pattern_cost = 0
        for (name, qty) in pattern.materials.items():
            in_bank = bank_fn(name)
            pattern_cost += 0 if in_bank else pattern.cost[name] * qty

        # For now we'll assume all materials are always available
        return (compute_wci(pattern_cost, prob_crafting), pattern_cost)

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

        def skip_bank_check(_) -> bool:
            return False

        # Compute cost for all patterns that match those where P != 0
        # and then use the cost to create a WCI score (worthy crafting index score)
        # for each of those patterns
        # Mapping from {name -> (wci, cost)} tuples
        wci_mapping = {}
        for (k, p_crafting) in probabilities.items():
            pattern = self.patterns[k]
            is_not_m_material = pattern.category != CATEGORY_NAME_MATERIAL

            # Return False from bank if material is not used in
            # future crafting scenarios
            #
            # Category will not relate to crafting, thus we know that this pattern
            # may include material=Crafting within the pattern
            bank_fn = self.is_in_bank \
                if is_not_m_material else skip_bank_check

            (score, cost) = self.compute_score(pattern, p_crafting, bank_fn)
            wci_mapping[k] = (score, cost)

        # Select the max cost from the wci_mapping as the pattern that
        # we will choose to craft. Remember: Max cost means we're MAXIMIZING a cost function
        # I.e. the cost function will give us the highest output for low cost/high prob patterns
        (name, (score, cost)) = max(wci_mapping.items(), key=lambda x: x[1])

        chosen_pattern = self.patterns[name]
        # If the chosen pattern is a crafting material, add that material
        # to our bank to augment future pattern WCI score
        is_m_material = chosen_pattern.category == CATEGORY_NAME_MATERIAL
        if is_m_material:
            self.add_to_bank(name)
        else:
            self.pop_materials_from_bank_if_exists(chosen_pattern)

        # Simulate a crafting scenario where we "roll" against the
        # probability of crafting the chosen pattern with max WCI score
        p_craft = probabilities[name] * 10 * 100
        p_roll = random.randrange(1, 1001)
        should_lv = p_roll <= p_craft
        updated_lv = level + 1 if should_lv else level

        # Cost will stay the same regardless if we are "able" to
        # craft the pattern
        return SimulationStep(updated_lv, cost, name)

    def run_simulation(self, config: RunConfigs) -> None:
        sim_costs = {}
        for i in range(config.simulations):
            current_lv = config.sim_start_lv
            total_cost = 0
            crafting_path = []
            while current_lv != config.sim_end_lv:
                step = self.step(current_lv)
                # Replace level & add the cost
                current_lv = step.level
                total_cost += step.cost
                crafting_path.append(step.pattern_name)

            cost_gold = ((total_cost / 100) / 100)
            sim_costs[i+1] = cost_gold

            # print(f'lv: {current_lv} | cost: {cost_gold}')

            # # Frequency count for debugging
            # freq = {}
            # for item in crafting_path:
            #     if item in freq:
            #         freq[item] += 1
            #     else:
            #         freq[item] = 1
            # print(f'Path: {freq}')

            # crafting_path_df = pd.DataFrame(list(zip(freq.keys(), freq.values())), columns =['pattern', 'num_crafted'])
            # sns.set_theme(style="whitegrid")
            # sns.barplot(x="num_crafted", y="pattern", data=crafting_path_df)
            # plt.show()

        # sim_costs = dict(sorted(sim_costs.items(), key=lambda x: x[1]))
        # sim_costs_df = pd.DataFrame(list(zip(sim_costs.keys(), sim_costs.values())), columns =['sim_run', 'cost_g'])
        # sns.set_theme(style="whitegrid")
        # sns.barplot(x="sim_run", y="cost_g", data=sim_costs_df)
        # plt.show()
            print(sim_costs)

        sim_costs_df = pd.DataFrame(list(sim_costs.values()), columns=['cost_g'])
        sns.histplot(sim_costs_df.cost_g, kde=True)
        plt.show()
