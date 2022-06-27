from collections import namedtuple


# Crafting material category name
CATEGORY_NAME_MATERIAL = "Crafting material"

# Pattern color skill-up probabilities
PROB_ORANGE = 1.00
PROB_YELLOW = 0.75
PROB_GREEN = 0.25
PROB_ZERO = 0.0

# For now we use this to penalize materials that have no
# quantity
EMPTY_MAT_COST = 999999999


def compute_wci(sum_cost: float, p_crafting: float, available_mats: bool=True) -> float:
    return (p_crafting * int(available_mats)) / sum_cost


def generate_skill_up_fn(pattern_skill: dict):
    lb_orange = int(pattern_skill['Orange'])
    lb_yellow = int(pattern_skill['Yellow'])
    lb_green = int(pattern_skill['Green'])
    lb_gray = int(pattern_skill['Gray'])

    # A skill threshold indicates the lower and upper bound
    # values from which a pattern is still viable to
    # receive a "skill point" from
    SkillThreshold = namedtuple('SkillThreshold', ['lb', 'ub'])

    def skill_up_fn(level: int) -> float:
        # Anything below the orange lb is not able to be crafted
        # Anything after the gray lb value is considered not viable
        orange = SkillThreshold(lb_orange, lb_yellow - 1)
        yellow = SkillThreshold(lb_yellow, lb_green - 1)
        green = SkillThreshold(lb_green, lb_gray - 1)

        if orange.ub >= level >= orange.lb:
            return PROB_ORANGE
        if yellow.ub >= level >= yellow.lb:
            return PROB_YELLOW
        if green.ub >= level >= green.lb:
            return PROB_GREEN

        return PROB_ZERO

    return skill_up_fn
