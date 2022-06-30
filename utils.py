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
    """
    Computes the Worthy Crafting Index (WCI) based on cost, probability of
    crafting, and if we have available materials

    Args:
        sum_cost (float): Summation of costs of materials for a pattern
        p_crafting (float): Probability of crafting the pattern
        available_mats (bool, optional): Do we have available materials? Defaults to True.

    Returns:
        float: The WCI score (The goal is to maximize this score)
    """
    return (p_crafting * int(available_mats)) / sum_cost


def generate_skill_up_fn(pattern_skill: dict):
    """
    Generates a skill up curried function based on the skill
    range of a specific crafting pattern

    Args:
        pattern_skill (dict): Dictionary of pattern {skill: value}
        Ex: {"Orange": "20"}

    Returns:
        function: A function that computes the probability of leveling
        based on captured values from that patterns skill range
    """
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
