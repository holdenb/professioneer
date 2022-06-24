from collections import namedtuple


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
            return 1.00
        if yellow.ub >= level >= yellow.lb:
            return 0.75
        if green.ub >= level >= green.lb:
            return 0.25

    return skill_up_fn
