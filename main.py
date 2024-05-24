import json
from models import CraftingPattern, MarketData
from sim import Simulation, RunConfigs


def main(args: dict) -> None:
    crafting_patterns = [CraftingPattern]
    with open(args.profession_json_file, "r", encoding="utf-8") as file:
        crafting_patterns = CraftingPattern.schema().load(
            json.load(file), many=True
        )

    market_data = {}
    with open(args.market_data_file, "r", encoding="utf-8") as file:
        market_data = json.load(file)
        market_data = {
            k: MarketData.schema().load(v) for k, v in market_data.items()
        }

    sim = Simulation(crafting_patterns, market_data)
    sim.run_simulation(
        RunConfigs(sim_start_lv=1, sim_end_lv=300, simulations=10)
    )


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser(
        description="Runs a Monte Carlo Simulation on WoW Classic profession data."
    )

    PARSER.add_argument(
        "-p",
        "--profession",
        dest="profession_json_file",
        help="JSON file containing profession data",
        required=True,
    )
    PARSER.add_argument(
        "-m",
        "--market",
        dest="market_data_file",
        help="JSON file containing market data related to a profession",
        required=True,
    )

    main(PARSER.parse_args())
