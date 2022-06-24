import json
from functools import reduce
import asyncio
import aiohttp
from datamodels import CraftingPattern, MarketItem, MarketData
from ratelimiter import RateLimiter


NEXUS_HUB_URL_ITEMS_BASE = 'https://api.nexushub.co/wow-classic/v1/items'


def nexus_hub_price_url(server: str, faction: str, item_unique_name: str) -> str:
    return f'{NEXUS_HUB_URL_ITEMS_BASE}/{server}-{faction}/{item_unique_name}/prices'


@RateLimiter(max_calls=20, period=6)
async def request_price_data(url: str, session: aiohttp.ClientSession) -> dict:
    async with session.get(url) as resp:
        prices = await resp.json()
        print(prices)
        return MarketItem.from_dict(prices)


async def main(args: dict) -> None:
    crafting_patterns = [CraftingPattern]
    with open(args.profession_json_file, 'r', encoding='utf-8') as file:
        crafting_patterns = CraftingPattern.schema().load(json.load(file), many=True)

    total_unique_materials = list(
        reduce(
            lambda a, b: a | b,
            map(lambda prof: set(list(prof.materials.keys())), crafting_patterns)
            )
        )

    # Format the output from the profession JSON to fit the input for nexus-hub API
    # i.e. 'Elemental Fire' -> 'elemental-fire'
    total_unique_materials = list(
        map(lambda a: a.lower().replace(' ', '-'), total_unique_materials)
    )

    item_price_mapping = {}

    # Note: We are currently rate limited by nexus-hub @ 20 requests per 5s so we
    # need to account for this
    async with aiohttp.ClientSession() as session:
        for unique_item_name in total_unique_materials:
            url = nexus_hub_price_url(args.server, args.faction, unique_item_name)
            market_item = await request_price_data(url, session)

            # We only care about the most recent market data, i.e. the last
            # one in the market_data list
            data = market_item.data[-1] \
                if market_item.data else MarketData(None, None, None, None)
            data = MarketData.schema().dump(data)

            item_price_mapping[market_item.name] = data

    # Transform path/possible path to file into a filename prefix:
    # i.e. ie data/path/to/prof-engineering.json -> prof-engineering
    filename_prefix = args.profession_json_file.split('.')[:-1][0].split('/')[-1]

    with open(f'{filename_prefix}-{args.server}-{args.faction}-market-data.json', 'w', encoding='utf-8') as file:
        json.dump(item_price_mapping, file)


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser(
        description='This script is used to gather market data for profession materials.')

    PARSER.add_argument(
        '-j',
        '--json',
        dest='profession_json_file',
        help='JSON file containing profession data',
        required=True
    )
    PARSER.add_argument(
        '-s',
        '--server',
        dest='server',
        help='Server to pull data from',
        required=True
    )
    PARSER.add_argument(
        '-f',
        '--faction',
        dest='faction',
        help='Faction to pull data from',
        required=True
    )

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(PARSER.parse_args()))
