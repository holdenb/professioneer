import json
from functools import reduce
import asyncio
import aiohttp
from models import Profession, MarketItem
from ratelimiter import RateLimiter


@RateLimiter(max_calls=20, period=5)
async def request_price_data(url: str, session: aiohttp.ClientSession) -> dict:
    async with session.get(url) as resp:
        prices = await resp.json()
        prices = MarketItem.from_dict(prices)
        # We only care about the most recent market data, i.e. the last
        # one in the market_data list
        print(prices.data[-1])


async def main(args: dict) -> None:
    if args.profession_json_file is None:
        raise Exception('Must use command "-f" and pass in a '
                        + ' JSON file containing valid profession data.')

    professions = [Profession]
    with open(args.profession_json_file, 'r', encoding='utf-8') as file:
        professions = Profession.schema().load(json.load(file), many=True)

    total_unique_materials = list(
        reduce(
            lambda a, b: a | b,
            map(lambda prof: set(list(prof.materials.keys())), professions)
            )
        )

    # Format the output from the profession JSON to fit the input for nexus-hub API
    # i.e. 'Elemental Fire' -> 'elemental-fire'
    total_unique_materials = list(
        map(lambda a: a.lower().replace(' ', '-'), total_unique_materials)
    )

    # Note: We are currently rate limited by nexus-hub @ 20 requests per 5s so we
    # need to account for this
    async with aiohttp.ClientSession() as session:
        url = 'https://api.nexushub.co/wow-classic/v1/items/grobbulus-alliance/felsteel-bar/prices'
        for i in range(1, 100):
            await request_price_data(url, session)
            print(i)


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser(
        description='This script is used to gather market data for profession materials.')

    PARSER.add_argument(
        '-f',
        '--file',
        dest='profession_json_file',
        help='JSON file containing profession data'
    )

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(PARSER.parse_args()))
