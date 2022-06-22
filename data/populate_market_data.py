import json
from models import Profession


def main(args: dict) -> None:
    if args.profession_json_file is None:
        raise Exception('Must use command "-f" and pass in a '
                        + ' JSON file containing valid profession data.')

    professions = []
    with open(args.profession_json_file, 'r', encoding='utf-8') as file:
        professions = Profession.schema().load(json.load(file), many=True)

    print(professions)

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

    main(PARSER.parse_args())
