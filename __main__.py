import json


def main(args: dict) -> None:
    if args.profession_json_file is None:
        raise Exception('Must use command "-f" and pass in a '
                        + ' JSON file containing valid profession data.')


    profession_data_list = []
    with open(args.profession_json_file, 'r', encoding='utf8') as file:
        profession_data_list = json.load(file)

    print(profession_data_list)


if __name__ == "__main__":
    import argparse

    PARSER = argparse.ArgumentParser(
        description='Runs a Monte Carlo Simulation on WoW Classic profession data.')
    
    PARSER.add_argument(
        '-f',
        '--file',
        dest='profession_json_file',
        help='JSON file containing profession data'
    )

    main(PARSER.parse_args())