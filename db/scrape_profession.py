import re
import requests
import sys
from bs4 import BeautifulSoup
from math import ceil
from string import printable


def main(args: dict) -> None:
    """[summary]
    Arguments:
        args {dict} -- [description]
    Raises:
        Exception: [description]
    """
    if args.profession is None:
        raise Exception('Must use command "-p" and a profession name.')

    url = f'https://wow.gamepedia.com/Classic_{args.profession}_schematics'

    try:
        resp = requests.get(url)
    except requests.RequestException as r_e:
        print(r_e)
        sys.exit(f'Unable to acquire a response from: {url}')

    soup = BeautifulSoup(resp.text, features='html.parser')
    print(url)

    # Grab the last table
    datatable = soup.findAll('table')[-1]
    row_data = datatable.findAll('tr')

    header_data = [x.get_text().rstrip() for x in row_data[0].findAll('th')]
    print(f'Header data found: {header_data}')

    header_sub_data = [x.get_text().rstrip() for x in row_data[1].findAll('th')]
    print(f'Header sub-data found: {header_sub_data}')
    '''
    Example schema of an item that will be scraped from the datatable:
    {
        item_name: Rough Blasting Powder,
        category: Crafting Material
        materials: {
            Rough Stone: 1
        },
        skill: {
            orange: 1,
            yellow: 20,
            green: 30,
            gray: 40
        },
        source: Trainer
    }

    We will end up with a list of these dictionaries on a successful scrape
    '''
    # All scraped items
    items = []

    index = 2
    for i in range(len(row_data) - index):
        # Iterate over row data in the table
        tr_data = row_data[index]
        index += 1

        # Scraped data for a specific item
        item_data = {k: '' for k in header_data}
        # Mold item data to our schema
        item_data[header_data[3]] = {k: 0 for k in header_sub_data}

        td_data = tr_data.findAll('td')

        if len(td_data) == (len(header_data) + len(header_sub_data) - 1):
            print(f'Found data to match headers for row: {index - 2}')
        else:
            sys.exit(f'Unable to find data to match headers at row: {index - 2}')

        # We will need to scrape line by line

        # Scrape the title data
        title_data = td_data[0]
        title_a_data = title_data.find('a')
        item_data[header_data[0]] = title_a_data.get('title')
        print(f'Scraping data for item: {item_data[header_data[0]]}')

        # Scrape the category data
        item_data[header_data[1]] = td_data[1].get_text()

        # Scrape the materials list
        materials_list_data_string = td_data[2].get_text()
        # Parse out the string
        # Ex: 1x [Item 1]2x [Item2]3x [Item3]
        for char in materials_list_data_string:
            if char.isdigit():
                materials_list_data_string = \
                    materials_list_data_string.replace(char, ' ' + char)

        materials_list = materials_list_data_string.lstrip()
        # Build a regular expression to capture data in between '[]'

        materials_groups = []
        while True:
            match = re.search(r'[^[]*\[([^]]*)\]', materials_list)
            if match is None:
                break

            group = match.group(1)
            materials_groups.append(group)

            removal_str = '[' + group + ']'
            materials_list = materials_list.replace(removal_str, '')

        # TODO implement some error handling here
        # # Catch errors in the table if the item has been added to the same location as
        # # it's amount
        # error_list_check_no_hidden_chars = ''.join(char for char in materials_list if char in printable)
        # error_list_check_no_hidden_chars = error_list_check_no_hidden_chars.replace(' ', '')
        # print(error_list_check_no_hidden_chars)

        # hidden_materials = []
        # for i, char in enumerate(error_list_check_no_hidden_chars):
        #     try:
        #         if char == 'x' and error_list_check_no_hidden_chars[i+1].isdigit():
        #             index = 1
        #             material = ''
        #             while error_list_check_no_hidden_chars[index] is not None:
        #                 material += error_list_check_no_hidden_chars[index]
        #                 index += 1
        #     except Exception as e:
        #         print(e)

        # print("hidden materials")
        # print(hidden_materials)

        # error_list_check = list(filter(lambda x: 'x' not in x, error_list_check_no_hidden_chars.split(' ')))
        # print(error_list_check)

        # Make an exception for broken dark iron rifle...
        if item_data[header_data[0]] == 'Dark Iron Rifle':
            materials_groups.append('Thorium Tube')

        # Now that we have a group of items, we need to get
        # the amount needed.
        # Filter on the materials for strings that contain 'x'
        materials_count = list(filter(lambda x: 'x' in x, materials_list.split(' ')))

        # Remove all 'x' chars so we can convert to integers
        materials_count = list(map(lambda x: int(x.replace('x', '')), materials_count))

        assert len(materials_groups) == len(materials_count)

        # Create the materials dictionary
        materials_dict = {materials_groups[i]:materials_count[i]
                          for i in range(len(materials_groups))}

        item_data[header_data[2]] = materials_dict

        # Scrape the skill levels
        sub_data_td_start_pos = 3
        for j in range(len(header_sub_data)):
            item_data[header_data[3]][header_sub_data[j]] = \
                td_data[sub_data_td_start_pos].find('span').get_text()

            sub_data_td_start_pos += 1

        # We are not worried about scraping source data for this
        items.append(item_data)

    print(items)


if __name__ == "__main__":
    import argparse


    # TODO use asyncio to capture all professions asynchronously
    # Select from list of professions
    AVAILABLE_PROFESSIONS = [
        'alchemy',
        'blacksmithing',
        'enchanting',
        'engineering',
        'tailoring',
    ]

    PARSER = argparse.ArgumentParser(
        description='This script is used to scrape wowhead to build profession databases.')
    PARSER.add_argument('-p', '--profession', choices=AVAILABLE_PROFESSIONS)

    main(PARSER.parse_args())
