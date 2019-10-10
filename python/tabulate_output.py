import argparse
import re
import os


def tabulate(file_path):
    started = False
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r'heritability estimates for (.+):', line)
            if match:
                started = True
                trait = match.groups()[0]
                print(f'\n{os.path.basename(trait)},', end='')
            if started and re.match(r'partition named|total estimate', line):
                tabulate_trait_estimates(file)
    print()


decimal_regex = r'[+-]?(([1-9][0-9]*)?[0-9](\.[0-9]*)?|\.[0-9]+)'


def tabulate_trait_estimates(file):
    point_estimate_without_jackknife = float(
        re.search(decimal_regex, file.readline()).group(0)
    )
    file.readline()
    file.readline()
    standard_error = float(
        re.search(decimal_regex, file.readline()).group(0)
    )
    file.readline()

    print(f'{point_estimate_without_jackknife:.5f} ({standard_error:.5f})', end=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', type=str)
    args = parser.parse_args()
    tabulate(args.filepath)
