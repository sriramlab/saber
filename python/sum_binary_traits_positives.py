import argparse


def count_positives(path, index, positive_label_list, skip_first_n_lines):
    positive_labels = set(positive_label_list)
    num_positives = 0
    total_num_samples = 0
    with open(path, 'r') as file:
        if skip_first_n_lines:
            _ = [file.readline() for _ in range(skip_first_n_lines)]
        for line in file:
            if not line:
                continue
            total_num_samples += 1
            num_positives += line.split()[index] in positive_labels
    return num_positives, total_num_samples


def run(path, index, positive_label_list, skip_first_n_lines):
    num_positives, total_num_samples = count_positives(path, index, positive_label_list, skip_first_n_lines)
    print(
        f'num_positives: {num_positives}\n'
        f'total_num_samples: {total_num_samples}\n'
        f'positive_ratio: {num_positives / total_num_samples:.5f}'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument(
        '-i', dest='index', type=int, required=True,
        help='0-based label field index'
    )
    parser.add_argument(
        '-l', dest='positive_label_list', nargs='+', required=True,
        help='list of positive labels, separated by space'
    )
    parser.add_argument(
        '--skip-first', '-s', dest='skips', type=int, metavar='N',
        help='skip the first N lines'
    )
    args = parser.parse_args()
    run(args.path, args.index, args.positive_label_list, args.skips)
