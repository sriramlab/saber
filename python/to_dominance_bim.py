import argparse

snp_id_index = 1


def to_dominance_bim(bim_path, out_path):
    assert bim_path != out_path, 'out_path cannot be the same as the original bim file path'
    print(f'=> converting {bim_path} to dominance bim out_path: {out_path}')
    with open(bim_path, 'r') as file, open(out_path, 'w') as out:
        for line in file:
            toks = line.split()
            toks[snp_id_index] = toks[snp_id_index] + '_dominance'
            out.write(f'{" ".join(toks)}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bim', type=str)
    parser.add_argument('out_path', type=str)
    args = parser.parse_args()
    to_dominance_bim(args.bim, args.out_path)
