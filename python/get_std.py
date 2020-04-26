import numpy as np
import sys
import re

est = []
std = []
path = sys.argv[1]

with open(path, 'r') as file:
    for line in file:
        toks = line.split()
        est.append(float(toks[0]))
        m = re.match(r'\((.+)\)', toks[1])
        std.append(float(m.group(1)))

print(f'est mean: {np.mean(est):.5f} std: {np.std(est):.5f}')
print(f'SE est mean: {np.mean(std):.5f}')

