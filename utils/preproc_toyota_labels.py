#%%
import csv
import os
import math

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='/datasets/toyotasm', type=str)
parser.add_argument('--protocol', default='CS', type=str)

args = parser.parse_args()


dir1 = args.root
f = open(f'{dir1}/splits/test_{args.protocol}.txt', "r")
outfile = open(f'{dir1}/test_Labels_{args.protocol}.csv', "w")
outf = csv.writer(outfile, delimiter=',')
outf.writerow(['name', 'start', 'end'])

for i in f.readlines():
    try:
        n_frames = len(os.listdir(dir1 + '/frames/'+ os.path.splitext(i.strip())[0]))
    except Exception as e:
        print(e)
        continue
    div = int(math.ceil(n_frames//128))

    if div>0:
        for j in range(0, div):
            t = 128
            outf.writerow([os.path.splitext(i.strip())[0], j*t, (j+1)*t])
    else:
        outf.writerow([os.path.splitext(i.strip())[0], '0', int(n_frames)])

outfile.close()