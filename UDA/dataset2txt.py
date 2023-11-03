import os
from glob import glob

if __name__ == '__main__':
    root = '/home/icml007/Nightmare4214/datasets/OfficeHomeDataset_10072016'
    for path in glob(os.path.join(root, '*')):
        if not os.path.isdir(path):
            continue
        target = path + '.txt'
        with open(target, 'w') as f:
            for cur in glob(os.path.join(path, '*', '*.jpg')):
                f.write(cur + '\n')
