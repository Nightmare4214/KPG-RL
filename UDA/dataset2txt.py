import os
from glob import glob

if __name__ == '__main__':
    root = '/home/icml007/Nightmare4214/datasets/OfficeHomeDataset_10072016'
    all_paths = [x for x in glob(os.path.join(root, '*')) if os.path.isdir(x)]
    all_classes = {cur_class:idx for idx, cur_class in enumerate(sorted(os.listdir(all_paths[0])))}
    print(all_classes)

    for path in all_paths:
        target = path + '.txt'
        with open(target, 'w') as f:
            for cur in glob(os.path.join(path, '*', '*.jpg')):
                cur_class = os.path.basename(os.path.dirname(cur))
                f.write(f'{os.path.relpath(cur, root)} {all_classes[cur_class]}\n')
        # target = path + '.txt'
        # with open(target, 'w') as f:
        #     for cur in glob(os.path.join(path, '*', '*.jpg')):
        #         f.write(cur + '\n')
        
