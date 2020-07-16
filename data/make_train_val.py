import os
import random
import glob

root = ''
train_files = 'train.txt'
val_files = 'val.txt'
ratio = 0.9

files = os.listdir(root)

for file_dir in files:
    if file_dir.startswith('.'):
        continue

    img_files = glob.glob(os.path.join(root, file_dir) + '/*.jpg')
    random.shuffle(img_files)
    num = int(ratio * len(img_files))
    train = img_files[:num]
    train = '\n'.join(train)
    val = img_files[num:]
    val = '\n'.join(val)
    with open(train_files, 'a+') as f:
        f.writelines(train)
    with open(val_files, 'a+') as f2:
        f2.writelines(val)
