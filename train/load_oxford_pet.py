import os
import sys
from argparse import ArgumentParser
from shutil import copyfile


def main(oxford_pet_dir, oxford_pet_anno_file, target_dir):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    dog_dir = os.path.join(target_dir, 'dog')
    if not os.path.exists(dog_dir):
        os.mkdir(dog_dir)

    cat_dir = os.path.join(target_dir, 'cat')
    if not os.path.exists(cat_dir):
        os.mkdir(cat_dir)

    with open(oxford_pet_anno_file, 'r', encoding='utf-8') as f:
        for i in range(0, 6):
            _ = f.readline()
        lines = f.readlines()

        for line in lines:
            split = line.split(' ')
            f_name = split[0]
            f_name = '{}.jpg'.format(f_name)
            print('{}'.format(file_name))

            dog_or_cat = split[2]

            file_name = os.path.join(oxford_pet_dir, f_name)
            if dog_or_cat == '1':
                copyfile(file_name, os.path.join(cat_dir, f_name))
            elif dog_or_cat == '2':
                copyfile(file_name, os.path.join(dog_dir, f_name))
            else:
                raise ValueError('Unknown species of {}'.format(file_name))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('oxford_pet_dir', type=str, help='Directory of Oxford-IIIT Pet dataset')
    parser.add_argument('oxford_pet_anno_file', type=str, help='Annotation file of Oxford-IIIT Pet dataset')
    parser.add_argument('target_dir', type=str, help='Directory to save output')
    args = parser.parse_args(sys.argv[1:])
    main(args.oxford_pet_dir, args.oxford_pet_anno_file, args.target_dir)
