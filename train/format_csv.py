import math
import sys
import random


def main(file_name, decimal=7, target_file=None):
    factor = 10 ** int(decimal)
    result = []
    with open(file_name, 'r', encoding='utf-8') as f:
        ignor_head = f.readline()
        lines = f.readlines()
        for line in lines:
            print(line)
            id_, label = line.split(',')
            label = math.ceil(float(label) * factor) / factor
            if math.isclose(0, label):
                # label = random.uniform(0, 0.000001)
                label = 0.0005
            elif math.isclose(1, label):
                # label = random.uniform(0.999992, 1)
                label = 0.9995
            result.append((int(id_), label))

    result.sort(key=lambda x: x[0])
    if target_file is None:
        target_file = 'submission_{}.csv'.format(decimal)
    with open(target_file, 'w', encoding='utf-8') as file:
        file.write('id,label\n')
        for id_, label in result:
            file.write('{},{}\n'.format(str(id_), str(label)))


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args[0], args[1], args[2])
