import math
import sys


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
