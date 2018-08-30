import os
import sys
from argparse import ArgumentParser

import cv2
import numpy as np

from models.resetnet import build_resnet50_model
from train.utils import read_img, save_img, put_txt

image_ext = ['jpg', 'jpeg', 'png', 'gif']

input_size = 224


def load(weight_path):
    model = build_resnet50_model(2, input_size)
    model.load_weights(weight_path, by_name=True, skip_mismatch=True)
    model.summary()
    return model


def main(weight_path, dataset, out_dir):
    model = load(weight_path=weight_path)

    count = 0
    results = []
    for root, dirs, files in os.walk(dataset):
        for im_f in files:
            split = im_f.split(os.extsep)
            fname = split[0]
            ext = split[-1]
            count += 1
            if count % 500 == 0:
                print("{} done.".format(count))
            if ext in image_ext:
                image_file = os.path.join(dataset, im_f)
                im = read_img(image_file, (input_size, input_size), rescale=1 / 255.)
                pred = model.predict(im)
                idx = np.argmax(pred, axis=1)
                f_name = os.path.join(out_dir, '{}'.format(im_f))
                img = cv2.imread(image_file)
                txt1 = 'cat: {0:.06f}%'.format(pred[0][0] * 100)
                txt2 = 'dog: {0:.06f}%'.format(pred[0][1] * 100)
                img = put_txt(img, txt1, (10, 30), (0, 255, 0))
                img = put_txt(img, txt2, (10, 60), (0, 255, 0))
                results.append((int(fname), idx[0]))
                save_img(img, f_name)

    results.sort(key=lambda x: x[0])
    with open(os.path.join(out_dir, 'submission.csv'), 'w', encoding='utf-8') as f_submission:
        f_submission.write('id,label\n')
        for id, label in results:
            f_submission.write('{},{}\n'.format(str(id), str(label)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('weight_path', type=str)
    parser.add_argument('dataset', type=str)
    parser.add_argument('--out_dir', type=str, default='.')
    args = parser.parse_args(sys.argv[1:])
    main(args.weight_path, args.dataset, args.out_dir)
