import os
import sys
from argparse import ArgumentParser

from keras.applications import InceptionResNetV2
from keras.applications.imagenet_utils import decode_predictions

from train.utils import inception_process_img

input_size = 299
image_ext = ['jpg', 'jpeg', 'png', 'gif']
model = InceptionResNetV2(include_top=True)


def main(dataset, out_dir):
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

                im = inception_process_img(image_file, 299)
                pred = model.predict(im)
                p = decode_predictions(pred, top=3)
                print(image_file, p[0])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('--out_dir', type=str, default='.')
    args = parser.parse_args(sys.argv[1:])
    main(args.dataset, args.out_dir)
