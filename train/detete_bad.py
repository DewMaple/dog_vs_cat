import os
import sys

file_2_delete = [
    "dog.10801.jpg",
    "dog.10161.jpg",
    "dog.12376.jpg",
    "cat.10365.jpg",
    "cat.7377.jpg",
    "cat.3004.jpg",
    "dog.9188.jpg",
    "cat.3672.jpg",
    "cat.4338.jpg",
    "dog.4507.jpg",
    "cat.7372.jpg",
    "cat.10712.jpg",
    "cat.10029.jpg",
    "dog.1194.jpg",
    "dog.5604.jpg",
    "cat.11777.jpg",
    "cat.8456.jpg",
    "cat.2520.jpg",
    "cat.4688.jpg",
    "dog.10747.jpg",
    "dog.11266.jpg",
    "dog.2614.jpg",
    "cat.9171.jpg",
    "cat.5071.jpg",
    "dog.1043.jpg",
    "cat.12476.jpg",
    "cat.9983.jpg",
    "dog.3889.jpg",
    "dog.6475.jpg",
    "cat.7682.jpg",
    "dog.10190.jpg",
    "cat.10636.jpg",
    "dog.8898.jpg",
    "cat.11184.jpg",
    "cat.3216.jpg",
    "cat.7968.jpg",
    "dog.4367.jpg",
    "dog.9517.jpg",
    "dog.11299.jpg",
    "dog.6725.jpg",
    "cat.7564.jpg",
    "cat.12424.jpg",
    "cat.8470.jpg",
    "cat.4308.jpg",
    "cat.7464.jpg",
    "dog.8736.jpg",
    "cat.2939.jpg",
    "cat.5418.jpg",
    "dog.2422.jpg",
    "dog.1773.jpg",
    "dog.1895.jpg",
    "cat.6345.jpg",
    "cat.5351.jpg",
    "dog.4127.jpg",
    "dog.10237.jpg",
    "dog.1259.jpg"
]


def main(files_dir):
    for root, dirs, files in os.walk(files_dir):
        for f in files:
            if f in file_2_delete:
                os.remove(os.path.join(files_dir, f))


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args[0])
