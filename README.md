# Dog VS Cat
Kaggle competition: Dogs vs. Cats Redux: Kernels Edition

## Usage

- Format dataset
- training
  ```shell
  python train/train_resnet50.py $train_dataset_dir $validation_dataset_dir --batch_size 256 --epochs 100 --lr 0.001
  ```
- evaluate
   ```shell
  python train/evaluate.py weights_file_path $test_dataset_dir --out_dir $out_dir
  ```
