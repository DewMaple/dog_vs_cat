# Dog VS Cat
Kaggle competition: Dogs vs. Cats Redux: Kernels Edition

## Usage

### Format dataset
  $dog_vs_cat_dataset_dir: path of train directory in the dog vs cat dataset downloaded from kaggle, after you unzip the all.zip
  $target_dir: the directory where you want to put your output

  ```shell
  python train/format_dataset.py $dog_vs_cat_dataset_dir $target_dir --val_ratio 0.1
  ```

### training

  - train resnet50

    ```shell
    python train/train_resnet50.py $train_dataset_dir $validation_dataset_dir --batch_size 256 --epochs 100 --lr 0.001
    ```
  - train resnet50 with center loss
    ```shell
    python train/train_resnet50_with_center_loss.py $train_dataset_dir $validation_dataset_dir --batch_size 256 --epochs 100 --lr 0.001
    ```

    - evaluate
    $test_dataset_dir: path of test dataset directory of kaggle dog vs cat dataset
    
    ```shell
    python train/evaluate.py weights_file_path $test_dataset_dir --out_dir $out_dir
    ```
