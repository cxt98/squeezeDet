
- SqueezeDet is a real-time object detector, which can be used to detect videos. The video demo will be released later.

## Training/Validation:
- Download KITTI object detection dataset: [images](http://www.cvlibs.net/download.php?file=data_object_image_2.zip) and [labels](http://www.cvlibs.net/download.php?file=data_object_label_2.zip). Put them under `$SQDT_ROOT/data/KITTI/`. Unzip them, then you will get two directories:  `$SQDT_ROOT/data/KITTI/training/` and `$SQDT_ROOT/data/KITTI/testing/`. 

models: SqueezeDet, SqueezeDet+, VGG16+ConvDet, ResNet50+ConvDet. 
  ```Shell
  cd $SQDT_ROOT/
  ./scripts/train.sh -net (squeezeDet|squeezeDet+|vgg16|resnet50) -train_dir /tmp/bichen/logs/squeezedet -gpu 0
  ```

  Training logs are saved to the directory specified by `-train_dir`. GPU id is specified by `-gpu`. Network to train is specificed by `-net` 

- Before evaluation, you need to first compile the official evaluation script of KITTI dataset
  ```Shell
  cd $SQDT_ROOT/src/dataset/kitti-eval
  make
  ```

- Finally, to monitor training and evaluation process, you can use tensorboard by

  ```Shell
  tensorboard --logdir=$LOG_DIR
  ```
  Here, `$LOG_DIR` is the directory where your training and evaluation threads dump log events, which should be the same as `-train_dir` and `-eval_dir` specified in `train.sh` and `eval.sh`. From tensorboard, you should be able to see a lot of information including loss, average precision, error analysis, example detections, model visualization, etc.

## Evalution
- After training, we test the model on testing dataset and generate heatmap and bounding boxes. To do so
  ```Shell
  python ./src/demo.py --mode image --input_path './data/progress_kitti/testing/image_2/exp*.png' --checkpoint /tmp/logs/SqueezeDet/train/model.ckpt-11000
  ```

  '''Shell
  python ./src/demo.py --mode image --input_path './data/progress_kitti/testing/image_2/exp*.png' --checkpoint ./data/model_checkpoints/squeezeDet/model.ckpt-12000
'''

## Visualization
- We can visualize either all bbox above certain confidence threshold or heatmap for any class. To do so
  ```Shell
  python ./src/viz_utils.py --mode [heatmap/bbox] --input_path ./data/progress_kitti/testing/image_2/exp14.png --checkpoint /tmp/logs/SqueezeDet/train/model.ckpt-11000 --test_cls ranch
  ```
