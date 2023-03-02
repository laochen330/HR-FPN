# HR-FPN

Because we set additional evaluation metrics, you will need to replace the cocoeval.py file in the “coco” folder with the cocoeval.py file in the “pycocotools” folder in your environment to ensure that the program runs correctly.

We give a weight file "best.pt" for a model trained on the tinyperson dataset, which is in "./runs/train/HR-FPN/weights/" directory.

If you want to run the model training program:
$ python train.py

If you want to test the models:
$ python val.py
