# Directory description

```
Data/
│
├── annotations/
│   ├── test_annotations.json
│   └── train_annotations.json
│
└── images/
    ├── train/
    └── test/
```
Data directory contains all files needed to train and test object detection model, in particular `annotations/` must contain two JSON files:

- `test_annotations.json`: JSON file containing annotations for test set in COCO format.
- `train_annotations.json`: JSON file containing annotations for training set in COCO format.

While `images/` must contain images used by the model divided in:

- `train/`
- `test/`