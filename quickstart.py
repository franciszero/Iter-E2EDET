# import os
# import fiftyone as fo
# import fiftyone.zoo as foz
#
# dataset = foz.load_zoo_dataset("quickstart")
#
# # Classes list
# classes = dataset.distinct("ground_truth.detections.label")
#
# # The directory in which the dataset's images are stored
# IMAGES_DIR = os.path.dirname(dataset.first().filepath)
#
# # Export some labels in COCO format
# dataset.take(5).export(
#     dataset_type=fo.types.COCODetectionDataset,
#     label_field="ground_truth",
#     labels_path="./datasets/CrowdHuman/annotations/val.json",
#     classes=classes,
# )
#
# # Load COCO formatted dataset
# coco_dataset = fo.Dataset.from_dir(
#     dataset_type=fo.types.COCODetectionDataset,
#     data_path=IMAGES_DIR,
#     labels_path="./datasets/CrowdHuman/annotations/val.json",
#     include_id=True,
# )
#
# # Verify that the class list for our dataset was imported
# print(coco_dataset.default_classes)  # ['airplane', 'apple', ...]
#
# print(coco_dataset)
#
# # Visualize results in the App
# session = fo.launch_app(view=view)
# # Blocks execution until the App is closed
# session.wait()
#
# i = None
#



import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

dataset = foz.load_zoo_dataset("quickstart")
print(dataset)

# Evaluate the objects in the `predictions` field with respect to the
# objects in the `ground_truth` field
results = dataset.evaluate_detections(
    "predictions",
    gt_field="ground_truth",
    method="coco",
    eval_key="eval",
)

# Get the 10 most common classes in the dataset
counts = dataset.count_values("ground_truth.detections.label")
classes = sorted(counts, key=counts.get, reverse=True)[:10]

# Print a classification report for the top-10 classes
results.print_report(classes=classes)

# Print some statistics about the total TP/FP/FN counts
print("TP: %d" % dataset.sum("eval_tp"))
print("FP: %d" % dataset.sum("eval_fp"))
print("FN: %d" % dataset.sum("eval_fn"))

# Create a view that has samples with the most false positives first, and
# only includes false positive boxes in the `predictions` field
view = (
    dataset
    .sort_by("eval_fp", reverse=True)
    .filter_labels("predictions", F("eval") == "fp")
)

# Visualize results in the App
session = fo.launch_app(view=view)

# Blocks execution until the App is closed
session.wait()

i = None
