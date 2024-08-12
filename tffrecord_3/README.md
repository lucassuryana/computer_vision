# Create tf records

## Objective
This script processes a Waymo Open Dataset TFRecord file by extracting images and their associated annotations (bounding boxes and classes) from each frame. It then converts this data into a format compatible with TensorFlow's Object Detection API and saves the result as a new TFRecord file.

## How to Run
Please note that it is not possible to run the code with the actual Waymo Open Dataset as the data is proprietary to Udacity. However, this repository provides an example of how the process works.

To run the script, use the following command:
``` sh
cd tffrecord_3
```
``` sh
python convert.py -p training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord
```
The script will output a new TFRecord file in the `output` folder, where the data has been translated into the TensorFlow Object Detection API format.