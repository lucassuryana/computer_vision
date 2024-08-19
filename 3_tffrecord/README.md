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

## Example of Data Transformation
### Original Waymo TFRecord Annotation
Here is an example of how an annotation might look in the original Waymo TFRecord format:

``` plaintext
ann box {
  center_x: 785.69796
  center_y: 681.48561
  width: 33.47427000000005
  length: 25.263599999999997
}
type: TYPE_VEHICLE
id: "7c79b738-419e-4ebf-acae-e8d26a102eaa"
detection_difficulty_level: LEVEL_2
tracking_difficulty_level: LEVEL_2
```

### Converted TFRecord for TensorFlow Object Detection API
After processing, the same annotation would be converted into the TensorFlow Object Detection API format as follows:
``` python
encoded_jpg_io = io.BytesIO(encoded_jpeg)
image = Image.open(encoded_jpg_io)
width, height = image.size

mapping = {1: 'vehicle', 2: 'pedestrian', 4: 'cyclist'}
image_format = b'jpg'
xmins = []
xmaxs = []
ymins = []
ymaxs = []
classes_text = []
classes = []
filename = filename.encode('utf8')

for ann in annotations:
    xmin, ymin = ann.box.center_x - 0.5 * ann.box.length, ann.box.center_y - 0.5 * ann.box.width
    xmax, ymax = ann.box.center_x + 0.5 * ann.box.length, ann.box.center_y + 0.5 * ann.box.width
    xmins.append(xmin / width)
    xmaxs.append(xmax / width)
    ymins.append(ymin / height)
    ymaxs.append(ymax / height)    
    classes.append(ann.type)
    classes_text.append(mapping[ann.type].encode('utf8'))

tf_example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': int64_feature(height),
    'image/width': int64_feature(width),
    'image/filename': bytes_feature(filename),
    'image/source_id': bytes_feature(filename),
    'image/encoded': bytes_feature(encoded_jpeg),
    'image/format': bytes_feature(image_format),
    'image/object/bbox/xmin': float_list_feature(xmins),
    'image/object/bbox/xmax': float_list_feature(xmaxs),
    'image/object/bbox/ymin': float_list_feature(ymins),
    'image/object/bbox/ymax': float_list_feature(ymaxs),
    'image/object/class/text': bytes_list_feature(classes_text),
    'image/object/class/label': int64_list_feature(classes),
}))
```

In this example:
* The bounding box coordinates are normalized by the image dimensions.
* The object type is mapped to a text label (e.g., vehicle) and stored as both text and an integer label.
* The image's height, width, and filename are also included in the converted TFRecord.
This transformation makes the data ready for training a TensorFlow Object Detection model.

