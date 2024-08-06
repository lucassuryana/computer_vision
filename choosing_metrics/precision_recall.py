import numpy as np

from iou import calculate_ious
from utils import get_data


def precision_recall(ious, gt_classes, pred_classes):
    """
    calculate precision and recall
    args:
    - ious [array]: NxM array of ious
    - gt_classes [array]: 1xN array of ground truth classes
    - pred_classes [array]: 1xM array of pred classes
    returns:
    - precision [float]
    - recall [float]
    """
    # Set IOU Threshold
    threshold = 0.5
    # Compare each element with the threshold and assign scores
    result = ious > threshold

    # Use np.where to get the indices of elements that are True
    indices = np.where(result)
    
    # Step 5: Convert the indices to a list of tuples for easier iteration
    indices_list = list(zip(indices[0], indices[1]))
    TP = 0
    FP = 0

    # Step 6: Loop through the indices and perform desired operations
    for idx in indices_list:
        row, col = idx
        # If the classes of GT are the same as the classes of Prediction, then TP + 1
        if (gt_classes[row] == pred_classes[col]):
            TP += 1
        # If different, then FP + 1
        else:
            FP += 1

    # FN is the total number of ground truth that an algorithm fails to detect
    FN = len(gt_classes) - len(indices[0])

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return precision, recall


if __name__ == "__main__": 
    ground_truth, predictions = get_data()
    
    # get bboxes array
    filename = 'segment-1231623110026745648_480_000_500_000_with_camera_labels_38.png'
    gt_bboxes = [g['boxes'] for g in ground_truth if g['filename'] == filename][0]
    gt_bboxes = np.array(gt_bboxes)
    gt_classes = [g['classes'] for g in ground_truth if g['filename'] == filename][0]
    

    pred_bboxes = [p['boxes'] for p in predictions if p['filename'] == filename][0]
    pred_boxes = np.array(pred_bboxes)
    pred_classes = [p['classes'] for p in predictions if p['filename'] == filename][0]
    
    ious = calculate_ious(gt_bboxes, pred_boxes)
    precision, recall = precision_recall(ious, gt_classes, pred_classes)
    print(precision)
    print(recall)