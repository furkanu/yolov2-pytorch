from __future__ import division
import collections
import pickle as pkl
import PIL.Image
from utils import *
from input_utils import *
from models import *

COLORS = pkl.load(open("pallete", "rb"))  # load predefined 100 different colors to be used when drawing boxes and text


def threshold_predictions(boxes, obj_probs, class_probs, threshold=0.6):
    if len(boxes.shape) > 2:
        assert boxes.shape[0] == 1, 'This function only supports single prediction'
        boxes.squeeze_(0)
        class_probs.squeeze_(0)
    obj_probs = obj_probs.view(class_probs.shape[0], 1)

    box_scores = class_probs * obj_probs   # (845, 20) for tiny yolo voc
    box_class_scores, box_classes = box_scores.max(1)  # (845), (845) for tiny yolo voc

    filtering_mask = box_class_scores >= threshold  # (845) for tiny yolo voc

    boxes = boxes[filtering_mask]
    scores = box_class_scores[filtering_mask]
    classes = box_classes[filtering_mask]

    return to_np(boxes), to_np(scores), to_np(classes)
def iou_np(box, others):
    """
    returns ious between the box and every other boxes
    """
    box, others = box.reshape(-1, 4), others.reshape(-1, 4)
    box = box.repeat(repeats=others.shape[0], axis=0)
    x1y1 = np.maximum(box[:, :2], others[:, :2])
    x2y2 = np.minimum(box[:, 2:], others[:, 2:])
    inter_edges = np.maximum(x2y2 - x1y1, 0)
    inter_areas = inter_edges[:, 0] * inter_edges[:, 1]

    box_edges = box[:, 2:4] - box[:, :2]
    box_areas = box_edges[:, 0] * box_edges[:, 1]

    others_edges = others[:, 2:4] - others[:, :2]
    others_areas = others_edges[:, 0] * others_edges[:, 1]

    union_areas = box_areas + others_areas - inter_areas

    return inter_areas / union_areas
def iou_pt(box, others):
    """
    returns ious between the box and every other boxes
    """
    box, others = box.view(-1, 4), others.view(-1, 4)
    box = box.repeat(others.shape[0], 1)
    x1y1 = torch.max(box[:, :2], others[:, :2])
    x2y2 = torch.min(box[:, 2:], others[:, 2:])
    inter_edges = torch.clamp(x2y2 - x1y1, min=0)
    inter_areas = inter_edges[:, 0] * inter_edges[:, 1]

    box_edges = box[:, 2:4] - box[:, :2]
    box_areas = box_edges[:, 0] * box_edges[:, 1]

    others_edges = others[:, 2:4] - others[:, :2]
    others_areas = others_edges[:, 0] * others_edges[:, 1]

    union_areas = box_areas + others_areas - inter_areas

    return inter_areas / union_areas
def nms(boxes, scores, classes, nms_threshold, cats):
    detected_classes = np.unique(classes)
    detections = collections.defaultdict(lambda: [])
    for cls in detected_classes:
        class_mask = classes == cls
        cls_boxes = boxes[class_mask]
        cls_scores = scores[class_mask]

        sort_idxs = np.argsort(-cls_scores, 0)  # sort in descending order
        cls_boxes = cls_boxes[sort_idxs]
        cls_scores = cls_scores[sort_idxs]
        while 1 < cls_boxes.shape[0]:
            detections[cats[cls]].append((cls_boxes[0], cls_scores[0]))
            ious = iou_np(cls_boxes[0], cls_boxes[1:])
            iou_mask = ious < nms_threshold
            cls_boxes = cls_boxes[1:][iou_mask]
            cls_scores = cls_scores[1:][iou_mask]
        # add last bounding box to the detections dictionary if there is one
        if cls_boxes.shape[0] == 1:
            detections[cats[cls]].append((cls_boxes[0], cls_scores[0]))

    return detections
def filter_predictions(pred_boxes, pred_obj_probs, pred_class_probs, cats, threshold=0.5, nms_threshold=0.5):
    # print("\033[H\033[J") #this will clear screen
    boxes, scores, classes = threshold_predictions(pred_boxes, pred_obj_probs, pred_class_probs, threshold)
    # It's easier to work with corner format when we calculate the IOU therefore we transform
    # (center_x, center_y, height, width) format to (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    boxes = centerwh_to_corners(boxes)
    # At this point, boxes are of shape (n_boxes, 4); scores and classes are of shape (n_boxes)
    detections = nms(boxes, scores, classes, nms_threshold, cats)

    return detections  # (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
def draw_predictions(im, detections):
    """
    :param im: numpy array of format (width, height, 3)
    :param detections: detections dictionary, detections themselves should be
                       in corner format, that is (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    :return: image with predictions drawn on it.
    """
    for cat in detections.keys():
        for (box, score) in detections[cat]:
            box = box.astype(np.int32)
            im = cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 1)
            im = cv2.rectangle(im, (box[0] + 1, box[1] + 1), (box[2] - 1, box[3] - 1), (0, 0, 0), 1)

            c1 = tuple(box[:2])
            fontface = cv2.FONT_HERSHEY_DUPLEX
            fontScale = 0.4
            thickness = 1
            textRectColor = [45, 116, 229]  # BGR
            textColor = [255, 255, 255]
            t_size = cv2.getTextSize(cat, fontface, fontScale, thickness)[0]
            c2 = c1[0] + t_size[0] + 2, c1[1] + t_size[1] + 5
            cv2.rectangle(im, (c1[0], c1[1]), (c2[0] + 2, c2[1] + 2), textRectColor, -1)
            cv2.putText(im, cat, (int(box[0]), int(box[1]) + t_size[1] + 4), fontface, fontScale, textColor, thickness)
    return im
def draw_predictions_nonfiltered(im, preds, cats, ground_truth=False):
    if isinstance(preds, torch.Tensor):
        preds = to_np(preds).astype(np.int32)
    im = np.ascontiguousarray(im, np.float32)
    for pred in preds:
        box = pred[:4]
        cls = pred[4]
        box = centerwh_to_corners(box).astype(np.int32).ravel()
        if ground_truth:
            cat = cats[cls]
        else:
            cat_idx = preds.ravel()[:4].argmax()
            cat = cats[cat_idx]

        im = cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 1)
        im = cv2.rectangle(im, (box[0] + 1, box[1] + 1), (box[2] - 1, box[3] - 1), (0, 0, 0), 1)

        c1 = tuple(box[:2])
        fontface = cv2.FONT_HERSHEY_DUPLEX
        fontScale = 0.4
        thickness = 1
        textRectColor = [45, 116, 229]  # BGR
        textColor = [0, 0, 0]
        t_size = cv2.getTextSize(cat, fontface, fontScale, thickness)[0]
        c2 = c1[0] + t_size[0] + 2, c1[1] + t_size[1] + 5

        cv2.rectangle(im, (c1[0], c1[1]), (c2[0] + 2, c2[1] + 2), textRectColor, -1)
        cv2.putText(im, cat, (int(box[0]), int(box[1]) + t_size[1] + 4), fontface, fontScale, textColor, thickness)

    return im
def run_on_camera(model, input_size, cats, device, threshold=0.5, nms_threshold=0.5):
    model.eval()
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600, 600)
    while True:
        ret, frame = cap.read()  # capture frame
        if ret:
            inp, orig_img = cv2cam_to_input(frame, input_size, device)
            boxes, obj_probs, class_probs = model.full_forward(inp)
            detections = filter_predictions(boxes, obj_probs, class_probs,cats, threshold, nms_threshold)
            img = draw_predictions(orig_img, detections)

            cv2.imshow('image', img)  # display the resulting frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print('Could not get the frame from camera for some reason.')
            break

    close_camera(cap)
def test_single_image(model, cats, image_path, input_size, device, figsize=(8, 8), threshold=0.5, nms_threshold=0.5):
    test_input = path_to_input(image_path, input_size, device)
    boxes, obj_probs, class_probs = model.full_forward(test_input)
    detections = filter_predictions(boxes, obj_probs, class_probs, cats, threshold, nms_threshold)
    img = PIL.Image.open(image_path)
    img = img.resize((input_size, input_size))
    img = np.array(img)


    im = draw_predictions(img, detections)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    ax.imshow(im)
