from efficientdet.dataset import CocoDataset, Augmenter, Resizer, Normalizer, collater
import torchvision.transforms as transforms
from pycocotools.cocoeval import COCOeval
import json
import torch
import numpy as np
import time
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess
COCO_CLASSES = ['wheat']


def predict(dataset, model, image_size, iter, regressBoxes, clipBoxes,
              score_threshold, iou_threshold, pred_version='v1'):
    if pred_version == 'v1':
        image_info = dataset.coco.loadImgs(iter)[0]
        image_path = '/home/data/detection/wheat/images/val2020/' + image_info['file_name']

        ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=image_size)
        x = torch.from_numpy(framed_imgs[0])
        # x = x.cuda()

        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        # run network
        regression, classification, anchors = model(x)

        preds = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            score_threshold, iou_threshold)

        preds = invert_affine(framed_metas, preds)[0]
        scores = preds['scores']
        labels = preds['class_ids']
        boxes = preds['rois']
    else:
        # 图像前处理用pytorch进行处理
        data = dataset[iter]
        scale = data['scale']

        data['img'] = data['img'].permute(2, 0, 1).float().cuda().unsqueeze(dim=0)

        regression, classification, anchors = model(data['img'])

        preds = postprocess(data['img'],
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            score_threshold, iou_threshold)[0]

        scores = preds['scores']
        labels = preds['class_ids']
        boxes = preds['rois']

        # correct boxes for image scale
        boxes /= scale

    return scores, labels, boxes

def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(dataset, retinanet, image_size, score_threshold=0.05, max_detections=100, save_path=None, use_gpu=True):
    """ Get the detections from the retinanet using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the retinanet.
        retinanet           : The retinanet to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(dataset.num_classes())] for j in range(len(dataset))]

    retinanet.eval()
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    pred_version = 'v1'
    with torch.no_grad():
        for index, imgid in enumerate(tqdm(dataset.image_ids)):
            if pred_version == 'v1':
                iter = imgid
            else:
                iter = index
            scores, labels, boxes = predict(dataset, model, image_size, iter, regressBoxes, clipBoxes,
                                            score_threshold, 0.5, pred_version)

            # select indices which have a score above the threshold
            indices = np.where(scores > score_threshold)[0]
            if indices.shape[0] > 0:
                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]

                # select detections
                image_boxes = boxes[indices[scores_sort], :]
                image_scores = scores[scores_sort]
                image_labels = labels[indices[scores_sort]]
                image_detections = np.concatenate(
                    [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
            else:
                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = np.zeros((0, 5))

            print('{}/{}'.format(index + 1, len(dataset)), end='\r')

    return all_detections


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(len(generator))]

    for i in range(len(generator)):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        print('{}/{}'.format(i + 1, len(generator)), end='\r')

    return all_annotations


def evaluate(
        generator,
        retinanet,
        image_size=512,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=100,
        save_path=None,
        use_gpu=True
):
    """ Evaluate a given dataset using a given retinanet.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        retinanet           : The retinanet to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """

    # gather all detections and annotations

    all_detections = _get_detections(generator, retinanet, image_size, score_threshold=score_threshold,
                                     max_detections=max_detections, save_path=save_path, use_gpu=use_gpu)
    all_annotations = _get_annotations(generator)

    average_precisions = {}

    for label in range(generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(generator)):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue

                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    print('\nmAP:')
    avg_mAP = []
    for label in range(generator.num_classes()):
        label_name = COCO_CLASSES[label]
        print('{}: {:.4f}'.format(label_name, average_precisions[label][0]))
        avg_mAP.append(average_precisions[label][0])
    print('avg mAP: {:.4f}'.format(np.mean(avg_mAP)))
    return np.mean(avg_mAP), average_precisions


def evaluate_coco(dataset, model, threshold=0.05, use_gpu=False):
    model.eval()
    with torch.no_grad():
        results = []
        image_ids = []
        mean_infer_time = []
        for index in tqdm(range(len(dataset))):

            data = dataset[index]
            scale = data['scale']

            data['img'] = data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0)

            start_infer_time = time.time()
            scores, labels, boxes = model(data['img'])
            end_infer_time = time.time()
            mean_infer_time.append(end_infer_time - start_infer_time)

            boxes /= scale

            if boxes.shape[0] > 0:

                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]

                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    if score < threshold:
                        break

                    image_result = {
                        'image_id': dataset.image_ids[index],
                        'category_id': dataset.label_to_coco_label(label),
                        'score': float(score),
                        'bbox': box.tolist(),
                    }

                    results.append(image_result)

            # append image to list of processed images
            image_ids.append(dataset.image_ids[index])

            # print progress
            # print('{}/{}'.format(index, len(dataset)), end='\r')
            # print('End inferring, infer time: %.8s s' % (end_infer_time - start_infer_time))

        mean_infer_time = np.mean(mean_infer_time)
        print('Mean inferring model time: %.8s s' % (mean_infer_time))

        if not len(results):
            return

        # write output
        json.dump(results, open('{}_bbox_results.json'.format(dataset.set_name), 'w'), indent=4)

        # load results in COCO evaluation tool
        coco_true = dataset.coco
        coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(dataset.set_name))

        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file, encoding='utf-8').read())

    def __getattr__(self, item):
        return self.params.get(item, None)

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    from backbone import EfficientDetBackbone

    use_gpu = False
    image_size = 640  # 512 640 768 896

    if use_gpu:
        device = torch.device("cuda")
        print('Use gpu to prediction!!!!!!')
    else:
        device = torch.device("cpu")
        print('Use cpu to prediction!!!!!!')
    modelv2_path = 'logs/d1-0525-mul-wheat/20200525-170748/d1_0.4842_0.0516_0.4326_108.pth'

    params = Params(f'projects/wheat.yml')

    dataset_val = CocoDataset("/home/data/detection/wheat", set='val2020',
                              transform=transforms.Compose([Normalizer(), Resizer(img_size=image_size)]))

    model = EfficientDetBackbone(compound_coef=1, num_classes=dataset_val.num_classes(), input_size=image_size,
                                 ratios=eval(params.anchors_ratios),
                                 scales=eval(params.anchors_scales))

    model.load_state_dict(torch.load(modelv2_path, map_location=lambda storage, loc: storage))
    model.to(device)

    evaluate(dataset_val, model, image_size, use_gpu=use_gpu)
