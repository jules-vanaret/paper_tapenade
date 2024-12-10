import tifffile
import numpy as np
import napari
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops
from skimage.segmentation import clear_border
import matplotlib.pyplot as plt
from pathlib import Path

def build_iou_matrix(gt_segmentation, pred_segmentation):
    """
    Input arrays can be 3D or 2D.
    """

    ndim = gt_segmentation.ndim

    gt_props = regionprops(gt_segmentation)
    pred_props = regionprops(pred_segmentation)

    iou_matrix = np.zeros((len(gt_props), len(pred_props)))

    for i_gt, gt_prop in enumerate(gt_props):

        gt_bb = gt_prop.bbox
        a = np.array(gt_bb[:ndim])
        b = np.array(gt_bb[ndim:])

        for i_pred, pred_prop in enumerate(pred_props):

            pred_bb = pred_prop.bbox
            c = np.array(pred_bb[:ndim])
            d = np.array(pred_bb[ndim:])

            if np.dot(a - d, b - c) < 0:  # if the bounding boxes intersect
                roi_slices = tuple(
                    slice(min(a[i], c[i]), max(b[i], d[i])) for i in range(ndim)
                )

                gt_roi_bool = gt_segmentation[roi_slices] == gt_prop.label
                pred_roi_bool = pred_segmentation[roi_slices] == pred_prop.label

                intersection = np.count_nonzero(gt_roi_bool * pred_roi_bool)

                if intersection > 0:
                    union = np.count_nonzero(gt_roi_bool + pred_roi_bool)

                    iou = intersection / union

                    iou_matrix[i_gt, i_pred] = iou

    return iou_matrix


def relabel_segmentation(segmentation):
    labels, label_counts = np.unique(segmentation, return_counts=True)
    temp = np.zeros(segmentation.shape).astype("uint16")
    for i, label in enumerate(labels):
        temp[segmentation == label] = i
    return temp


def filter_small_objects(segmentation, threshold=50):
    """
    Remove objects below the specified area threshold.
    """
    for prop in regionprops(segmentation):
        if prop.area < threshold:
            segmentation[segmentation == prop.label] = 0
    return segmentation



def visualize_tp(array, listTP):
    """
    Create a new array with only the true positives.
    """
    array_TP = np.zeros_like(array)
    for label in listTP:
        array_TP[array == label] = label
    return array_TP

def prepare_segmentation(segmentation, thresh_volume=50,remove_border_cells=True):
    """
    Prepare a segmentation for the iou matrix computation.
    Remove small segmented cells and the one touching the borders, and relabeling the segmentation for the IoU matric to be simpler
    """
    segmentation = filter_small_objects(segmentation, threshold=thresh_volume)
    if remove_border_cells == True:
        segmentation = clear_border(segmentation)
    segmentation = relabel_segmentation(segmentation)
    return segmentation

def compute_tp_fp_fn(
    annotation,
    prediction,
    thresh_IoU:float=0.5,
    visualize_napari:bool=False,
    visualize_false_detections:bool=False,
    remove_border_cells:bool=False,
    area_thresh: int = 0,
):
    FN = []
    FP = []
    annotation = prepare_segmentation(annotation, thresh_volume=area_thresh, remove_border_cells=remove_border_cells)
    prediction = prepare_segmentation(prediction, thresh_volume=area_thresh, remove_border_cells=remove_border_cells)
    iou_matrix = build_iou_matrix(annotation, prediction)
    TP = np.array(linear_sum_assignment(iou_matrix, maximize=True)).T
    listTP = np.array([
        [tp_gt_inds + 1, tp_pred_inds + 1, iou_matrix[tp_gt_inds, tp_pred_inds]]
        for tp_gt_inds, tp_pred_inds in TP if iou_matrix[tp_gt_inds, tp_pred_inds] > thresh_IoU
    ]).T
    array_FN = np.zeros_like(annotation)
    array_FP = np.zeros_like(prediction)
    if listTP.size != 0:
        FN, FP,array_FN,array_FP = _find_false_detections(
                annotation, prediction, listTP, visualize_false_detections, array_FN, array_FP
            )
        nb_tp = len(listTP[0])
        nb_fp = len(FP)
        nb_fn = len(FN)
    else:
        nb_tp, nb_fp, nb_fn = [0] * 3
    if visualize_napari == True:
        array_tp_ann = visualize_tp(annotation, listTP[0])
        array_tp_pred = visualize_tp(prediction, listTP[1])
        viewer = napari.Viewer()
        viewer.add_labels(annotation, name="annotation")
        viewer.add_labels(prediction, name='prediction')
        viewer.add_labels(array_tp_ann, name='TP_annotation')
        viewer.add_labels(array_tp_pred, name='TP_prediction')
        if visualize_false_detections == True:
            viewer.add_labels(array_FN, name='FN')
            viewer.add_labels(array_FP, name='FP')
        napari.run()
    return (
        nb_tp,
        nb_fp,
        nb_fn,
    )
def _find_false_detections(annotation, prediction, listTP, visualize, array_FN, array_FP):
    FN, FP = [], []
    for i in np.unique(annotation):
        if i > 0 and i not in listTP[0]:
            FN.append(i)
            if visualize:
                array_FN[annotation == i] = i
    for i in np.unique(prediction):
        if i > 0 and i not in listTP[1]:
            FP.append(i)
            if visualize:
                array_FP[prediction == i] = i
    return FN, FP,array_FN,array_FP

def compute_f1_score(liste):
    [tp,fp,fn]=liste
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

def compute_metrics_over_planes(annotation,prediction,visualize_napari,thresh_volume,thresh_IoU,remove_border_cells):
    sampling = np.arange(5, annotation.shape[0] - 5, 20) #planes every 20 um starting from 5 um and ending at 5 um from the end
    total_tp, total_fp, total_fn = [0, 0, 0]
    for plan in sampling:
        (nb_tp, nb_fp, nb_fn) = compute_tp_fp_fn(
            annotation=annotation[plan],
            prediction=prediction[plan],
            thresh_IoU=thresh_IoU,
            visualize_napari=visualize_napari,
            area_thresh=thresh_volume,
            remove_border_cells=remove_border_cells,
        )
        total_tp += nb_tp
        total_fp += nb_fp
        total_fn += nb_fn
    return total_tp,total_fp,total_fn

folder = Path(__file__).parents[3] / 'data'

visualize_napari = False
remove_border_cells = True
list_samples = [1, 2, 3, 4]
thresh_volume=50
thresh_IoU = 0.5
f1_XY = []
f1_YZ = []
f1_ZX = []
f1_2D = []
f1_3D = []

for num_sample in list_samples:
    im = tifffile.imread(Path(folder) / f"S2_segmentation_performances/S2b_2D_vs_3D/image{num_sample}.tif")
    prediction_3D = tifffile.imread(Path(folder) / f"S2_segmentation_performances/S2b_2D_vs_3D/segmentation{num_sample}.tif")
    annotation_3D = tifffile.imread(Path(folder) / f"S2_segmentation_performances/S2b_2D_vs_3D/label{num_sample}.tif")

    (nb_tp_3D, nb_fp_3D, nb_fn_3D) = compute_tp_fp_fn(
        annotation_3D,
        prediction_3D,
        thresh_IoU=thresh_IoU,
        visualize_napari=visualize_napari,
        area_thresh=thresh_volume,
        remove_border_cells=remove_border_cells,
    )
    f1_3D.append(compute_f1_score([nb_tp_3D, nb_fp_3D, nb_fn_3D]))

    annotation_3D_yz = np.swapaxes(annotation_3D, 0, 2)
    annotation_3D_zx = np.swapaxes(annotation_3D, 0, 1)
    prediction_3D_yz = np.swapaxes(prediction_3D, 0, 2)
    prediction_3D_zx = np.swapaxes(prediction_3D, 0, 1)
    tp_xy,fp_xy,fn_xy=compute_metrics_over_planes(annotation_3D,prediction_3D,visualize_napari,thresh_volume,thresh_IoU,remove_border_cells) #XY
    tp_yz,fp_yz,fn_yz=compute_metrics_over_planes(annotation_3D_yz,prediction_3D_yz,visualize_napari,thresh_volume,thresh_IoU,remove_border_cells) #YZ
    tp_zx,fp_zx,fn_zx=compute_metrics_over_planes(annotation_3D_zx,prediction_3D_zx,visualize_napari,thresh_volume,thresh_IoU,remove_border_cells) #ZX
    f1_XY.append(compute_f1_score([tp_xy,fp_xy,fn_xy]))
    f1_YZ.append(compute_f1_score([tp_yz,fp_yz,fn_yz]))
    f1_ZX.append(compute_f1_score([tp_zx,fp_zx,fn_zx]))
    f1_2D.append(compute_f1_score([tp_xy + tp_yz + tp_zx, fp_xy + fp_yz + fp_zx, fn_xy + fn_yz + fn_zx]))

    
fig, ax = plt.subplots()

x_list=np.arange(10,50)
plt.bar([10,20,30,40], f1_3D, label="3D",color='darkblue')
plt.bar([11,21,31,41], f1_2D, label="2D",color='lightskyblue')
plt.bar([12,22,32,42], f1_XY,label='XY',color='lightsalmon')
plt.bar([13,23,33,43], f1_YZ,label='YZ',color='lightcoral')
plt.bar([14,24,34,44], f1_ZX,label='ZX',color='khaki')

plt.ylim(0, 1)
plt.ylabel("f1 score", fontsize=25)
plt.xlabel("sample number", fontsize=25)
plt.xticks([])
plt.yticks(fontsize=25)
plt.legend(fontsize=20)
fig.savefig(
    Path(folder) /"S2b_plot.svg"
)
plt.show()