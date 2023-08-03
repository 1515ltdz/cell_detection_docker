import numpy as np
import os
import torch

from models.experimental import attempt_load
from utils.datasets import create_dataloader_c, LoadImages_c
from utils.general import check_img_size, non_max_suppression, xyxy2xywh
from utils.torch_utils import select_device

class Model():
    """
    Parameters
    ----------
    metadata: Dict
        Dataset metadata in case you wish to compute statistics

    """
    def __init__(self, metadata):
        self.metadata = metadata

    def __call__(self, cell_patch, tissue_patch, pair_id):
        """This function detects the cells in the cell patch. Additionally
        the broader tissue context is provided. 

        NOTE: this implementation offers a dummy inference example. This must be
        updated by the participant.

        Parameters
        ----------
        cell_patch: np.ndarray[uint8]
            Cell patch with shape [1024, 1024, 3] with values from 0 - 255
        tissue_patch: np.ndarray[uint8] 
            Tissue patch with shape [1024, 1024, 3] with values from 0 - 255
        pair_id: str
            Identification number of the patch pair

        Returns
        -------
            List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
        """
        # Getting the metadata corresponding to the patch pair ID
        meta_pair = self.metadata[pair_id]

        #############################################
        #### YOUR INFERENCE ALGORITHM GOES HERE #####
        #############################################

        # The following is a dummy cell detection algorithm
        weights='weights/best.pt'
        batch_size=1
        imgsz=1024
        conf_thres=0.2
        iou_thres=0.2  # for NMS
        device='0'
        task='test'
        augment=False
        half_precision=True

        # Initialize/load model and set device
        device = select_device(device, batch_size=batch_size)

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size

        # Half
        half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
        if half:
            model.half()

        # Configure
        model.eval()

        # Dataloader
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = LoadImages_c(cell_patch, img_size=imgsz, stride=gs)

        name_list = []
        for path, img, im0s, vid_cap in dataloader:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            nb, _, height, width = img.shape  # batch size, channels, height, width

            with torch.no_grad():
                # Run model
                out, train_out = model(img, augment=augment)  # inference and training outputs

                # Run NMS
                out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, agnostic=True, multi_label=True) #加了agnostic

            # per image
            for si, pred in enumerate(out):

                # Predictions
                predn = pred.clone()

                # 增加代码
                gn = torch.tensor([1024, 1024])[[1, 0, 1, 0]]  # normalization gain whwh
                xs = []
                ys = []
                cls_id = []
                probs = []
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf)  # label format
                    p_tmp = [int(line[1]*1024), int(line[2]*1024), int(line[0])+1, line[5]]
                    if p_tmp[0] > 1023:
                        p_tmp[0] = 1023
                    if p_tmp[0] < 0:
                        p_tmp[0] = 0
                    if p_tmp[1] > 1023:
                        p_tmp[1] = 1023
                    if p_tmp[1] < 0:
                        p_tmp[1] = 0
                    xs.append(p_tmp[0])  # 添加新读取的数据
                    ys.append(p_tmp[1])
                    cls_id.append(p_tmp[2])
                    probs.append(p_tmp[3])

        # Return results
        xs = np.array(xs)
        ys = np.array(ys)
        class_id = np.array(cls_id)
        probs = np.array(probs)

        #############################################
        ####### RETURN RESULS PER SAMPLE ############
        #############################################

        # We need to return a list of tuples with 4 elements, i.e.:
        # - int: cell's x-coordinate in the cell patch
        # - int: cell's y-coordinate in the cell patch
        # - int: class id of the cell, either 1 (BC) or 2 (TC)
        # - float: confidence score of the predicted cell
        return list(zip(xs, ys, class_id, probs))
