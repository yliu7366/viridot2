import os
import sys
import torch
import torch.nn as nn
import bitsandbytes as bnb
import numpy as np
import time

from PySide6.QtCore import QObject, Signal

from PIL import Image, ImageFilter, ImageDraw

from skimage.filters import threshold_otsu, gaussian
from skimage.measure import label, regionprops

import cv2

from utils import separate_stains_nnls, scale_channel

class SAM2Worker(QObject):
  # Signals
  finished = Signal(list) # returns list of pixel masks
  progress = Signal(int)
  error = Signal(str)

  def __init__(self):
    super().__init__()
    self.sam2 = None
    self.model_type = None
    self.modelChanged = False

    self.clickToSegment = False
    self.clickX = 0
    self.clickY = 0
    self.clickToSegmentImage = None

  def setupParametersForSingleClick(self, name, x, y, model_type, model_path, para_dict, debug_mode, fp4):
    self.modelChanged = self.model_type != model_type

    self.clickToSegmentImage = name
    self.clickX = x
    self.clickY = y
    self.clickToSegment = True

    self.model_type = model_type
    self.model_path = model_path
    self.debug_mode = debug_mode
    self.fp4 = fp4

    # TODO: verify the following parameters still needed for clickToSegment
    # AutoMaskGenerator
    self.create_masked_point_grids = para_dict["custom_point_grid"]

    self.points_per_side = para_dict["points_per_side"]
    self.points_per_batch = para_dict["points_per_batch"]
    self.crop_n_layers = para_dict["crop_n_layers"]
    self.crop_scale_factor = 2
    self.circularity_threshold = para_dict["circularity_thresh"]
    self.min_label_size = para_dict["min_label_size"]
    self.pred_iou_thresh = para_dict["pred_iou_thhresh"]
    self.stability_score_thresh = para_dict["stability_score_thresh"]
    self.stability_score_offset = para_dict["stability_score_offset"]
    self.box_nms_thresh = para_dict["box_nms_thresh"]

    # other parameters
    self.blue_channel_threshold_factor = para_dict["blue_channel_thresh_factor"]
    self.largest_label_size = para_dict["largest_label_size"]

    # SAM2 model can't be created here because this function is called from the main thread
    # creating SAM2 model here will have context corruption problem in HPC environments

  def setupParameters(self, image_names, model_type, model_path, para_dict, debug_mode, fp4):
    self.modelChanged = self.model_type != model_type

    self.image_names = image_names
    self.model_type = model_type
    self.model_path = model_path
    self.debug_mode = debug_mode
    self.fp4 = fp4

    # AutoMaskGenerator
    self.create_masked_point_grids = para_dict["custom_point_grid"]

    self.points_per_side = para_dict["points_per_side"]
    self.points_per_batch = para_dict["points_per_batch"]
    self.crop_n_layers = para_dict["crop_n_layers"]
    self.crop_scale_factor = 2
    self.circularity_threshold = para_dict["circularity_thresh"]
    self.min_label_size = para_dict["min_label_size"]
    self.pred_iou_thresh = para_dict["pred_iou_thhresh"]
    self.stability_score_thresh = para_dict["stability_score_thresh"]
    self.stability_score_offset = para_dict["stability_score_offset"]
    self.box_nms_thresh = para_dict["box_nms_thresh"]

    # other parameters
    self.blue_channel_threshold_factor = para_dict["blue_channel_thresh_factor"]
    self.largest_label_size = para_dict["largest_label_size"]

    # SAM2 model can't be created here because this function is called from the main thread
    # creating SAM2 model here will have context corruption problem in HPC environments

  def initializeModel(self):
    try:
      print(f"Creating SAM2 model...", flush=True)
      tStart = time.time()
      self.createSAM2Segmentor(self.model_type, self.model_path)
      tEnd = time.time()
      print(f"SAM2 model created in {tEnd-tStart:.2f} seconds", flush=True)
    except Exception as e:
      self.error.emit(str(e))

  def run(self):
    if not self.sam2 or self.modelChanged:
      self.initializeModel()

    masks_outlines = []

    if self.clickToSegment:
      print('clickToSegment started', self.clickX, self.clickY, self.clickToSegmentImage, flush=True)
      m, o = self.singleClickSegmentation(self.clickToSegmentImage, self.clickX, self.clickY)

      masks_outlines_dict = {}
      masks_outlines_dict['masks'] = m
      masks_outlines_dict['outlines'] = o
      masks_outlines.append(masks_outlines_dict) # single click result validation done in the parent class

      print('clickToSegment finished', flush=True)
    else:
      print("Starting plaque segmentation", flush=True)
      
      total = len(self.image_names)
      for i, name in enumerate(self.image_names):
        tStart = time.time()
        # m, o = self.segmentor.segmentOneImage(name)
        m, o = self.segmentOneImageAuto(name)
        tEnd = time.time()

        print(os.path.basename(name), "{:.2f}".format(tEnd-tStart), 'seconds')

        progress = int(((i+1)/total)*100)
        self.progress.emit(progress)

        masks_outlines_dict = {}
        masks_outlines_dict['masks'] = m
        masks_outlines_dict['outlines'] = o
        masks_outlines.append(masks_outlines_dict)

      print("Plaque segmentation done", flush=True)

    self.finished.emit(masks_outlines)
    return

  def createSAM2Segmentor(self, model_type, model_path):
    osType = sys.platform

    if osType == 'darwin':
      self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if self.device.type == "cuda":
      torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
      if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if model_type == 'large':
      self.sam2_checkpoint = os.path.join(model_path, 'sam2.1_hiera_large.pt')
      self.model_cfg = 'configs/sam2.1/sam2.1_hiera_l.yaml'
    elif model_type == 'baseplus':
      self.sam2_checkpoint = os.path.join(model_path, 'sam2.1_hiera_base_plus.pt')
      self.model_cfg = 'configs/sam2.1/sam2.1_hiera_b+.yaml'
    elif model_type == 'small':
      self.sam2_checkpoint = os.path.join(model_path, 'sam2.1_hiera_small.pt')
      self.model_cfg = 'configs/sam2.1/sam2.1_hiera_s.yaml'
    elif model_type == 'tiny':
      self.sam2_checkpoint = os.path.join(model_path, 'sam2.1_hiera_tiny.pt')
      self.model_cfg = 'configs/sam2.1/sam2.1_hiera_t.yaml'

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if not self.fp4:
      self.sam2 = build_sam2(self.model_cfg,
                            self.sam2_checkpoint,
                            device=self.device,
                            apply_postprocessing=False)
      self.sam2.eval()
    else:
      # convert SAM2 model to FP4
      self.sam2 = build_sam2(self.model_cfg,
                             self.sam2_checkpoint,
                             device='cpu',
                             apply_postprocessing=False)
      self.sam2.eval()

      if self.debug_mode:
        print("Quantizing model to FP4...")
      
      self.quantizeModel2FP4(self.sam2)

      if self.debug_mode:
        print(f"Moving quantized model to device: {self.device}")
      
      self.sam2 = self.sam2.to(self.device)

      if self.debug_mode:
        print("FP4-compatible SAM2 model is ready")

    self.predictor = SAM2ImagePredictor(self.sam2)

  def quantizeModel2FP4(self, module):
    compute_dtype = torch.bfloat16
    quant_type = 'nf4'

    for name, child in module.named_children():
      if isinstance(child, nn.Linear):
        quantized_layer = bnb.nn.Linear4bit(child.in_features,
                                            child.out_features,
                                            bias=child.bias is not None,
                                            compute_dtype=compute_dtype,
                                            quant_type=quant_type)
        quantized_layer.weight = bnb.nn.Params4bit(child.weight.data,
                                                   requires_grad=False,
                                                   quant_type=quant_type)
        if child.bias is not None:
          quantized_layer.bias = child.bias

        setattr(module, name, quantized_layer)
      else:
        self.quantizeModel2FP4(child)

  def getBlueChannelColorDecon(self, image):
    # ImageJ Brilliant_Blue
    MATRIX = np.array([
                      [0.31465548, 0.66023946, 0.68196464],
                      [0.383573,   0.5271141,  0.7583024],
                      [0.7433543,  0.51731443, 0.4240403]
                      ])
    
    stains = separate_stains_nnls(image, MATRIX)
    # stain channel shows white objects on black background
    blue_ch = scale_channel(stains[:,:,0])
      
    # gaussian filtering
    blue_ch = gaussian(blue_ch, sigma=4, preserve_range=True).astype(np.uint8)

    # otsu thresholding
    threshold = threshold_otsu(blue_ch) * self.blue_channel_threshold_factor
    blue_ch[blue_ch <= threshold] = 0
    blue_ch[blue_ch > 0] = 1
    return blue_ch

  def getBlueChannelLab(self, image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    b_channel = lab_image[:, :, 2]
    threshold = 120 * self.blue_channel_threshold_factor
    mask = cv2.inRange(b_channel, 0, threshold)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask[mask > 0] = 1
    return mask
  
  def getBlueChannelLabAdaptive(self, image, k=2.5, min_seed_pixels=1000):
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # --- Step 1: Find a high-confidence "seed" mask ---
    # Use a stricter threshold to isolate only the most definite blue pixels.
    b_channel = lab_image[:, :, 2]
    seed_threshold = 115
    seed_mask = (b_channel < seed_threshold).astype(np.uint8)
    
    # If no seed pixels are found, return a blank mask to avoid errors.
    if np.sum(seed_mask) == 0:
        return np.zeros_like(b_channel)

    seed_pixel_count = np.sum(seed_mask)
    if self.debug_mode:
      print("Native blue mask pixel size:", seed_pixel_count)

    if seed_pixel_count > min_seed_pixels:
      # --- Step 2: Analyze the color properties of the seed pixels ---
      # Calculate the mean and std dev for L, a, and b channels within the seed area.
      l_mean, l_std = cv2.meanStdDev(lab_image[:, :, 0], mask=seed_mask)
      a_mean, a_std = cv2.meanStdDev(lab_image[:, :, 1], mask=seed_mask)
      b_mean, b_std = cv2.meanStdDev(lab_image[:, :, 2], mask=seed_mask)

      # --- Step 3: Create a dynamic color range based on the stats ---
      # The range is defined as mean +/- k * standard deviations.
      lower_bound = np.array([l_mean[0][0] - k * l_std[0][0], 
                              a_mean[0][0] - k * a_std[0][0], 
                              0]) # We keep b's lower bound at 0
      upper_bound = np.array([255, # Let lightness go to max
                              255, # Let 'a' go to max
                              b_mean[0][0] + k * b_std[0][0]])
      
      # Ensure bounds are valid (0-255)
      lower_bound = np.clip(lower_bound, 0, 255)
      upper_bound = np.clip(upper_bound, 0, 255)

      # --- Step 4: Create the final mask using the adaptive range ---
      final_mask = cv2.inRange(lab_image, lower_bound, upper_bound)
    else:
      kernel = np.ones((3, 3), np.uint8)
      final_mask = cv2.dilate(seed_mask, kernel, iterations=1)

    # --- Step 5: Final Cleanup ---
    kernel = np.ones((5, 5), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

    # remove white background pixels
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    final_mask[gray_image > 200] = 0
    final_mask[final_mask > 0] = 1

    return final_mask
  
  def getBrightnessMap(self, image):
    # filter out too bright pixels
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    value_channel = hsv_image[:,:,2]

    return value_channel

  def getPointPromptsFromMask(self, mask):
    labelImg = label(mask)
    regions = regionprops(labelImg)

    points = []
    labels = []
    for region in regions:
      points.append([int(region.centroid[1]), int(region.centroid[0])])
      labels.append(1)

    return points, labels

  def masks2Outlines(self, masks):
    outlines = []
    for m in masks:
      contours, _ = cv2.findContours(m[0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      # pick the largest contour, sometimes, there are single point contours
      contours = list(contours)
      contours = sorted(contours, key=len, reverse=True)

      outlines.append(contours[0])
    return outlines

  def annotations2Outlines(self, anns):
    outlines = []

    for ann in anns:
      msk = ann['segmentation']
      contours, _ = cv2.findContours(msk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      contours = list(contours)
      contours = sorted(contours, key=len, reverse=True)
      outlines.append(contours[0])
    return outlines

  def mask2Annotation(self,
                      mask: np.ndarray, 
                      predicted_iou: float, 
                      point_prompt: np.ndarray) -> dict | None:
    """
    Constructs a SAM-style annotation dictionary from a single segmentation mask.

    Args:
        mask (np.ndarray): A 2D boolean numpy array representing the segmentation mask.
        predicted_iou (float): The IoU score predicted by the model for this mask.
        point_prompt (np.ndarray): The [x, y] coordinates of the point prompt used.

    Returns:
        dict | None: An annotation dictionary, or None if the mask is empty.
    """
    # 1. Calculate the area
    area = np.sum(mask)
    if area == 0:
        return None  # Return None for empty masks

    # 2. Calculate the bounding box [x, y, w, h]
    # Find the indices of the non-zero elements (the mask)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    bbox = [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]

    # 3. Assemble the annotation dictionary
    annotation = {
        'segmentation': mask,
        'area': int(area),
        'bbox': bbox,
        'predicted_iou': predicted_iou,
        
        # --- Add placeholder/known values for other common keys ---
        
        # The stability score is not calculated by the single-point predictor,
        # so we can use a placeholder like 0.0 or None.
        'stability_score': 0.0, 
        
        # Store the point that generated this mask
        'point_coords': [point_prompt.tolist()],
        
        # The crop box for a single prediction is the whole image
        'crop_box': [0, 0, mask.shape[1], mask.shape[0]] 
    }

    return annotation

  def singleClickSegmentation(self, name, x, y):
    image = Image.open(name)
    image = image.convert("RGB")

    self.predictor.set_image(image)

    w, h = image.size

    points = np.array([[int(x*w), int(y*h)]])
    labels = np.array([1])

    masks = []
    m, s, l = self.predictor.predict(
      point_coords=points,
      point_labels=labels,
      multimask_output=False,
    )

    # m is a (1, H, W) array, so we take the first element
    single_mask = m[0]
    # s is a (1,) array, so we take the first element
    single_iou = s[0]
    # use points[0] to get the [x, y] part
    annotation = self.mask2Annotation(single_mask, single_iou, points[0])

    masks.append(annotation)

    outlines = self.annotations2Outlines(masks)

    return masks, outlines
  
  # create custom point points using each label centriod on the blue channel
  def segmentOneImageByAutoPrompts(self, name):
    image = Image.open(name)
    image = image.convert("RGB")

    enhanced_image = image.filter(ImageFilter.GaussianBlur(radius=2))

    image          = np.array(image)
    enhanced_image = np.array(enhanced_image)

    mask_blue, _ = self.getBlueChannelColorDecon(image)
    points, labels = self.getPointPromptsFromMask(mask_blue)

    self.predictor.set_image(enhanced_image)

    masks = []
    for pt, lbl in zip(points, labels):
      m, s, l = self.predictor.predict(
        point_coords=np.array([pt]),
        point_labels=np.array([lbl]),
        multimask_output=False,
      )
      masks.append(m)

    outlines = self.masks2Outlines(masks)

    return masks, outlines

  def getPointGrid(self, mask, factor=1):
    pps = int(self.points_per_side * factor)
    if pps > 152:
      pps = 152

    coords = np.linspace(0, 1, pps, endpoint=False) + 0.5 / pps
    x, y = np.meshgrid(coords, coords)
    points = np.stack([x.ravel(), y.ravel()], axis=-1)

    pixel_points = (points * np.array([mask.shape[1], mask.shape[0]])).astype(int)

    dilation_kernel_size = 7
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

    keep = dilated_mask[pixel_points[:,1], pixel_points[:,0]] # np array indexing
    filtered_points = points[keep.astype(bool)] # make sure keep is of boolean type

    if filtered_points.size > 0:
      return [filtered_points]
    else:
      return []
  
  def segmentOneImageAuto(self, name):
    np.random.seed(3)

    image = Image.open(name)
    image = image.convert("RGB")

    enhanced_image = image.filter(ImageFilter.GaussianBlur(radius=2))

    image          = np.array(image)
    enhanced_image = np.array(enhanced_image)

    #mask_blue = self.getBlueChannelColorDecon(image)
    #mask_blue = self.getBlueChannelLab(image)
    mask_blue = self.getBlueChannelLabAdaptive(image, self.blue_channel_threshold_factor)

    if self.debug_mode:
      dirname = os.path.dirname(name)
      os.makedirs(os.path.join(dirname, 'debug'), exist_ok=True)
      bn = os.path.splitext(os.path.basename(name))[0]
      outName = os.path.join(dirname, 'debug', bn+'_blue_mask.png')
      outImg = Image.fromarray(mask_blue*255, mode='L')
      outImg.save(outName)

    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    if self.create_masked_point_grids:
      
      point_grids = self.getPointGrid(mask_blue, 2)

      if not point_grids:# empty point grid
        return [], []

      if self.debug_mode:
        self.savePointGridOverlay(name, point_grids)

      maskGenerator = SAM2AutomaticMaskGenerator(
        model=self.sam2,
        points_per_side=None,
        points_per_batch=self.points_per_batch,
        point_grids=point_grids,
        pred_iou_thresh=self.pred_iou_thresh,
        stability_score_thresh=self.stability_score_thresh,
        stability_score_offset=self.stability_score_offset,
        crop_n_layers=0,
        crop_n_points_downscale_factor=self.crop_scale_factor,
        box_nms_thresh=self.box_nms_thresh,
        min_mask_region_area=self.min_label_size*self.min_label_size,
        use_m2m=True,
      )
    else:
      maskGenerator = SAM2AutomaticMaskGenerator(
        model=self.sam2,
        points_per_side=self.points_per_side,
        points_per_batch=self.points_per_batch,
        pred_iou_thresh=self.pred_iou_thresh,
        stability_score_thresh=self.stability_score_thresh,
        stability_score_offset=self.stability_score_offset,
        crop_n_layers=self.crop_n_layers,
        crop_n_points_downscale_factor=self.crop_scale_factor,
        box_nms_thresh=self.box_nms_thresh,
        min_mask_region_area=self.min_label_size*self.min_label_size,
        use_m2m=True,
      )

    masks = maskGenerator.generate(image)
    masks = self.filterLargeAnns(masks)

    if self.debug_mode:
      print("Masked point grid:", self.create_masked_point_grids)
      print("Number of raw masks", len(masks))
      self.savePixelMaskOverlay(name, masks, 'masks_raw')

    masks = self.filterAnnsByMask(masks, mask_blue)
    masks = sorted(masks, key=(lambda x: x['area']), reverse=True)

    if self.debug_mode:
      print("Number of masks after filtering by blue channel mask", len(masks))
      self.savePixelMaskOverlay(name, masks, 'masks_filtered')

    iouMatrix = self.computeIouMatrix(masks)
    overlapped = self.findMultipleLabelsOnLargeCommonBackground(iouMatrix, 0.01, 1)
    filtered_masks = self.removeOverlappedLabels(masks, overlapped)
    c_scores = self.calculateCircularity(filtered_masks)

    if self.debug_mode:
      print("Number of masks after removing overlapped masks", len(filtered_masks))

    final_masks = []
    for m, c in zip(filtered_masks, c_scores):
      if c[1] >= self.circularity_threshold:
        final_masks.append(m)
    
    if self.debug_mode:
      print("Number of masks after circularity filtering", len(final_masks))

    """
    brightness_mask = self.getBrightnessMap(image)
    final_masks = self.filterByBrightness(final_masks, brightness_mask)
    """

    if self.debug_mode:
      print("Final number of masks", len(final_masks))

    outlines = self.annotations2Outlines(final_masks)
    return final_masks, outlines

  def filterByBrightness(self, anns, b_map):
    filtered = []

    for ann in anns:
      m = ann['segmentation']

      foreground_pixels = b_map[m]
      avg_brightness = foreground_pixels.mean()

      if avg_brightness <= 190: # skip bright non-blue-purple labels
        filtered.append(ann)
    
    return filtered
  
  def filterLargeAnns(self, anns):
    filtered = []

    for ann in anns:
      if ann['area'] <= self.largest_label_size * self.largest_label_size:
        filtered.append(ann)

    return filtered
  
  def filterAnnsByMask(self, anns, mask):
    filtered = []

    for ann in anns:
      m = ann['segmentation']
      
      m_area_test = m.copy()
      m_area_test[mask == 0] = False

      blue_pixels = np.sum(m_area_test.astype(np.uint8))
      blue_size = 5*5

      # the segmentation will be kept if there's at least one corresponding blue mask pixel
      # and the segmentation has enough number of blue pixels
      if np.any(np.logical_and(m, mask)): 
        if blue_pixels >= blue_size:
          filtered.append(ann)

    return filtered
  
  def computeIouMatrix(self, anns):
    """
    the background mask has been removed from the list of input anns
    """
    numLabels = len(anns)
    iouMatrix = np.zeros((numLabels, numLabels), dtype=np.float32)

    for i in range(numLabels):
      for j in range(i + 1, numLabels):  # check only smaller labels to avoid redundant computing
        maski = anns[i]['segmentation']
        maskj = anns[j]['segmentation']
        intersection = np.logical_and(maski, maskj).sum()
        union = np.logical_or(maski, maskj).sum()

        if union > 0:
          iouMatrix[i, j] = intersection / union
          iouMatrix[j, i] = iouMatrix[i, j]

    return iouMatrix

  def findMultipleLabelsOnLargeCommonBackground(self, iouMatrix, iouThres=0.1, minNumLabels=2):
    """
    iouThres: IoU threshold for checking overlapping
    minNumLabels: minimal number of smaller labels on top of larger common background
    """

    numLabels = iouMatrix.shape[0]
    overlapped = []

    for i in range(numLabels):
      foregroundLabels = [j for j in range(i + 1, numLabels) if iouMatrix[i, j] > iouThres]
      if len(foregroundLabels) >= minNumLabels:
        overlapped.append((i, foregroundLabels))
    return overlapped

  def removeOverlappedLabels(self, anns, overlapped):
    to_remove = []

    for overlap in overlapped:
      background = overlap[0]
      foregrounds = overlap[1]

      """
      for m in foregrounds:
        to_remove.append(m)
      """
      # this behavior worked on SAM2 with CUDA
      if len(foregrounds) > 1:
        to_remove.append(background)
      elif len(foregrounds) == 1:
        to_remove.append(foregrounds[0])
      
    updated_anns = [ann for i, ann in enumerate(anns) if i not in to_remove]
    return updated_anns

  def calculateCircularity(self, anns):
    circularity = []

    for ann in anns:
      numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(ann['segmentation'].astype(np.uint8))

      contours, hierarchy = cv2.findContours(ann['segmentation'].astype(np.uint8), cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)
      perimeter = 0
      for contour in contours:
        perimeter += cv2.arcLength(contour, True)

      area = np.sum(ann['segmentation'].astype(np.uint8))
      c = 4 * np.pi * area / (perimeter * perimeter)
      circularity.append((numLabels, c))

    return circularity

  def savePointGridOverlay(self, name, point_grids):
    """
    a debug function to show point prompts overlaid to the original image
    :param name: path to the input image
    :param points: masked point grid
    :return: None
    """
    image = Image.open(name)
    image = image.convert("RGB")

    # match with the actual input
    image.filter(ImageFilter.GaussianBlur(radius=2))

    colors = ['red', 'yellow', 'cyan']

    for points, color in zip(point_grids, colors):
      pixel_points = (points * np.array([image.height, image.width])).astype(int)
      point_list = [tuple(pt) for pt in pixel_points]

      painter = ImageDraw.Draw(image)
      painter.point(point_list, fill=color)

    dirname = os.path.dirname(name)
    os.makedirs(os.path.join(dirname, 'debug'), exist_ok=True)
    bn = os.path.splitext(os.path.basename(name))[0]
    image.save(os.path.join(dirname, 'debug', bn+'_prompts.png'))

  def savePixelMaskOverlay(self, name, anns, suffix):
    image = Image.open(name)
    image = image.convert("RGB")

    dirname = os.path.dirname(name)
    os.makedirs(os.path.join(dirname, 'debug'), exist_ok=True)
    bn = os.path.splitext(os.path.basename(name))[0]
    outName = os.path.join(dirname, 'debug', bn+'_'+suffix+'.png')

    if not anns:
      image.save(outName)

    labelAlpha = 0.7
    image = np.array(image)

    msk = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    img = np.zeros((image.shape[0], image.shape[1], 3))
    for ann in anns:
      m = ann['segmentation']
      
      # cumulate binary mask image 
      msk[m] = 255

      rgb = np.random.random(3)
      img[m] = rgb

    img = (img*255).astype(np.uint8)
    # composite
    blended = cv2.addWeighted(image, 1-labelAlpha, img, labelAlpha, 0.0)
    results = np.where(msk == 0, image, blended)
    results = cv2.cvtColor(results, cv2.COLOR_RGB2BGR)# prepare for opencv imwrite

    cv2.imwrite(outName, results)
