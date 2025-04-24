import os
import sys
import torch
import numpy as np
import time

from PySide6.QtCore import QObject, Signal

from PIL import Image, ImageFilter, ImageDraw

from histomicstk.preprocessing.color_deconvolution import color_deconvolution

from skimage.filters import threshold_otsu, gaussian
from skimage.measure import label, regionprops

import cv2

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

  def setupParameters(self, image_names, model_type, model_path, para_dict, debug_mode):
    self.modelChanged = self.model_type != model_type

    self.image_names = image_names
    self.model_type = model_type
    self.model_path = model_path
    self.debug_mode = debug_mode

    # AutoMaskGenerator
    self.create_masked_point_grids = False

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
      tStart = time.time()
      self.SAM2Segmentor(self.model_type, self.model_path)
      tEnd = time.time()
      print("SAM2 model created in", (tEnd-tStart), "seconds", flush=True)
    except Exception as e:
      self.error.emit(str(e))

  def run(self):
    if not self.sam2 or self.modelChanged:
      self.initializeModel()

    masks_outlines = []
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

    self.finished.emit(masks_outlines)
    return

  def SAM2Segmentor(self, model_type, model_path):
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

    self.sam2 = build_sam2(self.model_cfg,
                           self.sam2_checkpoint,
                           device=self.device,
                           apply_postprocessing=False)
    
    self.predictor = SAM2ImagePredictor(self.sam2)

  def getBlueChannel(self, image):
    # ImageJ Brilliang_Blue
    MATRIX = [[0.31465548, 0.383573, 0.7433543],
              [0.66023946, 0.5271141, 0.51731443],
              [0.68196464, 0.7583024, 0.4240403]]

    stains, _, _ = color_deconvolution(image, MATRIX)
    blue_ch = 255 - stains[:, :, 0]
    # gaussian filtering
    blue_ch = gaussian(blue_ch, sigma=4, preserve_range=True).astype(np.uint8)

    # otsu thresholding
    threshold = threshold_otsu(blue_ch) * self.blue_channel_threshold_factor
    blue_ch[blue_ch <= threshold] = 0
    blue_ch[blue_ch > 0] = 1
    return blue_ch

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

  # create custom point points using each label centriod on the blue channel
  def segmentOneImageByAutoPrompts(self, name):
    image = Image.open(name)
    image = image.convert("RGB")

    enhanced_image = image.filter(ImageFilter.GaussianBlur(radius=2))

    image          = np.array(image)
    enhanced_image = np.array(enhanced_image)

    mask_blue = self.getBlueChannel(image)
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

  def getPointGrid(self, mask):
    coords = np.linspace(0, 1, self.points_per_side, endpoint=False) + 0.5 / self.points_per_side
    x, y = np.meshgrid(coords, coords)
    points = np.stack([x.ravel(), y.ravel()], axis=-1)

    pixel_points = (points * np.array([mask.shape[1], mask.shape[0]])).astype(int)
    keep = mask[pixel_points[:,1], pixel_points[:,0]] # np array indexing
    filtered_points = points[keep.astype(bool)] # make sure keep is of boolean type

    return [filtered_points]

  def getPointGridCropUniform(self, mask):
    """
    Generate layer-specific uniform point grids, filtered by a binary mask.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).

    Returns:
        list: List of np.ndarray, each containing normalized points (0 to 1) for a layer.
    """
    H, W = mask.shape
    point_grids = []

    for layer in range(self.crop_n_layers + 1):
      # Adjust points_per_side based on layer scale (denser for higher layers)
      scale = self.crop_scale_factor ** layer
      layer_points_per_side = self.points_per_side * scale  # Increase density for smaller crops

      # Generate uniform grid for the full image
      coords = np.linspace(0, 1, layer_points_per_side, endpoint=False) + 0.5 / layer_points_per_side
      x, y = np.meshgrid(coords, coords)
      points = np.stack([x.ravel(), y.ravel()], axis=-1)

      # Filter points using the mask
      pixel_points = (points * np.array([W, H])).astype(int)
      keep = mask[pixel_points[:, 1], pixel_points[:, 0]]
      filtered_points = points[keep.astype(bool)]

      # Add to point_grids, ensuring non-empty arrays
      point_grids.append(filtered_points if filtered_points.size > 0 else np.array([]))

    return point_grids

  def segmentOneImageAuto(self, name):
    np.random.seed(3)

    image = Image.open(name)
    image = image.convert("RGB")

    enhanced_image = image.filter(ImageFilter.GaussianBlur(radius=2))

    image          = np.array(image)
    enhanced_image = np.array(enhanced_image)

    mask_blue = self.getBlueChannel(image)

    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    
    if self.create_masked_point_grids:
      point_grids = self.getPointGridCropUniform(mask_blue)

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
        crop_n_layers=self.crop_n_layers,
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

    if self.debug_mode:
      print("Masked point grid:", self.create_masked_point_grids)
      print("Number of raw masks", len(masks))

    masks = self.filterAnnsByMask(masks, mask_blue)
    masks = sorted(masks, key=(lambda x: x['area']), reverse=True)

    if self.debug_mode:
      print("Number of masks after filtering by blue channel mask", len(masks))
      self.savePixelMaskOverlay(name, masks)

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
      print("Final number of masks", len(final_masks))

    outlines = self.annotations2Outlines(final_masks)
    return final_masks, outlines

  def filterAnnsByMask(self, anns, mask):
    filtered = []

    for ann in anns:
      m = ann['segmentation']
      if ann['area'] > self.largest_label_size*self.largest_label_size:
        continue

      m[mask == 0] = False
      if np.sum(m.astype(np.uint8)) >= self.min_label_size:
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
    bn = os.path.splitext(os.path.basename(name))[0]
    image.save(os.path.join(dirname, bn+'_prompts.png'))

  def savePixelMaskOverlay(self, name, anns):
    image = Image.open(name)
    image = image.convert("RGB")

    dirname = os.path.dirname(name)
    bn = os.path.splitext(os.path.basename(name))[0]
    outName = os.path.join(dirname, bn+'_masks.png')

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
