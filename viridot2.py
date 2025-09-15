"""
**dependencies:
conda install pyside6
pip install natsort scikit-image
pip install histomicstk --find-links https://girder.github.io/large_image_wheels --use-pep517 --no-cache-dir
install sam2 & pytorch
"""
import sys
import glob
import os
import pickle
from natsort import natsorted
import time

print('Viridot2 starting...')
tStart = time.time()

if sys.version_info.major == 3 and sys.version_info.minor < 11:
  from typing_extensions import Self

from PySide6.QtWidgets import (QApplication, QWidget, QPushButton, QFileDialog, QCheckBox,
                                QComboBox, QLabel, QSpinBox, QDoubleSpinBox, QMessageBox,
                                QVBoxLayout, QHBoxLayout, QGroupBox, QListWidget)
from PySide6.QtCore import QSize, QSettings, QThread, Slot, QObject, Signal

from ImageViewerWidget import ImageViewerWidget
from SettingsDialog import SettingsDialog
from SAM2Segmentor import SAM2Worker
from BatchProcessorDialog import BatchProcessorDialog

from utils import savePlaqueCounts, saveSegmentation, findAllImages

print(f"Dependencies loaded in {time.time()-tStart:.2f}s")

class ResultsEmitter(QObject):
  results_ready = Signal(list)

class DatasetEmitter(QObject):
  dataset_ready = Signal(dict)

class MainGUI(QWidget):
  """Main application window"""
  # signal to run SAM2 segmentation
  start_computation = Signal()

  def __init__(self):
    super().__init__()

    self.sam2_model_labels = ['SAM2 Tiny', 'SAM2 Small', 'SAM2 Base Plus', 'SAM2 Large']
    self.sam2_model_keys = ['tiny', 'small', 'baseplus', 'large']
    self.model_index = 2 # default model, could be moved to settings

    # batch processing
    self.datasets = [] # a list of datasets for batch processing
    self.current_dataset_index = -1
    self.current_image_list = []

    # single plate processing
    self.image_list = []

    self.result_emitter = ResultsEmitter()
    self.dataset_emitter = DatasetEmitter()

    self.controlPanelWidgetWidth = 200
    self.labelPanelWidgetWidth   = 200

    self.thread = None
    self.worker = None
    self.is_worker_running = False

    self.initUI()
    self.initWorker()

  def initWorker(self):
    # should only be called once

    self.thread = QThread()
    self.worker = SAM2Worker()

    self.worker.moveToThread(self.thread)

    # connect signals
    self.start_computation.connect(self.worker.run)
    self.worker.finished.connect(self.computationDone)
    self.worker.progress.connect(self.updateProgress)
    self.worker.error.connect(self.handleError)

    self.thread.start()

  def initUI(self):
    self.setWindowTitle("Viridot2")
    self.setGeometry(200, 200, 1200, 750)

    self.layout = QHBoxLayout()

    self.controlPanelLayout = QVBoxLayout()
    self.viewerPanelLayout = QVBoxLayout()
    self.labelPanelLayout = QVBoxLayout()

    # Image View Widget Panel (Center)
    self.image_view = ImageViewerWidget()
    self.result_emitter.results_ready.connect(self.image_view.updateMasksOutlines)
    self.dataset_emitter.dataset_ready.connect(self.image_view.updateDataset)
    self.image_view.currentSeg.connect(self.populateSegmentationList)

    self.viewerPanelLayout.addWidget(self.image_view)

    # Control Panel (Left)
    controlPanelButtonSize = QSize(self.controlPanelWidgetWidth, 24)
    self.settingsButton = QPushButton("Settings", self)
    self.settingsButton.setFixedSize(controlPanelButtonSize)
    self.settingsButton.clicked.connect(self.showSettingsDlg)

    self.createParameterWidgets(controlPanelButtonSize)

    ## process single plate
    self.singlePlateGroup = QGroupBox("Process Single Plate")
    self.singlePlateGroup.setFixedWidth(self.controlPanelWidgetWidth)
    self.singleGroupLayout = QVBoxLayout()

    self.loadDatasetButton = QPushButton("Load Dataset", self)
    self.loadDatasetButton.clicked.connect(self.loadDataset)

    self.loadSegmentationButton = QPushButton("Load Segmentation", self)
    self.loadSegmentationButton.clicked.connect(self.loadSegmentation)
    self.loadSegmentationButton.setEnabled(False) # disabled if no dataset loaded

    self.goButton = QPushButton("GO!", self)
    self.goButton.clicked.connect(self.onGoButtonClicked)
    self.goButton.setEnabled(False) # disabled if no dataset loaded

    self.singleGroupLayout.addWidget(self.loadDatasetButton)
    self.singleGroupLayout.addWidget(self.loadSegmentationButton)
    self.singleGroupLayout.addWidget(self.goButton)
    self.singlePlateGroup.setLayout(self.singleGroupLayout)

    ## batch processing
    self.multiPlateGroup = QGroupBox("Process Multiple Plates")
    self.multiPlateGroup.setFixedWidth(self.controlPanelWidgetWidth)
    self.multiGroupLayout = QVBoxLayout()

    self.batchButton = QPushButton("Launch Batch Processor", self)
    self.batchButton.clicked.connect(self.launchBatchProcessor)
    self.multiGroupLayout.addWidget(self.batchButton)
    self.multiPlateGroup.setLayout(self.multiGroupLayout)

    self.progressLabel = QLabel("0%")
    self.progressLabel.setEnabled(False)    

    self.controlPanelLayout.addWidget(self.settingsButton)    
    self.controlPanelLayout.addWidget(self.singlePlateGroup)
    self.controlPanelLayout.addWidget(self.multiPlateGroup)
    self.addParameterWidgets()
    self.controlPanelLayout.addWidget(self.progressLabel)

    # Label Widget Panel (right)
    self.labelListLabel = QLabel("Segmentations")

    self.labelListWidget = QListWidget()
    self.labelListWidget.setFixedWidth(self.labelPanelWidgetWidth)

    labelPanelButtonSize = QSize(self.labelPanelWidgetWidth, 32)
    self.testButton = QPushButton("Under construction")
    self.testButton.setFixedSize(labelPanelButtonSize)
    
    self.labelPanelLayout.addWidget(self.labelListLabel)
    self.labelPanelLayout.addWidget(self.labelListWidget)
    self.labelPanelLayout.addWidget(self.testButton)

    self.layout.addLayout(self.controlPanelLayout)
    self.layout.addLayout(self.viewerPanelLayout)
    self.layout.addLayout(self.labelPanelLayout)

    self.setLayout(self.layout)

  def addParameterWidgets(self):
    self.controlPanelLayout.addWidget(self.modelSelector)

    r1 = QHBoxLayout()
    r1Label = QLabel("Grid Size:") # Points per Side:
    r1.addWidget(r1Label)
    r1.addWidget(self.pointsPerSide)

    r2 = QHBoxLayout()
    r2Label = QLabel("Batch Size:") # Points per Batch:
    r2.addWidget(r2Label)
    r2.addWidget(self.pointsPerBatch)

    r3 = QHBoxLayout()
    r3Label = QLabel("IoU Threshold:")
    r3.addWidget(r3Label)
    r3.addWidget(self.predIOUThresh)

    r4 = QHBoxLayout()
    r4Label = QLabel("Stability Thresh.:")
    r4.addWidget(r4Label)
    r4.addWidget(self.stabilityScoreThresh)

    r5 = QHBoxLayout()
    r5Label = QLabel("Stability Offset:")
    r5.addWidget(r5Label)
    r5.addWidget(self.stabilityScoreOffset)

    r6 = QHBoxLayout()
    r6Label = QLabel("Min. Label Size:")
    r6.addWidget(r6Label)
    r6.addWidget(self.minLabelSize)

    r7 = QHBoxLayout()
    r7Label = QLabel("Max. Label Size:")
    r7.addWidget(r7Label)
    r7.addWidget(self.largestLabelSize)

    r8 = QHBoxLayout()
    r8Label = QLabel("Blue Factor:")
    r8.addWidget(r8Label)
    r8.addWidget(self.blueChannelThreshFactor)

    r9 = QHBoxLayout()
    r9Label = QLabel("Circularity Thresh.:")
    r9.addWidget(r9Label)
    r9.addWidget(self.circularityThresh)

    r10 = QHBoxLayout()
    r10Label = QLabel("Crop N Layers:")
    r10.addWidget(r10Label)
    r10.addWidget(self.cropNLayers)

    r11 = QHBoxLayout()
    r11Label = QLabel("Box NMS Thresh.:")
    r11.addWidget(r11Label)
    r11.addWidget(self.boxNMSThresh)

    r12 = QHBoxLayout()
    r12Label = QLabel("Use Custom Point Grid")
    r12.addWidget(r12Label)
    r12.addWidget(self.customPointGrid)

    commonGroupBox = QGroupBox("Common Parameters")
    commonGroupBox.setFixedWidth(self.controlPanelWidgetWidth)
    commonGroupLayout = QVBoxLayout()

    commonGroupLayout.addLayout(r1)
    commonGroupLayout.addLayout(r2)
    commonGroupLayout.addLayout(r9)
    commonGroupLayout.addLayout(r6)
    commonGroupLayout.addLayout(r7)
    commonGroupLayout.addLayout(r8)
    commonGroupLayout.addLayout(r12)

    commonGroupBox.setLayout(commonGroupLayout)

    uncommonGroupBox = QGroupBox("Uncommon Parameters")
    uncommonGroupBox.setFixedWidth(self.controlPanelWidgetWidth)
    uncommonGroupLayout = QVBoxLayout()

    uncommonGroupLayout.addLayout(r3)
    uncommonGroupLayout.addLayout(r4)
    uncommonGroupLayout.addLayout(r5)
    uncommonGroupLayout.addLayout(r10)
    uncommonGroupLayout.addLayout(r11)

    uncommonGroupBox.setLayout(uncommonGroupLayout)

    self.controlPanelLayout.addWidget(commonGroupBox)
    self.controlPanelLayout.addWidget(uncommonGroupBox)
    return
  
  def createParameterWidgets(self, buttonSize):
    """ 
    # SAM2 parameters
    SAM2 model type
    points_per_side = 38
    points_per_batch = 1024
    crop_n_layers = 1
    crop_scale_factor = 2
    min_label_size = 5*5
    pred_iou_thresh = 0.8
    stability_score_thresh = 0.92
    stability_score_offset = 0.7
    box_nms_thresh = 0.7

    # other parameters
    circularity_threshold = 0.35
    blue_channel_threshold_factor = 2.5
    largest_label_size = 150*150
    """
    self.modelSelector = QComboBox()
    self.modelSelector.setFixedSize(buttonSize)
    self.modelSelector.addItems(self.sam2_model_labels)
    self.modelSelector.setCurrentIndex(self.model_index)
    self.modelSelector.currentIndexChanged.connect(self.onModelSelectorChanged)

    self.pointsPerSide = QSpinBox()
    self.pointsPerBatch = QSpinBox()
    self.cropNLayers = QSpinBox()

    self.minLabelSize = QSpinBox()
    self.predIOUThresh = QDoubleSpinBox()
    self.stabilityScoreThresh = QDoubleSpinBox()
    self.stabilityScoreOffset = QDoubleSpinBox()
    self.boxNMSThresh = QDoubleSpinBox()
    self.circularityThresh = QDoubleSpinBox()
    self.blueChannelThreshFactor = QDoubleSpinBox()
    self.largestLabelSize = QSpinBox()
    self.customPointGrid = QCheckBox("")

    osType = sys.platform

    self.pointsPerSide.setRange(19, 152)# may need adjustment later
    if osType == 'darwin':
      self.pointsPerBatch.setRange(16, 128)
    else:
      self.pointsPerBatch.setRange(16, 2048)
    self.cropNLayers.setRange(0, 5)
    self.minLabelSize.setRange(1, 10)
    self.predIOUThresh.setRange(0.1, 1.0)
    self.stabilityScoreThresh.setRange(0.1, 1.0)
    self.stabilityScoreOffset.setRange(0.1, 1.0)
    self.boxNMSThresh.setRange(0.1, 1.0)
    self.circularityThresh.setRange(0.1, 1.0)
    self.blueChannelThreshFactor.setRange(1.0, 3.0)
    self.largestLabelSize.setRange(100, 250)
    
    self.pointsPerSide.setSingleStep(1)
    self.pointsPerBatch.setSingleStep(1)
    self.minLabelSize.setSingleStep(1)
    self.largestLabelSize.setSingleStep(5)
    self.predIOUThresh.setSingleStep(0.1)
    self.stabilityScoreThresh.setSingleStep(0.1)
    self.stabilityScoreOffset.setSingleStep(0.1)
    self.boxNMSThresh.setSingleStep(0.1)
    self.circularityThresh.setSingleStep(0.1)
    self.blueChannelThreshFactor.setSingleStep(0.1)

    self.predIOUThresh.setDecimals(2)
    self.stabilityScoreThresh.setDecimals(2)
    self.stabilityScoreOffset.setDecimals(2)
    self.boxNMSThresh.setDecimals(2)
    self.circularityThresh.setDecimals(2)
    self.blueChannelThreshFactor.setDecimals(2)

    self.pointsPerSide.setValue(48)
    if osType == 'darwin':
      self.pointsPerBatch.setValue(64)
    else:
      self.pointsPerBatch.setValue(1024)
    self.minLabelSize.setValue(3)
    self.largestLabelSize.setValue(150)
    self.predIOUThresh.setValue(0.8)
    self.stabilityScoreThresh.setValue(0.92)
    self.stabilityScoreOffset.setValue(0.7)
    self.cropNLayers.setValue(1)
    self.boxNMSThresh.setValue(0.7)
    self.circularityThresh.setValue(0.35)
    self.blueChannelThreshFactor.setValue(2.5)
    self.customPointGrid.setChecked(True)

  def getExtraParameters(self):
    paraDict = {}

    paraDict["points_per_side"] = self.pointsPerSide.value()
    paraDict["points_per_batch"] = self.pointsPerBatch.value()
    paraDict["crop_n_layers"] = self.cropNLayers.value()
    paraDict["min_label_size"] = self.minLabelSize.value()
    paraDict["pred_iou_thhresh"] = self.predIOUThresh.value()
    paraDict["stability_score_thresh"] = self.stabilityScoreThresh.value()
    paraDict["stability_score_offset"] = self.stabilityScoreOffset.value()
    paraDict["box_nms_thresh"] = self.boxNMSThresh.value()
    paraDict["circularity_thresh"] = self.circularityThresh.value()
    paraDict["blue_channel_thresh_factor"] = self.blueChannelThreshFactor.value()
    paraDict["largest_label_size"] = self.largestLabelSize.value()
    paraDict["custom_point_grid"] = self.customPointGrid.isChecked()
    
    return paraDict

  def launchBatchProcessor(self):
    dialog = BatchProcessorDialog(self)
    dialog.processing_requested.connect(self.startBatchProcessing)
    dialog.exec()

  def startBatchProcessing(self, datasets, mode, root):
    print("Starting batch processing")

    print('Root folder:', root)
    print('Processing mode:', mode)
    print('Datasets:', datasets)
    
    self.datasets = datasets

    if mode == 'local':
      self.startLocalBatchProcessing()
    elif mode == 'slurm':
      self.startSlurmBatchProcessing()

    return
  
  def setupWorker(self, image_list):
    paraDict = self.getExtraParameters()
    #print(paraDict)

    settings = QSettings("IDSS", "Viridot2")
    cpt_path = settings.value("modelpath", "./sam2_repo/checkpoints")
    debug_mode = settings.value("debugmode", False, type=bool)
    fp4 = settings.value("fp4", False, type=bool)

    self.worker.setupParameters(
      image_list,
      self.sam2_model_keys[self.model_index],
      cpt_path,
      paraDict,
      debug_mode,
      fp4
    )

    print('Dataset:', os.path.basename(os.path.dirname(image_list[0])))
    return
  
  def setupGUIforComputing(self):
    # enable/disable GUI controls for computing
    self.goButton.setEnabled(False)
    self.enableParameterWidgets(False)
    self.settingsButton.setEnabled(False)
    self.loadDatasetButton.setEnabled(False)
    self.loadSegmentationButton.setEnabled(False)
    self.batchButton.setEnabled(False)

    self.progressLabel.setEnabled(True)
    self.progressLabel.setText("0%")
    return
  
  def startLocalBatchProcessing(self):
    self.current_dataset_index = 0
    
    # self.datasets contains only valid subfolders with at least one image
    self.current_image_list = findAllImages(self.datasets[self.current_dataset_index])

    if self.is_worker_running:
      return # prevent multiple runs
    
    self.is_worker_running = True

    self.setupGUIforComputing()
    self.setupWorker(self.current_image_list)

    self.start_computation.emit()
    return
  
  def startSlurmBatchProcessing(self):
    # right now do nothing
    return
  
  def onGoButtonClicked(self):
    if not self.image_list:
      return # no data 
    
    if self.is_worker_running:
      return # prevent multiple runs
    
    self.is_worker_running = True

    self.setupGUIforComputing()
    self.setupWorker(self.image_list)

    self.start_computation.emit()
    return

  def populateSegmentationList(self, result):
    self.labelListWidget.clear()
    
    for idx, mask in enumerate(result['masks']):
      item_str = " ".join(['#'+str(idx+1),
                           "Size:",
                           str(mask['area']),
      ])
      self.labelListWidget.addItem(item_str)

  @Slot(int)
  def updateProgress(self, value):
    self.progressLabel.setText(f"{value}%")

  def setupGUIAfterComputing(self):
    # enable GUI after segmentation is done
    self.goButton.setEnabled(True)
    self.enableParameterWidgets(True)
    self.settingsButton.setEnabled(True)
    self.loadDatasetButton.setEnabled(True)
    self.loadSegmentationButton.setEnabled(True)
    self.batchButton.setEnabled(True)

  @Slot(list)
  def computationDone(self, results):
    if not self.datasets: # single plate processing
      self.progressLabel.setEnabled(False)

      saveSegmentation(self.image_list, results)
      savePlaqueCounts(self.image_list, results)
      
      self.result_emitter.results_ready.emit(results)
      
      # populate the first image segmentations to the label list widget
      self.populateSegmentationList(results[0])
      self.setupGUIAfterComputing()

      self.is_worker_running = False
    else:
      # TODO: batch processing
      total = len(self.datasets)

      # save results always
      saveSegmentation(self.current_image_list, results)
      savePlaqueCounts(self.current_image_list, results)

      # do not emit results to update ImageViewerWidget
      # do not call populateSegmentationList()
      # because this is batch processing 

      self.current_dataset_index += 1 # advance the index

      if self.current_dataset_index >= total: # all datasets processed
        # reset self.datasets
        self.datasets = []
        self.setupGUIAfterComputing()
        self.is_worker_running = False # we are done here, no more emits
      else:
        # advance to the next dataset
        self.current_image_list = findAllImages(self.datasets[self.current_dataset_index])

        self.setupWorker(self.current_image_list)
        self.start_computation.emit()

    return

  @Slot(str)
  def handleError(self, message):
    QMessageBox.critical(self, "Application Error", message)
    return

  def onModelSelectorChanged(self, index):
    self.model_index = index
    return

  def showSettingsDlg(self):
    dlg = SettingsDialog(self)
    dlg.exec()

  def loadDataset(self):
    folder = QFileDialog.getExistingDirectory(self, "Select a dataset folder")
    if not folder:
      return

    names = findAllImages(folder)
    if not names:
      return
    
    dataset = os.path.basename(os.path.dirname(names[0]))
    self.image_list = names

    data_dict = {}
    data_dict['dataset'] = dataset
    data_dict['names'] = names
    self.dataset_emitter.dataset_ready.emit(data_dict)

    # enable GUI when a valid dataset is loaded
    self.loadSegmentationButton.setEnabled(True)
    self.goButton.setEnabled(True)

    self.labelListWidget.clear()

  def loadSegmentation(self):
    fileName, _ = QFileDialog.getOpenFileName(self, "Open Segmentation", "", "Pickle Files (*.pkl)")
    if not fileName:
      return
    
    with open(fileName, 'rb') as file:
      results = pickle.load(file)

    moList = []
    for seg in results:
      mo = {}
      mo['masks'] = seg['masks']
      mo['outlines'] = seg['outlines']
      moList.append(mo)

    self.result_emitter.results_ready.emit(moList)
    self.populateSegmentationList(moList[0])

    QMessageBox.information(self, "Success", f"Successfully loaded {len(moList)} segmentation results from\n{fileName}")
    
    return
  
  def enableParameterWidgets(self, enable):
    self.modelSelector.setEnabled(enable)
    self.pointsPerSide.setEnabled(enable)
    self.pointsPerBatch.setEnabled(enable)
    self.cropNLayers.setEnabled(enable)

    self.minLabelSize.setEnabled(enable)
    self.predIOUThresh.setEnabled(enable)
    self.stabilityScoreThresh.setEnabled(enable)
    self.stabilityScoreOffset.setEnabled(enable)
    self.boxNMSThresh.setEnabled(enable)
    self.circularityThresh.setEnabled(enable)
    self.blueChannelThreshFactor.setEnabled(enable)
    self.largestLabelSize.setEnabled(enable)

  def closeEvent(self, event):
    if self.thread is not None and self.thread.isRunning():
      self.thread.quit()
      self.thread.wait()
    super().closeEvent(event)

# Run the application
if __name__ == "__main__":
  app = QApplication(sys.argv)

  # set company and app name for QSettings
  app.setOrganizationName('IDSS')
  app.setApplicationName('Viridot2')

  window = MainGUI()
  window.show()
  sys.exit(app.exec())
