import os
import pickle
import pandas as pd

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QScrollBar, QLabel, QGridLayout, QScrollArea)
from PySide6.QtGui import QPixmap, QPainter, QPen
from PySide6.QtCore import Qt, QSize, QPoint, Signal

from WellImageGraphicsView import WellImageGraphicsView

class ImageViewerWidget(QWidget):
  # Signals
  currentSeg = Signal(dict) # return current image's segmentation

  def __init__(self, dataset=None, image_list=None, masks_outlines=None):
    super().__init__()
    self.image_list = image_list or []
    self.image_data = []
    self.masks_outlines = masks_outlines or []
    self.dataset_name = dataset or ""
    self.current_index = 0
    self.is_grid_view = False
    self.cols = 12

    self.outline_color = Qt.yellow
    self.outline_style = Qt.SolidLine
    self.outline_thickness = 1
    # Initialize UI
    self.initUI()

  def initUI(self):
    # Main layout
    self.main_layout = QVBoxLayout()

    # File name label
    self.filename_label = QLabel()
    self.filename_label.setAlignment(Qt.AlignCenter)
    self.filename_label.setStyleSheet("font-weight: bold;")  # Optional styling
    self.main_layout.addWidget(self.filename_label)

    # Image display area
    self.image_view = WellImageGraphicsView()
    self.image_view.setMinimumSize(1, 1)  # Allow shrinking

    # Scroll area for grid view
    self.scroll_area = QScrollArea()
    self.scroll_area.setWidgetResizable(True)
    self.grid_widget = QWidget()
    self.grid_layout = QGridLayout(self.grid_widget)
    self.scroll_area.setWidget(self.grid_widget)

    # Control layout
    control_layout = QHBoxLayout()

    # View toggle button
    self.view_btn = QPushButton('Show Grid')
    self.view_btn.clicked.connect(self.toggleView)
    control_layout.addWidget(self.view_btn)

    # Scroll bar for list view
    self.scroll_bar = QScrollBar(Qt.Horizontal)
    self.scroll_bar.setMinimum(0)
    self.scroll_bar.setMaximum(max(0, len(self.image_list) - 1))
    self.scroll_bar.setSingleStep(1)
    self.scroll_bar.valueChanged.connect(self.updateImage)
    control_layout.addWidget(self.scroll_bar, stretch=1)

    # Add widgets to main layout
    self.main_layout.addWidget(self.image_view, stretch=1)
    self.main_layout.addWidget(self.scroll_area)
    self.main_layout.addLayout(control_layout)

    self.setLayout(self.main_layout)

    # Initial setup
    self.scroll_area.hide()
    self.updateImage(0)

  def updateImage(self, value):
    """Update displayed image in list view"""
    if self.is_grid_view:
      return
    
    if self.image_list:
      self.current_index = value
      pixmap = self.image_data[self.current_index]
      if pixmap:
        # Scale image to fit window while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
          self.image_view.size(),
          Qt.KeepAspectRatio,
          Qt.SmoothTransformation
        )
        # Set image and outlines in CustomGraphicsView
        outlines = self.masks_outlines[self.current_index]['outlines'] if self.masks_outlines else []
        self.image_view.set_image(scaled_pixmap, outlines)
        self.image_view.set_original_size(pixmap.width(), pixmap.height())
        # Update filename label
        filename = os.path.basename(self.image_list[self.current_index])
        self.filename_label.setText(self.dataset_name + ' - ' + filename)
        # signal main GUI to update the label list widget
        if self.masks_outlines:
          self.currentSeg.emit(self.masks_outlines[self.current_index])
    else:
      self.filename_label.setText("No image loaded")

  def loadImage(self, image_path):
    """Load image from path"""
    try:
      pixmap = QPixmap(image_path)
      if pixmap.isNull():
        return None
      return pixmap
    except Exception as e:
      print(f"Error loading image: {e}")
      return None

  def createGridView(self):
    """Create grid view with thumbnails"""
    # Clear existing grid
    for i in reversed(range(self.grid_layout.count())):
      self.grid_layout.itemAt(i).widget().setParent(None)

    spacings = 9
    margins = self.grid_layout.getContentsMargins()
    self.grid_layout.setSpacing(spacings)

    margin_and_spacing = spacings*(self.cols-1) + margins[0] + margins[2]

    thumbSize = (self.grid_widget.size().width() - margin_and_spacing) // self.cols
    thumbSize = 100 if thumbSize < 100 else thumbSize

    # Add thumbnails
    cols = self.cols  # Number of columns in grid
    for i, _ in enumerate(self.image_list):
      pixmap = self.image_data[i]
      if pixmap:
        # Create thumbnail
        thumb = pixmap.scaled(
          QSize(thumbSize, thumbSize),
          Qt.KeepAspectRatio,
          Qt.SmoothTransformation
        )
        # draw contours
        if self.masks_outlines:
          xratio = thumb.width() / pixmap.width()
          yratio = thumb.height() / pixmap.height()
          painter = QPainter(thumb)
          pen = QPen(self.outline_color, self.outline_thickness, self.outline_style)
          painter.setPen(pen)
          for contour in self.masks_outlines[i]['outlines']:
            points = [QPoint(x * xratio, y * yratio) for x, y in contour.squeeze()]
            painter.drawPolygon(points)
          painter.end()
        label = QLabel()
        label.setPixmap(thumb)
        label.setAlignment(Qt.AlignCenter)

        label.setMouseTracking(True)
        label.mouseDoubleClickEvent = lambda event, index=i: self.onThumbnailDoubleClicked(index)

        # Calculate position in grid
        row = i // cols
        col = i % cols
        self.grid_layout.addWidget(label, row, col)
    self.filename_label.setText("Grid View - " + self.dataset_name)

  def onThumbnailDoubleClicked(self, index):
    """Handle double-click on thumbnail: switch to list view and show image"""
    self.is_grid_view = False
    self.view_btn.setText('Show Grid')
    self.scroll_area.hide()
    self.image_view.show()
    self.scroll_bar.show()
    self.current_index = index
    self.scroll_bar.setValue(index)  # Update scrollbar to match

  def rebuildView(self):
    if self.is_grid_view:
      self.view_btn.setText('Show List')
      self.image_view.hide()
      self.scroll_bar.hide()
      self.createGridView()
      self.scroll_area.show()
    else:
      self.view_btn.setText('Show Grid')
      self.scroll_area.hide()
      self.image_view.show()
      self.scroll_bar.show()
      self.scroll_bar.setValue(self.current_index)

  def toggleView(self):
    """Switch between list and grid view"""
    self.is_grid_view = not self.is_grid_view
    self.rebuildView()

  def resizeEvent(self, event):
    """Handle window resize"""
    super().resizeEvent(event)
    if not self.is_grid_view and self.image_list:
      self.updateImage(self.current_index)
    elif self.is_grid_view and self.image_list:
      self.createGridView()

  def updateDataset(self, data_dict):
    self.image_list = data_dict['names'] or []
    self.image_data = []
    self.masks_outlines = []

    for name in self.image_list:
      img = self.loadImage(name)
      if img:
        self.image_data.append(img)

    assert len(self.image_list) == len(self.image_data)

    self.dataset_name = data_dict['dataset'] or ""
    self.scroll_bar.setMaximum(max(0, len(self.image_list) - 1))
    
    if self.is_grid_view:
      self.createGridView()
    else:
      self.scroll_bar.setValue(0)
      self.updateImage(0) # why scroll_bar.setValue doesn't trigger updateImage?

  def updateMasksOutlines(self, masks_outlines):
    self.masks_outlines = masks_outlines

    self.current_index = 0
    self.rebuildView()
    self.updateImage(self.current_index)