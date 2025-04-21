"""
Custom QGraphicsView widget for:
1. display a well image with/without virus plages
2. display segmented label outlines
3. track mouse movement for SAM2 prompt based segmentation
"""
import sys
import os
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                               QScrollBar, QLabel, QGridLayout, QScrollArea,
                               QGraphicsView, QGraphicsScene, QApplication, QSizePolicy)
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen
from PySide6.QtCore import Qt, QSize, QPoint, QRectF

class WellImageGraphicsView(QGraphicsView):
  def __init__(self, parent=None):
    super().__init__(parent)
    self.setScene(QGraphicsScene(self))
    self.setAlignment(Qt.AlignCenter)
    self.setMouseTracking(True)  # Enable mouse tracking
    self.current_pixmap = None
    self.outlines = []
    self.outline_color = Qt.yellow
    self.outline_thickness = 1
    self.outline_style = Qt.SolidLine
    self.mouse_pos = None

  def set_image(self, pixmap, outlines=None):
    """Set the image and outlines to display"""
    self.current_pixmap = pixmap
    self.outlines = outlines or []
    self.update_scene()

  def update_scene(self):
    """Update the QGraphicsScene with the current pixmap and outlines"""
    if not self.current_pixmap:
      return

    scene = self.scene()
    scene.clear()

    # Add pixmap to scene
    pixmap_item = scene.addPixmap(self.current_pixmap)
    scene.setSceneRect(QRectF(self.current_pixmap.rect()))

    # Draw outlines
    if self.outlines:
      xratio = self.current_pixmap.width() / self.original_pixmap_width
      yratio = self.current_pixmap.height() / self.original_pixmap_height
      for contour in self.outlines:
        points = [QPoint(x * xratio, y * yratio) for x, y in contour.squeeze()]
        # Create a QGraphicsPolygonItem for the outline
        poly_item = scene.addPolygon(points, QPen(self.outline_color, self.outline_thickness, self.outline_style))
        poly_item.setZValue(1)  # Ensure outlines are above the image

    self.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

  def set_original_size(self, width, height):
    """Store original pixmap dimensions for correct outline scaling"""
    self.original_pixmap_width = width
    self.original_pixmap_height = height

  def mouseMoveEvent(self, event):
    """Track mouse movement for SAM2 prompts"""
    scene_pos = self.mapToScene(event.pos())
    self.mouse_pos = (scene_pos.x(), scene_pos.y())
    # Optionally emit signal or store for SAM2
    # Example: print coordinates (replace with SAM2 integration)
    #print(f"Mouse position: ({scene_pos.x():.2f}, {scene_pos.y():.2f})")
    super().mouseMoveEvent(event)

  def mousePressEvent(self, event):
    """Handle mouse clicks for SAM2 prompts"""
    if event.button() == Qt.LeftButton:
      scene_pos = self.mapToScene(event.pos())
      # Store or process click for SAM2 (e.g., as a prompt point)
      #print(f"Mouse clicked at: ({scene_pos.x():.2f}, {scene_pos.y():.2f})")
      # Add your SAM2 prompt logic here
    super().mousePressEvent(event)

  def resizeEvent(self, event):
    """Adjust the view when resized"""
    if self.current_pixmap:
      self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
    super().resizeEvent(event)