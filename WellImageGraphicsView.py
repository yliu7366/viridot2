"""
Custom QGraphicsView widget for:
1. display a well image with/without virus plages
2. display segmented label outlines
3. track mouse movement for SAM2 prompt based segmentation
"""
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItem
from PySide6.QtGui import QPainter, QPen, QFont, QFontMetrics, QColor
from PySide6.QtCore import Qt, QPoint, QRectF, Slot, Signal

class XORTextItem(QGraphicsItem):
  def __init__(self, text, pos, font=QFont("Arial", 10), use_xor=True, use_background=False):
    super().__init__()
    self.text = text
    self.pos = pos
    self.font = font
    self.use_xor = use_xor
    self.use_background = use_background
    self.setPos(pos)

  def boundingRect(self):
    """Define the bounding rectangle for the text item"""
    fm = QFontMetrics(self.font)
    text_width = fm.horizontalAdvance(self.text)
    text_height = fm.height()
    return QRectF(0, -text_height / 2, text_width, text_height)

  def paint(self, painter, option, widget=None):
    """Draw the text in XOR mode or with an outline, optionally with a background"""
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.TextAntialiasing, True)
    painter.setFont(self.font)

    # Get bounding rect for background
    fm = QFontMetrics(self.font)
    text_width = fm.horizontalAdvance(self.text)
    text_height = fm.height()
    padding = 2 if self.use_background else 0
    bg_rect = QRectF(-padding, -text_height / 2 - padding, text_width + 2 * padding, text_height + 2 * padding)

    if self.use_background:
      # Draw semi-transparent black background
      painter.setBrush(QColor(0, 0, 0, 100))  # Semi-transparent black
      painter.setPen(Qt.NoPen)
      painter.drawRect(bg_rect)

    if self.use_xor:
      # XOR mode with white pen for high contrast
      # TODO: doesn't work on Rocky linux, always draws black text color
      painter.setCompositionMode(QPainter.CompositionMode_Xor)
      painter.setPen(Qt.white)  # White for clear XOR effect
      painter.drawText(0, 0, self.text)
    else:
      # Fallback: White text
      painter.setPen(Qt.white)
      painter.drawText(0, 0, self.text)

    painter.restore()

class WellImageGraphicsView(QGraphicsView):
  # signal to trigger a click based segmentation
  requestSegmentation = Signal(float, float)

  def __init__(self, parent=None):
    super().__init__(parent)
    self.setScene(QGraphicsScene(self))
    self.setAlignment(Qt.AlignCenter)
    self.setMouseTracking(True)  # Enable mouse tracking

    self.current_pixmap = None
    self.pixmap_item = None

    self.outlines = []
    self.outline_color = Qt.yellow
    self.outline_thickness = 1
    self.outline_style = Qt.SolidLine
    self.mouse_pos = None
    self.clickToSegment = False

    self.original_pixmap_width = 1
    self.original_pixmap_height = 1

  @Slot(bool)
  def setSegmentationMode(self, enabled: bool):
    self.clickToSegment = enabled
    if self.clickToSegment:
      self.setCursor(Qt.CursorShape.CrossCursor)
    else:
      self.setCursor(Qt.CursorShape.ArrowCursor)

  def setImage(self, pixmap, outlines=None):
    """Set the image and outlines to display"""
    self.current_pixmap = pixmap
    self.outlines = outlines or []
    self.updateScene()

  def getBoundingBox(self, points):
    if not points:
      return QPoint(0, 0)
    
    xx = [p.x() for p in points]
    yy = [p.y() for p in points]

    minX, maxX = min(xx), max(xx)
    minY, maxY = min(yy), max(yy)

    cx = (minX + maxX) // 2
    cy = (minY + maxY) // 2

    return QPoint(cx, cy)
  
  def updateScene(self):
    """Update the QGraphicsScene with the current pixmap and outlines"""
    if not self.current_pixmap:
      self.scene().clear()
      self.pixmap_item = None
      return

    scene = self.scene()
    scene.clear()

    # Add pixmap to scene
    self.pixmap_item = scene.addPixmap(self.current_pixmap)
    scene.setSceneRect(QRectF(self.current_pixmap.rect()))

    # Draw outlines
    if self.outlines:
      xratio = self.current_pixmap.width() / self.original_pixmap_width
      yratio = self.current_pixmap.height() / self.original_pixmap_height
      for idx, contour in enumerate(self.outlines):
        points = [QPoint(x * xratio, y * yratio) for x, y in contour.squeeze()]
        # Create a QGraphicsPolygonItem for the outline
        poly_item = scene.addPolygon(points, QPen(self.outline_color, self.outline_thickness, self.outline_style))
        poly_item.setZValue(1)  # Ensure outlines are above the image

        # draw index at the center of each outline
        center = self.getBoundingBox(points)
        text_item = XORTextItem(str(idx+1), center, QFont("Arial", 10), False, False)
        text_item.setZValue(2)
        scene.addItem(text_item)

    self.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

  def setOriginalSize(self, width, height):
    """Store original pixmap dimensions for correct outline scaling"""
    self.original_pixmap_width = width if width > 0 else 1
    self.original_pixmap_height = height if height > 0 else 1

  def mouseMoveEvent(self, event):
    """Track mouse movement for SAM2 prompts"""
    #scene_pos = self.mapToScene(event.pos())
    #self.mouse_pos = (scene_pos.x(), scene_pos.y())
    # Optionally emit signal or store for SAM2
    # Example: print coordinates (replace with SAM2 integration)
    #print(f"Mouse position: ({scene_pos.x():.2f}, {scene_pos.y():.2f})")
    super().mouseMoveEvent(event)

  def mousePressEvent(self, event):
    super().mousePressEvent(event)

    """Handle mouse clicks for SAM2 prompts"""
    if event.button() == Qt.LeftButton and self.clickToSegment:
      if not self.pixmap_item:
        return
      
      scene_pos = self.mapToScene(event.pos())

      if self.pixmap_item.sceneBoundingRect().contains(scene_pos):
        
        # The pixmap item's top-left might not be at (0,0) if you move it,
        # so we transform the scene position to be relative to the item's origin.
        item_pos = self.pixmap_item.mapFromScene(scene_pos)

        pixmap = self.pixmap_item.pixmap()
        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()

        # Avoid division by zero
        if pixmap_width == 0 or pixmap_height == 0:
            return

        # Calculate normalized coordinates
        normalized_x = item_pos.x() / pixmap_width
        normalized_y = item_pos.y() / pixmap_height

        # The values should now be within [0.0, 1.0].
        # We can add a final clamp for safety.
        normalized_x = max(0.0, min(normalized_x, 1.0))
        normalized_y = max(0.0, min(normalized_y, 1.0))

        print(f"Normalized click: ({normalized_x:.3f}, {normalized_y:.3f})")

        # Emit the new signal with the normalized coordinates
        self.requestSegmentation.emit(normalized_x, normalized_y)

  def resizeEvent(self, event):
    """Adjust the view when resized"""
    if self.pixmap_item:
      self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
    super().resizeEvent(event)