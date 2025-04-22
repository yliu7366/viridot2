"""
Custom QGraphicsView widget for:
1. display a well image with/without virus plages
2. display segmented label outlines
3. track mouse movement for SAM2 prompt based segmentation
"""
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItem
from PySide6.QtGui import QPainter, QPen, QFont, QFontMetrics, QColor
from PySide6.QtCore import Qt, QPoint, QRectF

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

  def get_bounding_box(self, points):
    if not points:
      return QPoint(0, 0)
    
    xx = [p.x() for p in points]
    yy = [p.y() for p in points]

    minX, maxX = min(xx), max(xx)
    minY, maxY = min(yy), max(yy)

    cx = (minX + maxX) // 2
    cy = (minY + maxY) // 2

    return QPoint(cx, cy)
  
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
      for idx, contour in enumerate(self.outlines):
        points = [QPoint(x * xratio, y * yratio) for x, y in contour.squeeze()]
        # Create a QGraphicsPolygonItem for the outline
        poly_item = scene.addPolygon(points, QPen(self.outline_color, self.outline_thickness, self.outline_style))
        poly_item.setZValue(1)  # Ensure outlines are above the image

        # draw index at the center of each outline
        center = self.get_bounding_box(points)
        text_item = XORTextItem(str(idx+1), center, QFont("Arial", 10), False, False)
        text_item.setZValue(2)
        scene.addItem(text_item)

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