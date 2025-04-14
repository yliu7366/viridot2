from PySide6.QtWidgets import (QPushButton, QDialog,
                               QVBoxLayout, QLineEdit,
                               QDialogButtonBox, QHBoxLayout,
                                QCheckBox, QLabel, QFileDialog)
from PySide6.QtCore import QSettings

class SettingsDialog(QDialog):
  def __init__(self, parent=None):
    super().__init__(parent)
    self.setWindowTitle("Settings")
    self.settings = QSettings("IDSS", "Viridot2")

    # Create layout and form
    self.layout = QVBoxLayout(self)
    
    self.modelPathLayout = QHBoxLayout()

    # Create settings fields
    self.modelPathLabel = QLabel("Model checkpoints:")
    self.modelPathInput = QLineEdit()
    self.modelPathButton = QPushButton("Browse")

    self.modelPathButton.clicked.connect(self.browseModelPath)

    self.modelPathLayout.addWidget(self.modelPathLabel)
    self.modelPathLayout.addWidget(self.modelPathInput, stretch=1)
    self.modelPathLayout.addWidget(self.modelPathButton)

    self.debugCheckBox = QCheckBox("Debug mode")

    # Load existing settings
    self.modelPathInput.setText(self.settings.value("modelpath", "../sam2/checkpoints"))
    self.debugCheckBox.setChecked(self.settings.value("debugmode", False, type=bool))

    # Add buttons
    self.buttons = QDialogButtonBox(
      QDialogButtonBox.Ok | QDialogButtonBox.Cancel
    )
    self.buttons.accepted.connect(self.save_settings)
    self.buttons.rejected.connect(self.reject)

    # Assemble layout
    self.layout.addLayout(self.modelPathLayout)
    self.layout.addWidget(self.debugCheckBox)
    self.layout.addWidget(self.buttons)

  def save_settings(self):
    """Save settings when OK is clicked"""
    self.settings.setValue("modelpath", self.modelPathInput.text())
    self.settings.setValue("debugmode", self.debugCheckBox.isChecked())
    self.accept()

  def browseModelPath(self):
    filePath = QFileDialog.getExistingDirectory(self, "Select Model Checkpoint Directory", self.modelPathInput.text())
    if filePath:
      self.modelPathInput.setText(filePath)