from PySide6.QtWidgets import (QPushButton, QDialog, QTreeWidget,
                               QVBoxLayout, QLineEdit, QRadioButton,
                               QHBoxLayout, QFileDialog, QHeaderView,
                                QGroupBox, QMessageBox, QTreeWidgetItem)
from PySide6.QtCore import Signal

import os
import glob

from SAM2Segmentor import SAM2Worker

class BatchProcessorDialog(QDialog):
  # The signal still emits the list of subfolders to process, the mode, and the root folder.
  processing_requested = Signal(list, str, str)

  def __init__(self, parent=None):
    super().__init__(parent)
    self.setWindowTitle("Batch Processor Setup")
    self.setMinimumSize(700, 500) # Increased size for the tree view
    self.setModal(True)

    layout = QVBoxLayout(self)

    # Folder selection
    folder_layout = QHBoxLayout()
    self.folder_path_edit = QLineEdit()
    self.folder_path_edit.setPlaceholderText("Select the root folder...")
    self.browse_button = QPushButton("Browse...")
    folder_layout.addWidget(self.folder_path_edit)
    folder_layout.addWidget(self.browse_button)
    layout.addLayout(folder_layout)

    # Add a QTreeWidget for displaying folder contents
    self.folder_preview_tree = QTreeWidget()
    self.folder_preview_tree.setColumnCount(2)
    self.folder_preview_tree.setHeaderLabels(["Subfolder Name", "Well Images (.CTL) Found"])
    self.folder_preview_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    self.folder_preview_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
    layout.addWidget(self.folder_preview_tree)

    # Processing options
    options_group = QGroupBox("Processing Mode")
    options_layout = QVBoxLayout()
    self.local_radio = QRadioButton("Sequential on Local Machine")
    self.slurm_radio = QRadioButton("Submit Slurm Job Array (HPC)")
    self.local_radio.setChecked(True)
    options_layout.addWidget(self.local_radio)
    options_layout.addWidget(self.slurm_radio)
    options_group.setLayout(options_layout)
    layout.addWidget(options_group)

    # Buttons
    button_layout = QHBoxLayout()
    self.start_button = QPushButton("Start Batch")
    self.cancel_button = QPushButton("Cancel")
    button_layout.addStretch()
    button_layout.addWidget(self.start_button)
    button_layout.addWidget(self.cancel_button)
    layout.addLayout(button_layout)
    
    # Connections
    self.browse_button.clicked.connect(self.select_root_folder)
    self.folder_path_edit.textChanged.connect(self.scan_root_folder)
    self.start_button.clicked.connect(self.start_and_close)
    self.cancel_button.clicked.connect(self.reject)

    self.found_subfolders = []
    self.start_button.setEnabled(False)

  def select_root_folder(self):
    folder = QFileDialog.getExistingDirectory(self, "Select Root Folder")
    if folder:
      self.folder_path_edit.setText(folder)

  # The core scanning and UI update logic
  def scan_root_folder(self, root_path):
    """
    Scans the given path for subfolders and populates the tree widget.
    """
    self.folder_preview_tree.clear()
    self.found_subfolders = [] # Reset the list
    self.start_button.setEnabled(False) # Disable until scan is complete

    if not os.path.isdir(root_path):
      return

    try:
      for item_name in sorted(os.listdir(root_path)):
        full_path = os.path.join(root_path, item_name)
        if os.path.isdir(full_path):
          ctl_files = glob.glob(os.path.join(full_path, '*.CTL'))
          num_ctl_files = len(ctl_files)
          
          # skip empty subfolders
          if num_ctl_files > 0:
            tree_item = QTreeWidgetItem([item_name, str(num_ctl_files)])
            self.folder_preview_tree.addTopLevelItem(tree_item)
            
            self.found_subfolders.append(full_path)

    except OSError as e:
      error_item = QTreeWidgetItem([f"Error reading directory: {e}"])
      self.folder_preview_tree.addTopLevelItem(error_item)
    
    if self.found_subfolders:
      self.start_button.setEnabled(True)

  def start_and_close(self):
    if not self.found_subfolders:
      QMessageBox.warning(self, "No Folders", "No valid subfolders to process.")
      return

    mode = "slurm" if self.slurm_radio.isChecked() else "local"
    root_folder = self.folder_path_edit.text()

    self.processing_requested.emit(self.found_subfolders, mode, root_folder)

    self.accept()
