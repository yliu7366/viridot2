from PySide6.QtWidgets import (QPushButton, QDialog, QPlainTextEdit,
                               QVBoxLayout, QLineEdit, QRadioButton,
                               QDialogButtonBox, QHBoxLayout, QProgressBar,
                                QCheckBox, QLabel, QFileDialog,
                                QGroupBox, QMessageBox)
#from PySide6.QtCore import QSettings

import os

class BatchProcessorDialog(QDialog):
  def __init__(self, parent=None):
    super().__init__(parent)
    self.setWindowTitle("Batch Processor")
    self.setMinimumSize(600, 450)

    # --- Widgets ---
    self.folder_path_edit = QLineEdit()
    self.folder_path_edit.setPlaceholderText("Select the root folder containing subfolders to process...")
    self.browse_button = QPushButton("Browse...")
        
    self.local_radio = QRadioButton("Sequential on Local GPU")
    self.slurm_radio = QRadioButton("Submit Slurm Job Array (HPC)")
    self.local_radio.setChecked(True)

    self.progress_bar = QProgressBar()
    self.log_output = QPlainTextEdit()
    self.log_output.setReadOnly(True)

    self.start_button = QPushButton("Start Processing")
    self.close_button = QPushButton("Close")

    # --- Layout ---
    layout = QVBoxLayout(self)

    # Folder selection layout
    folder_layout = QHBoxLayout()
    folder_layout.addWidget(self.folder_path_edit)
    folder_layout.addWidget(self.browse_button)
    layout.addLayout(folder_layout)

    # Processing options group
    options_group = QGroupBox("Processing Mode")
    options_layout = QVBoxLayout()
    options_layout.addWidget(self.local_radio)
    options_layout.addWidget(self.slurm_radio)
    options_group.setLayout(options_layout)
    layout.addWidget(options_group)

    # Progress and logging
    layout.addWidget(self.progress_bar)
    layout.addWidget(self.log_output)

    # Buttons
    button_layout = QHBoxLayout()
    button_layout.addStretch()
    button_layout.addWidget(self.start_button)
    button_layout.addWidget(self.close_button)
    layout.addLayout(button_layout)
        
    # --- Connections ---
    self.browse_button.clicked.connect(self.select_root_folder)
    self.start_button.clicked.connect(self.start_processing)
    self.close_button.clicked.connect(self.reject) # QDialog.reject() closes the dialog

  def select_root_folder(self):
    folder = QFileDialog.getExistingDirectory(self, "Select Root Folder")
    if folder:
      self.folder_path_edit.setText(folder)

  def log(self, message):
    self.log_output.appendPlainText(message)

  def start_processing(self):
    root_folder = self.folder_path_edit.text()
    if not os.path.isdir(root_folder):
      QMessageBox.warning(self, "Invalid Folder", "Please select a valid root folder.")
      return

    try:
      subfolders = [os.path.join(root_folder, d) for d in os.listdir(root_folder) 
                          if os.path.isdir(os.path.join(root_folder, d))]
    except OSError as e:
      QMessageBox.critical(self, "Error Reading Folder", f"Could not read subfolders: {e}")
      return

    if not subfolders:
      QMessageBox.information(self, "No Subfolders", "No subfolders found in the selected directory.")
      return

    self.start_button.setEnabled(False)
    self.close_button.setEnabled(False)
    self.log_output.clear()
    self.log(f"Found {len(subfolders)} subfolders to process.")

    """
    if self.local_radio.isChecked():
      self.run_local_processing(subfolders)
    else:
      self.run_slurm_submission(subfolders, root_folder)
      # Re-enable buttons immediately for Slurm as submission is fast
      self.start_button.setEnabled(True)
      self.close_button.setEnabled(True)
    """