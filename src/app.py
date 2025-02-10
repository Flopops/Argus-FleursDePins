"""Application for Argus"""
import os
import sys
from pathlib import Path
import time

import pandas as pd
from PyQt5 import QtWidgets, QtCore, QtGui
from sahi.predict import get_sliced_prediction

from models import models
from utils import data, files_manipulator, LOGGER, capture_print


class InferenceWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(bool)
    progress = QtCore.pyqtSignal(int)
    resultsReady = QtCore.pyqtSignal(dict)

    def __init__(self, model, images_paths):
        super().__init__()
        self.images_paths = images_paths
        self.model = model

    def run(self) -> None:
        LOGGER.info(f"Launching inference on {len(self.images_paths)} images")
        results = {i: 0 for i in self.images_paths}
        for i, image_path in enumerate(self.images_paths):
            self.progress.emit(i)
            if self.isInterruptionRequested():
                self.finished.emit(False)
                return
            with capture_print() as captured:
                result = get_sliced_prediction(
                    image_path,
                    self.model,
                    slice_height=320,
                    slice_width=320,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2,
                    verbose=2,
                )
            results[image_path] = [
                len(result.object_prediction_list),
                data.get_nb_objects_in_circle(
                    files_manipulator.get_dimensions(image_path),
                    result.object_prediction_list,
                ),
            ]
            LOGGER.info(
                f"From get_sliced_prediction : {captured.getvalue().strip()}"
                f"At {image_path}, circle selection: {results[image_path]}"
            )
        self.resultsReady.emit(results)
        self.finished.emit(True)


class MainWindow(QtWidgets.QMainWindow):
    images_paths = []
    results = {}
    inferenceProgressDialog: QtWidgets.QProgressDialog
    worker: InferenceWorker

    def __init__(self, model: models.PinesDetectionModel, parent=None):
        super().__init__(
            parent=parent,
            flags=QtCore.Qt.Window,
            windowIcon=QtGui.QIcon(
                str(
                    Path(__file__).parent.parent.parent / "assets" / "icon.ico"
                )
            ),
            acceptDrops=True,
            windowTitle="Argus : Dénombrement fleurs pins",
        )
        self.setCentralWidget(QtWidgets.QWidget(self, flags=QtCore.Qt.Widget))
        self.resize(800, 600)
        self.initUI()
        self._model: models.PinesDetectionModel = model

    def initUI(self):
        self.centerText = QtWidgets.QLabel(
            "Faire glisser le dossier des images ici",
        )
        # TODO: shortcut on &L
        self.launchDetectionButton = QtWidgets.QPushButton(
            "&Lancer la détection", enabled=False
        )
        self.launchDetectionButton.clicked.connect(self.onButtonClick)

        self.exportButton = QtWidgets.QPushButton(
            "&Exporter en Excel", enabled=False
        )
        self.exportButton.clicked.connect(self.export_result_to_excel)
        # FIXME: QLayout: Attempting to add QLayout "" to MainWindow "" which already has a layout
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addWidget(self.centerText, alignment=QtCore.Qt.AlignCenter)
        btn_layout = QtWidgets.QHBoxLayout(self)
        btn_layout.addWidget(self.launchDetectionButton)
        btn_layout.addWidget(self.exportButton)
        main_layout.addLayout(btn_layout)
        self.centralWidget().setLayout(main_layout)

    @QtCore.pyqtSlot()
    def onButtonClick(self):
        self.inferenceProgressDialog = QtWidgets.QProgressDialog(
            "Détection des fleurs sur les images",
            "Annuler",
            0,
            len(self.images_paths),
            self,
            windowModality=QtCore.Qt.WindowModal,
            labelText="Détection en cours n/N",
            # autoClose=False,
            # minimumDuration=0,
        )
        self.inferenceProgressDialog.canceled.connect(self.cancelTask)

        self.worker = InferenceWorker(self._model, self.images_paths)
        self.worker.finished.connect(self.finishedCallback)
        self.worker.progress.connect(self.inferenceProgressDialog.setValue)
        self.worker.resultsReady.connect(self.handleResults)
        self.worker.start()

    def cancelTask(self) -> None:
        if self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait()

    def finishedCallback(self, success: bool) -> None:
        if success:
            self.inferenceProgressDialog.setValue(len(self.images_paths))
            LOGGER.info("Inference on all the images done with no issues")
        else:
            LOGGER.warn("Inference has been stopped by cancel button!")
        self.exportButton.setEnabled(True)

    def handleResults(self, results) -> None:
        LOGGER.info(f"Results received : {results}")
        self.results = results

    def export_result_to_excel(self):
        # TODO: test this function
        LOGGER.info("Exporting to Excel")
        pd.DataFrame(
            [(key, *value) for key, value in self.results.items()],
            columns=["Nom image", "Nombre sans cercle", "Nombre avec cercle"],
        ).to_excel(
            excel_writer=Path(self.images_paths[0]).parent.parent
            / "output.xlsx",
            index=False,
        )

    def dragEnterEvent(self, event):
        # TODO: test this function
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        # TODO: test this function
        self.exportButton.setEnabled(False)
        self.launchDetectionButton.setEnabled(True)
        urls = event.mimeData().urls()
        # TODO: change system to add more files to self.images_paths instead of replacing
        if len(urls) == 1 and os.path.isdir(urls[0].toLocalFile()):
            self.images_paths = files_manipulator.images_paths_from_dir(
                QtCore.QDir(urls[0].toLocalFile())
            )
            if len(self.images_paths) == 0:
                self.launchDetectionButton.setEnabled(False)
        else:
            self.images_paths = [
                u.toLocalFile() for u in event.mimeData().urls()
            ]
        self.centerText.setText("\n".join(self.images_paths))


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    # Clean trained
    # checkpoint_path = (
    #     "../outputs/FasterRCNN 30 juin 0.53 top accury/best_model_AP.pth"
    # )
    # saved_checkpoint = torch.load(checkpoint_path)
    # model = models.get_fasterrcnn_mobilenet_v3()
    # model.load_state_dict(saved_checkpoint["model_state_dict"])

    # Dirty trained
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pines_detection_model = models.PinesDetectionModel(
        model=(
            models.get_fasterrcnn_mobilenet_v3(
                weights_path=str(
                    Path(__file__).parent.parent
                    / "pretrained-models"
                    / "fasterrcnn-scale1-bs32-epochs100-weights.pth"
                )
            )
        ),
        confidence_threshold=0.5,
        device=device,
        load_at_init=True,
    )
    LOGGER.info(f"Model loaded on {device}")

    widget = MainWindow(model=pines_detection_model)
    widget.show()

    sys.exit(app.exec())
