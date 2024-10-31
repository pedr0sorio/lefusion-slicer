import logging
import os
from typing import Annotated, Dict, Optional, Union

import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode

import numpy as np
import tempfile
import threading
import requests
import time
import datetime
from copy import deepcopy

# Anatomical coordinate system (IJK)
# i (patient right to left)
# j (anterior to posterior)
# k (anatomical height - inferior to superior))
FIXED_CROP_SIZE_IJK = [64, 64, 32]  # voxel (IJK)
FIXED_CROP_SIZE_KJI = [32, 64, 64]  # voxel (KJI)
FIXED_RESOLUTION = [1, 1, 2]  # mm (IJK)

#
# Utils
#


def arrayFromVTKMatrix(
    vmatrix: Union[vtk.vtkMatrix4x4, vtk.vtkMatrix3x3]
) -> np.ndarray:
    """Return vtkMatrix4x4 or vtkMatrix3x3 elements as numpy array.
    The returned array is just a copy and so any modification in the array will not affect the input matrix.
    To set VTK matrix from a numpy array, use :py:meth:`vtkMatrixFromArray` or
    :py:meth:`updateVTKMatrixFromArray`.
    From: https://discourse.slicer.org/t/vtk-transform-matrix-as-python-list-tuple-array/11797/2
    """
    if isinstance(vmatrix, vtk.vtkMatrix4x4):
        matrixSize = 4
    elif isinstance(vmatrix, vtk.vtkMatrix3x3):
        matrixSize = 3
    else:
        raise RuntimeError("Input must be vtk.vtkMatrix3x3 or vtk.vtkMatrix4x4")
    narray = np.eye(matrixSize)
    vmatrix.DeepCopy(narray.ravel(), vmatrix)
    return narray


#
# LeFusion
#


class LeFusion(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _(
            "LeFusion"
        )  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [
            translate("qSlicerAbstractCoreModule", "Inpainting (dev)")
        ]
        self.parent.dependencies = (
            []
        )  # TODO: add here list of module names that this module requires
        self.parent.contributors = [
            "Pedro Osorio (Bayer AG)",
        ]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _(
            """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#MedSAM2">module documentation</a>.
"""
        )
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _(
            """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""
        )

        # Additional initialization step after application startup is complete


#
# LeFusionParameterNode
#


@parameterNodeWrapper
class LeFusionParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputVolume: vtkMRMLScalarVolumeNode
    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode


#
# LeFusionWidget
#


class LeFusionWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/LeFusion.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = LeFusionLogic()
        self.logic.widget = self

        # Connections

        # Setting icons
        # Icons used here are downloaded from flaticon's free icons package. Detailed attributes can be found in slicer/MedSAM2/MedSAM2/Resources/Icons/attribute.html
        from PythonQt.QtGui import QIcon

        iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")
        self.ui.ResizeButton.setIcon(QIcon(os.path.join(iconsPath, "verify.png")))
        self.ui.ProcessROIpushButton.setIcon(
            QIcon(os.path.join(iconsPath, "verify.png"))
        )
        self.ui.btnROI.setIcon(QIcon(os.path.join(iconsPath, "bounding-box.png")))
        self.ui.InpaintPushButton.setIcon(
            QIcon(os.path.join(iconsPath, "body-scan.png"))
        )
        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose
        )
        self.addObserver(
            slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose
        )

        # Preprocessing
        # TODO Possibly add more preprocessing options depending on preliminary results
        self.ui.ResizeButton.connect("clicked(bool)", self.logic.resize_CT_slicer)

        # Define Hyperparameters for model inference
        self.ui.histType.addItems(["type 1", "type 2", "type 3"])
        self.ui.pathModel.connect(
            "currentPathChanged(const QString&)",
            lambda: setattr(self.logic, "newModelUploaded", False),
        )

        # Buttons
        self.ui.ResizeButton.connect("clicked(bool)", self.logic.resize_CT_slicer)
        self.ui.btnROI.connect("clicked(bool)", self.drawBBox)
        # Process selections to match exapected model input dimensions
        self.ui.ProcessROIpushButton.connect(
            "clicked(bool)",
            lambda: self.processBBox(size=FIXED_CROP_SIZE_IJK, use_resized=True),
        )

        # Inpainting on selected ROIs
        self.ui.InpaintPushButton.connect("clicked(bool)", self.logic.inpaint)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def setManualPreprocessVis(self, visible):
        self.ui.lblLevel.setVisible(visible)
        self.ui.lblWidth.setVisible(visible)
        self.ui.sldWinLevel.setVisible(visible)
        self.ui.sldWinWidth.setVisible(visible)

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass(
                "vtkMRMLScalarVolumeNode"
            )
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(
        self, inputParameterNode: Optional[LeFusionParameterNode]
    ) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)

    def drawBBox(self):
        """
        Draws a bounding box (BBox) in the Slicer scene.

        This method creates a new ROI node in the Slicer scene,
        sets it as the active place node, and configures the interaction mode to
        allow placing the ROI in the scene. The glyph scale and interaction handle
        scale of the ROI's display node are also adjusted.

        Note:
            This method is adapted from the implementation found at:
            https://github.com/bingogome/samm/blob/7da10edd7efe44d10369aa13eddead75a7d3a38a/samm/SammBase/SammBaseLib/WidgetSammBase.py

        """
        # Add new node to the scene
        planeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode").GetID()
        # ROI node is now the target for placement operations.
        selectionNode = slicer.mrmlScene.GetNodeByID("vtkMRMLSelectionNodeSingleton")
        selectionNode.SetReferenceActivePlaceNodeID(planeNode)
        # Let user select the ROI
        interactionNode = slicer.mrmlScene.GetNodeByID(
            "vtkMRMLInteractionNodeSingleton"
        )
        placeModePersistence = 0
        interactionNode.SetPlaceModePersistence(placeModePersistence)
        # mode 1 is Place, can also be accessed via slicer.vtkMRMLInteractionNode().Place
        interactionNode.SetCurrentInteractionMode(1)

        slicer.mrmlScene.GetNodeByID(planeNode).GetDisplayNode().SetGlyphScale(0.5)
        slicer.mrmlScene.GetNodeByID(
            planeNode
        ).GetDisplayNode().SetInteractionHandleScale(1)

    def processBBox(self, size=FIXED_CROP_SIZE_IJK, use_resized=True):
        """
        Processes all existing ROI nodes to have a fixed dimension, while still enabling user dragging.
        """
        # Get right volume node (og or resampled if preferred and exists)
        if use_resized:
            self.logic.captureImage(resized=True)
            volume_node = self.logic.ResizedVolumeNode
        else:
            self.logic.captureImage()
            volume_node = self.logic.volume_node

        # Get rotation and scaling matrix from IJK to RAS
        ijk_to_ras_RotationMatrix = self.logic.FromIJKToRAS(
            ijk_point=size, volume_node=volume_node, return_matrix=True
        )["matrix"][:3, :3]

        # Actual trasnform the crop dimensions from IJK to RAS
        size_ras = np.abs(np.dot(ijk_to_ras_RotationMatrix, size)).tolist()
        print(f"crop shape in IKJ (2x1x1) mm :  {size     =} (vx)")
        print(f"crop shape in RAS (2x1x1) mm :  {size_ras =} (mm)")
        roiNodes = slicer.util.getNodesByClass("vtkMRMLMarkupsROINode")
        for roiNode in roiNodes:
            roiNode.SetSize(size_ras)

    def lockBBox(self):
        roiNodes = slicer.util.getNodesByClass("vtkMRMLMarkupsROINode")
        for roiNode in roiNodes:
            # do not let the user resize the box
            roiNode.GetDisplayNode().SetHandlesInteractive(False)


#
# LeFusionLogic
#


class LeFusionLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    boundaries = None
    volume_node = None
    ResizedVolumeNode = None
    image_data = None
    widget = None
    middleMaskNode = None
    allSegmentsNode = None
    newModelUploaded = False
    segmentation_res_path = "/home/rasakereh/Desktop"

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

        # Clear the python console
        slicer.app.pythonConsole().clear()

    def getParameterNode(self):
        return LeFusionParameterNode(super().getParameterNode())

    def captureImage(self, resized=False):
        """Gets the self.logic.image_data and self.logic.volume_node attributes either
        from resized or original resolution volume."""
        print("Updating image data and volume node self.logic variables.")
        if not resized:
            # Load first volume node in the scene
            print("  from original volume")
            self.volume_node = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")[0]
            if (
                self.volume_node.GetNodeTagName() == "LabelMapVolume"
            ):  ### some volumes are loaded as LabelMapVolume instead of ScalarVolume, temporary
                outputvolume = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLScalarVolumeNode", self.volume_node.GetName()
                )
                sef = slicer.modules.volumes.logic().CreateScalarVolumeFromVolume(
                    slicer.mrmlScene, outputvolume, self.volume_node
                )
                slicer.mrmlScene.RemoveNode(self.volume_node)

                appLogic = slicer.app.applicationLogic()
                selectionNode = appLogic.GetSelectionNode()
                selectionNode.SetActiveVolumeID(sef.GetID())
                appLogic.PropagateVolumeSelection()
                self.volume_node = sef

            self.image_data = slicer.util.arrayFromVolume(
                self.volume_node
            )  ################ Only one node?
        else:
            # If the volume has been resized, use the resized volume
            if self.ResizedVolumeNode is None:
                raise ValueError(
                    "Resized volume not found. Please Resize the "
                    "volume before procedding."
                )
            print("  from resized volume")
            self.image_data = slicer.util.arrayFromVolume(self.ResizedVolumeNode)

        # NOTE that numpy array index order is kji (not ijk)

        print("     updated image data shape is: ", self.image_data.shape)

    def FromRASToIJK(self, ras_point, volume_node=None, return_matrix=False) -> dict:
        """
        Transforms a point from RAS to IJK coordinates given the volume node. Uses
        teh 4x4 affine trasnform matrix, including rotation, scaling (to account for
        the spacing) and translation.
        """  # If volume node is transformed, apply that transform to get volume's RAS coordinates
        # from: https://slicer.readthedocs.io/en/latest/developer_guide/script_repository/volumes.html#:~:text=Get%20volume%20voxel,%EF%83%81
        transformRasToVolumeRas = vtk.vtkGeneralTransform()
        if volume_node is None:
            volume_node = self.volume_node
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
            None, volume_node.GetParentTransformNode(), transformRasToVolumeRas
        )

        # Get point coordinate in RAS
        point_VolumeRas = transformRasToVolumeRas.TransformPoint(ras_point)

        # Get voxel coordinates from physical coordinates
        volumeRasToIjk = vtk.vtkMatrix4x4()
        volume_node.GetRASToIJKMatrix(volumeRasToIjk)
        # # print(f"{dir(volumeRasToIjk) = }")
        # print(f"{[volumeRasToIjk.GetElement(0,j) for j in range(4)] = }")
        # print(f"{[volumeRasToIjk.GetElement(1,j) for j in range(4)] = }")
        # print(f"{[volumeRasToIjk.GetElement(2,j) for j in range(4)] = }")
        # print(f"{[volumeRasToIjk.GetElement(3,j) for j in range(4)] = }")
        point_Ijk = [0, 0, 0, 1]
        volumeRasToIjk.MultiplyPoint(np.append(point_VolumeRas, 1.0), point_Ijk)
        point_Ijk = [int(round(c)) for c in point_Ijk[0:3]]

        return_dict = {
            "point": point_Ijk,
        }
        if return_matrix:
            return_dict["matrix"] = arrayFromVTKMatrix(volumeRasToIjk)
        return return_dict

    def FromIJKToRAS(self, ijk_point, volume_node=None, return_matrix=False) -> dict:
        """
        Transforms a point from IJK to RAS coordinates given the volume node. Uses
        teh 4x4 affine trasnform matrix, including rotation, scaling (to account for
        the spacing) and translation.
        """
        # Get physical coordinates from voxel coordinates
        volumeIjkToRas = vtk.vtkMatrix4x4()
        if volume_node is None:
            volume_node = self.volume_node
        volume_node.GetIJKToRASMatrix(volumeIjkToRas)
        point_VolumeRas = [0, 0, 0, 1]
        volumeIjkToRas.MultiplyPoint(np.append(ijk_point, 1.0), point_VolumeRas)

        # If volume node is transformed, apply that transform to get volume's RAS coordinates
        transformVolumeRasToRas = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
            volume_node.GetParentTransformNode(), None, transformVolumeRasToRas
        )
        point_Ras = transformVolumeRasToRas.TransformPoint(point_VolumeRas[0:3])

        return_dict = {
            "point": point_Ras,
        }
        if return_matrix:
            return_dict["matrix"] = arrayFromVTKMatrix(volumeIjkToRas)
        return return_dict

    def getRASBoundFromMarkupsROINode(self, roiNode):
        """
        Return range in RAS coordinates of the possible coordinate values fo reach axis.
        [xmin, xmax, ymin, ymax, zmin, zmax] in RAS coordinates (mmm).
        """
        # Option 1: Uisng buoltin GetBounds method
        bounds = np.zeros(6)
        roiNode.GetBounds(bounds)
        # Option 2: Using GetCenter and GetSize
        # center = [0] * 3
        # roiNode.GetCenter(center)
        # roi_points_ras = [
        #     (x - s / 2, x + s / 2) for x, s in zip(center, roiNode.GetSize())
        # ]
        # roi_points_ras = [item for sublist in roi_points_ras for item in sublist]
        return bounds

    def get_bounding_box(self) -> dict:
        """Can only be computed on resized volume."""
        # Make sure volume voxeld data is available at self.image_data
        self.captureImage(resized=True)
        # Get all the ROI nodes
        roiNodes = slicer.util.getNodesByClass("vtkMRMLMarkupsROINode")
        # If volume node is transformed, apply that transform to get volume's RAS coordinates
        # from: https://slicer.readthedocs.io/en/latest/developer_guide/script_repository/volumes.html#:~:text=Get%20volume%20voxel,%EF%83%81
        transformRasToVolumeRas = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
            None,
            self.ResizedVolumeNode.GetParentTransformNode(),
            transformRasToVolumeRas,
        )

        bboxes_ijk, bboxes_kji_numpy = [], []
        for roiNode in roiNodes:

            # For each MarkUpROI node, get the bounding box in RAS coordinates and convert them to IJK
            bounds = self.getRASBoundFromMarkupsROINode(roiNode=roiNode)
            print("RAS bounds are: ", bounds)

            # Get the most largest diagonal vertices and compute their IJK coordinates
            # i.e. this will implicitly define the IJK coordinates of the bounding box
            point1 = bounds[::2].copy()
            point2 = bounds[1::2].copy()
            centre = [0] * 3
            roiNode.GetCenter(centre)

            point1_ijk = self.FromRASToIJK(
                point1,
                volume_node=self.ResizedVolumeNode,
            )["point"]
            point2_ijk = self.FromRASToIJK(
                point2,
                volume_node=self.ResizedVolumeNode,
            )["point"]
            centre_ijk = self.FromRASToIJK(
                centre,
                volume_node=self.ResizedVolumeNode,
            )["point"]
            print(f"{centre = } \n{centre_ijk =}")

            # Change order to ensure lower bound is first
            for j in range(len(point1_ijk)):
                if point1_ijk[j] > point2_ijk[j]:
                    point1_ijk[j], point2_ijk[j] = point2_ijk[j], point1_ijk[j]

            # Get the bounding box in IJK coordinates
            bounds_ijk = [
                coord for pair in zip(point1_ijk, point2_ijk) for coord in pair
            ]
            print("IJK bounds are: ", bounds_ijk)
            # Get the bounding box in IKJ coordinates for direct indexing of the numpy array
            # # NOTE that numpy array index order is kji (not ijk)
            bounds_kji_numpy = bounds_ijk.copy()
            bounds_kji_numpy[0:2] = bounds_ijk[-2:]
            bounds_kji_numpy[-2:] = bounds_ijk[0:2]
            # Save
            bboxes_ijk.append(bounds_ijk)
            bboxes_kji_numpy.append(bounds_kji_numpy)

            print(
                "Testing bounding box on numpy image with "
                f"shape: \n Before crop: {self.image_data.shape} \n After crop: ",
                self.image_data[
                    bounds_kji_numpy[0] : bounds_kji_numpy[1],
                    bounds_kji_numpy[2] : bounds_kji_numpy[3],
                    bounds_kji_numpy[4] : bounds_kji_numpy[5],
                ].shape,
            )

        return {
            "bboxes_ras": bounds,
            "bboxes_ijk": bboxes_ijk,
            "bboxes_kji_numpy": bboxes_kji_numpy,
        }

    def run_on_background(self, target, args, title):
        self.progressbar = slicer.util.createProgressDialog(autoClose=False)
        self.progressbar.minimum = 0
        self.progressbar.maximum = 0
        self.progressbar.setLabelText(title)

        job_event = threading.Event()
        paral_thread = threading.Thread(
            target=target,
            args=(
                *args,
                job_event,
            ),
        )
        paral_thread.start()
        while not job_event.is_set():
            slicer.app.processEvents()
        paral_thread.join()

        self.progressbar.close()

    def inpainting_helper(self, img_path, result_path, ip, port, job_event):
        """Send and retrieve data from server for inpainting task."""
        # TODO later add choice for specif weights. For now assuming all weights are
        # already in the server.

        # 1. Send whole processed volume and bounding boxes to server
        self.progressbar.setLabelText(" uploading image... ")
        upload_url = "http://%s:%s/upload" % (ip, port)

        with open(img_path, "rb") as file:
            files = {"file": file}
            response = requests.post(upload_url, files=files)

        # 2. Run inference script on server
        self.progressbar.setLabelText(" inpainting ... ")
        run_script_url = "http://%s:%s/run_script" % (ip, port)

        data_dict = {
            "input": os.path.basename(img_path),
            "histogram": self.widget.ui.histType.currentText,
            # TODO Add here other hps for the inpainting model
        }
        print(f"data sent is: {data_dict}")
        response = requests.post(
            run_script_url,
            data=data_dict,
        )

        # 3. Downloading results saved on server file system
        self.progressbar.setLabelText(" downloading results... ")
        download_file_url = "http://%s:%s/download_file" % (ip, port)
        response = requests.get(
            download_file_url,
            data={"output": "data/video/segs_tiny/%s" % os.path.basename(img_path)},
        )

        with open(result_path, "wb") as f:
            f.write(response.content)

        job_event.set()

    def inpaint(self):
        # Get the whole volume
        self.captureImage(resized=True)

        # Get the bounding box for each ROI
        bboxes_dict = self.get_bounding_box()

        # TODO Save the image data, bboxes and potential inference HPs to a temp directory
        # NOTE First release will use the same preset LESION MASK for every inpainting task. Once DiffMask is
        # avaialble, the mask will be infered from the bounding box.
        with tempfile.TemporaryDirectory() as tmpdirname:
            img_path = "%s/img_data.npz" % (tmpdirname,)
            result_path = "%s/result.npz" % (tmpdirname,)
            np.savez(
                img_path,
                imgs=self.image_data,  # voxel array
                # list of all ROI's bbox
                boxes_numpy=bboxes_dict["bboxes_kji_numpy"],  # in numpy array indexes
                # boxes_ijk=bboxes_dict["bboxes_ijk"],  # in ikj indexes
                # boxes_ras=bboxes_dict["bbox_ras"],  # in ras coordinates
            )

            # Send the data to the server and get response saved in result_path
            self.run_on_background(
                self.inpainting_helper,
                (
                    img_path,
                    result_path,
                    self.widget.ui.txtIP.plainText.strip(),  # IP address
                    self.widget.ui.txtPort.plainText.strip(),  # Port number
                ),
                "Inpainting...",
            )
            # TODO from sever side:
            # - Simple test function to see receiving format
            # - Data loading fucntionality for:
            #    - Extract crops from the volume based on bboxes
            #    - Feed the crops as torch tensor for the inference pipeline
            # - Save inpainted crop in server local file system
            # - Send ONLY the inpainted crops back to the client

            # TODO Get Response and create a new volume node with the inpainted lesion
            # Loading results: Expects the inpainted crop with lesion
            # inpainted_crop_numpy = np.load(result_path, allow_pickle=True)["crop"]
            inpainted_crop_numpy = np.array(
                [np.zeros(FIXED_CROP_SIZE_KJI) for _ in bboxes_dict["bboxes_kji_numpy"]]
            )
            self.addInpaintedCrops(
                inpainted_crop_numpy=inpainted_crop_numpy,
                bboxes_dict=bboxes_dict,
                volume_node=self.ResizedVolumeNode,
            )

        # Remove selected ROIs
        roiNodes = slicer.util.getNodesByClass("vtkMRMLMarkupsROINode")
        for roiNode in roiNodes:
            slicer.mrmlScene.RemoveNode(roiNode)

    def addInpaintedCrops(
        self, inpainted_crop_numpy: np.ndarray, bboxes_dict: Dict, volume_node
    ):
        """Add the inpainted crop to a volume node in the scene."""
        # self.captureImage(resized=True) -> Not needed because it is already loaded
        # Edit the volume data i.e. self.image_data
        inpainted_volume = deepcopy(self.image_data)

        # Iteratively replace the multiple inpainted crops in the volume
        for idx, bbox in enumerate(bboxes_dict["bboxes_kji_numpy"]):
            print("inpainted_crop_numpy shape: ", inpainted_crop_numpy.shape)
            print(
                f"{inpainted_volume[bbox[0] : bbox[1], bbox[2] : bbox[3], bbox[4] : bbox[5]].shape = }"
            )
            inpainted_volume[
                bbox[0] : bbox[1], bbox[2] : bbox[3], bbox[4] : bbox[5]
            ] = inpainted_crop_numpy[idx]

        # Update resized volume node with the inpainted volume
        self.updateImage(inpainted_volume, volume_node)

    def resize_CT_slicer(self):
        """
        Use Slicer's resample scalar volume module to resize the volume to a fixed
        resolution. Creates a new volume node with the resized volume.
        """
        print(
            "[Volume Resizing] "
            "Resizing volume to fixed (2x1x1) mm expected by the inpainuting model."
        )
        self.captureImage()
        volumesLogic = slicer.modules.volumes.logic()
        self.ResizedVolumeNode = volumesLogic.CloneVolume(
            slicer.mrmlScene, self.volume_node, "ResizedVolumeSlicer"
        )
        # Set parameters
        parameters = {
            "InputVolume": self.volume_node.GetID(),
            "referenceVolume": self.volume_node.GetID(),
            "OutputVolume": self.ResizedVolumeNode.GetID(),
            "outputPixelSpacing": FIXED_RESOLUTION,
            "interpolationMode": "linear",
        }
        # Execute
        resampleModule = slicer.modules.resamplescalarvolume
        cliNode = slicer.cli.runSync(resampleModule, None, parameters)
        # Process results
        if cliNode.GetStatus() & cliNode.ErrorsMask:
            # error
            errorText = cliNode.GetErrorText()
            slicer.mrmlScene.RemoveNode(cliNode)
            raise ValueError("CLI execution failed: " + errorText)
        # success
        slicer.mrmlScene.RemoveNode(cliNode)
        self.captureImage(resized=True)

    def processScanServer(self):
        """
        Use Slicer's resample scalar volume module to resize the volume to a fixed
        resolution. Creates a new volume node with the resized volume.
        """
        print(
            "[Volume Resizing] "
            "Resizing volume to fixed (2x1x1) mm expected by the inpainuting model."
        )
        self.captureImage()
        volumesLogic = slicer.modules.volumes.logic()
        self.ResizedVolumeNode = volumesLogic.CloneVolume(
            slicer.mrmlScene, self.volume_node, "ResizedVolumeSlicer"
        )
        # Set parameters
        parameters = {
            "InputVolume": self.volume_node.GetID(),
            "referenceVolume": self.volume_node.GetID(),
            "OutputVolume": self.ResizedVolumeNode.GetID(),
            "outputPixelSpacing": FIXED_RESOLUTION,
            "interpolationMode": "linear",
        }
        # Execute
        resampleModule = slicer.modules.resamplescalarvolume
        cliNode = slicer.cli.runSync(resampleModule, None, parameters)
        # Process results
        if cliNode.GetStatus() & cliNode.ErrorsMask:
            # error
            errorText = cliNode.GetErrorText()
            slicer.mrmlScene.RemoveNode(cliNode)
            raise ValueError("CLI execution failed: " + errorText)
        # success
        slicer.mrmlScene.RemoveNode(cliNode)
        self.captureImage(resized=True)

    def updateImage(self, new_image, volume_node):
        self.image_data[:, :, :] = new_image
        slicer.util.arrayFromVolumeModified(volume_node)


#
# LeFusionTest
#


class LeFusionTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_LeFusion()

    def test_LeFusion(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        self.delayDisplay("Test passed")
