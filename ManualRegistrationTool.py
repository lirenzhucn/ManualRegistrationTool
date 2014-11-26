#!/usr/bin/env python

# my own library
from imageproc.ImageSliceDisplay import MinMaxDialog, ImageStat
from PyQt4.QtGui import QApplication, QFont
from PyQt4.QtGui import QWidget, QFrame, QGroupBox, QTabWidget
from PyQt4.QtGui import QPushButton, QComboBox, QLabel
from PyQt4.QtGui import QDoubleSpinBox, QTextEdit
from PyQt4.QtGui import QHBoxLayout, QVBoxLayout
from PyQt4.QtGui import QFileDialog, QSizePolicy
from PyQt4.QtCore import pyqtSlot, SIGNAL

import vtk
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

import skimage.io._plugins.freeimage_plugin as fi
import numpy as np
import sys

from os.path import expanduser

from scipy.ndimage.interpolation import affine_transform


_hot_data = (
    (0.0, 0.0416, 0.0, 0.0),
    (0.365079, 1.0, 0.0, 0.0),
    (0.746032, 1.0, 1.0, 0.0),
    (1.0, 1.0, 1.0, 1.0, 1.0)
)

_gray_data = (
    (0.0, 0.0, 0.0, 0.0),
    (1.0, 1.0, 1.0, 1.0)
)

DATA_FOLDER = expanduser('~/Documents/Project_data')


class VolumeRenderingManager:

    def __init__(self, cmData, reslice=None):
        self.cmData = cmData
        self._min = 0
        self._max = 255
        self.dataImport = vtk.vtkImageImport()
        self.dataImport.SetDataScalarTypeToUnsignedChar()
        self.dataImport.SetNumberOfScalarComponents(1)
        self.alphaFunc = vtk.vtkPiecewiseFunction()
        self.colorFunc = vtk.vtkColorTransferFunction()
        self.setupColorFunc()
        self.prop = vtk.vtkVolumeProperty()
        self.prop.SetColor(self.colorFunc)
        self.prop.SetScalarOpacity(self.alphaFunc)
        self.mapper = vtk.vtkSmartVolumeMapper()
        self.mapper.SetBlendModeToMaximumIntensity()
        if reslice is not None:
            reslice.SetInputConnection(self.dataImport.GetOutputPort())
            self.mapper.SetInputConnection(reslice.GetOutputPort())
        else:
            self.mapper.SetInputConnection(self.dataImport.GetOutputPort())
        self.volume = vtk.vtkVolume()
        self.volume.SetMapper(self.mapper)
        self.volume.SetProperty(self.prop)

    def setInput(self, imgNpy):
        assert(imgNpy.dtype == np.uint8)
        nx, ny, nz = imgNpy.shape
        dataString = imgNpy.tostring(order='F')
        self.dataImport.CopyImportVoidPointer(dataString,
                                              len(dataString))
        self.dataImport.SetDataExtent(0, nx-1, 0, ny-1, 0, nz-1)
        self.dataImport.SetWholeExtent(0, nx-1, 0, ny-1, 0, nz-1)
        self.dataImport.Update()

    def setMinMax(self, _min, _max):
        assert(_max >= _min)
        self._max = _max
        self._min = _min
        self.setupColorFunc()

    def update(self):
        self.volume.Update()

    def setupColorFunc(self):
        scale = self._max - self._min
        shift = self._min
        self.colorFunc.RemoveAllPoints()
        for idx in xrange(len(self.cmData)):
            record = self.cmData[idx]
            pointIdx = int(record[0]*scale + shift)
            redVal = record[1]
            greenVal = record[2]
            blueVal = record[3]
            self.colorFunc.AddRGBPoint(pointIdx, redVal,
                                       greenVal, blueVal)
        # setup alpha function
        self.alphaFunc.RemoveAllPoints()
        self.alphaFunc.AddPoint(self._min, 0.0)
        self.alphaFunc.AddPoint(self._max, 1.0)


class VtkFgBgDisplayWidget(QFrame):

    def __init__(self, parent=None):
        super(VtkFgBgDisplayWidget, self).__init__(parent)
        self.imVtkBg = None
        self.imVtkFg = None
        self.matrix4x4 = vtk.vtkMatrix4x4()
        self.matrix4x4.DeepCopy((
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ))
        self.reslice = vtk.vtkImageReslice()
        self.reslice.SetOutputDimensionality(3)
        self.reslice.SetResliceAxes(self.matrix4x4)
        self.reslice.SetInterpolationModeToLinear()
        self.setupVtkRendering()

    def setupVtkRendering(self):
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.ren = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.ren.ResetCamera()
        self.ren.GetActiveCamera().ParallelProjectionOn()
        self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        # layout
        vlayout = QVBoxLayout()
        vlayout.addWidget(self.vtkWidget)
        self.setLayout(vlayout)

    def initialize(self):
        self.iren.Initialize()

    @staticmethod
    def convertNpyToVtk(imgNpy):
        assert(imgNpy.dtype == np.uint8)
        nx, ny, nz = imgNpy.shape
        dataImporter = vtk.vtkImageImport()
        dataString = imgNpy.tostring(order='F')
        dataImporter.CopyImportVoidPointer(dataString, len(dataString))
        dataImporter.SetDataScalarTypeToUnsignedChar()
        dataImporter.SetNumberOfScalarComponents(1)
        dataImporter.SetDataExtent(0, nx-1, 0, ny-1, 0, nz-1)
        dataImporter.SetWholeExtent(0, nx-1, 0, ny-1, 0, nz-1)
        dataImporter.Update()
        return dataImporter.GetOutput()

    def setMatrix(self, matrix):
        if matrix is not None:
            assert(matrix.shape == (4, 4))
        self.setMatrixTuple(tuple(matrix.flatten()))

    def setMatrixTuple(self, matrix):
        if matrix is not None:
            assert(len(matrix) == 16)
        self.matrix4x4.DeepCopy(matrix)

    def render(self):
        # self.ren.Render()
        self.ren.ResetCamera()

    def setDirection(self, direction):
        if direction == 0:
            # X-Y plane
            self.ren.GetActiveCamera().SetPosition(0, 0, -1)
            self.ren.GetActiveCamera().SetFocalPoint(0, 0, 0)
            self.ren.GetActiveCamera().SetViewUp(-1, 0, 0)
        elif direction == 1:
            # X-Z plane
            self.ren.GetActiveCamera().SetPosition(0, -1, 0)
            self.ren.GetActiveCamera().SetFocalPoint(0, 0, 0)
            self.ren.GetActiveCamera().SetViewUp(0, 0, -1)
        elif direction == 2:
            # Y-Z plane
            self.ren.GetActiveCamera().SetPosition(-1, 0, 0)
            self.ren.GetActiveCamera().SetFocalPoint(0, 0, 0)
            self.ren.GetActiveCamera().SetViewUp(0, 0, -1)

    def update(self):
        self.vtkWidget.update()
        super(VtkFgBgDisplayWidget, self).update()


class OverlayDisplayWidget(VtkFgBgDisplayWidget):

    def __init__(self, parent=None):
        super(OverlayDisplayWidget, self).__init__(parent)
        self.mgrBg = None
        self.mgrFg = None

    def setVolumeBg(self, imBg, _min, _max):
        if imBg is None:
            return
        else:
            assert(imBg.dtype == np.uint8)
        if self.mgrBg is None:
            self.mgrBg = VolumeRenderingManager(_gray_data)
            self.ren.AddVolume(self.mgrBg.volume)
        self.mgrBg.setInput(imBg)
        self.mgrBg.setMinMax(_min, _max)
        self.mgrBg.update()

    def setVolumeFg(self, imFg, _min, _max):
        if imFg is None:
            return
        else:
            assert(imFg.dtype == np.uint8)
        if self.mgrFg is None:
            self.mgrFg = VolumeRenderingManager(_hot_data, self.reslice)
            self.ren.AddVolume(self.mgrFg.volume)
        self.mgrFg.setInput(imFg)
        self.mgrFg.setMinMax(_min, _max)
        self.mgrFg.update()

    def setBgMinMax(self, _min, _max):
        self.mgrBg.setMinMax(_min, _max)
        self.mgrBg.update()

    def setFgMinMax(self, _min, _max):
        self.mgrFg.setMinMax(_min, _max)
        self.mgrFg.update()


class MultiplyDisplayWidget(VtkFgBgDisplayWidget):

    def __init__(self, parent=None):
        super(MultiplyDisplayWidget, self).__init__(parent)
        self.imBg = None
        self.imFg = None
        self.imFg_t = None
        self.imOut = None
        self.matrix = np.array(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]]
        )
        self.mgr = None

    def setupVolumes(self):
        if (self.imBg is not None) and (self.imFg is not None):
            self.mgr = VolumeRenderingManager(_gray_data)
            self.ren.AddVolume(self.mgr.volume)
            print 'Transforming volume...'
            self.imFg_t =\
                affine_transform(self.imFg,
                                 matrix=self.matrix[0:3, 0:3],
                                 offset=self.matrix[0:3, 3],
                                 output_shape=self.imBg.shape)
            print 'Done'
            imOut = self.imBg * self.imFg_t
            imOut[imOut < 0.0] = 0.0
            imOut[imOut > 1.0] = 1.0
            self.mgr.setInput((imOut * 255.0).astype(np.uint8))
            self.mgr.update()

    def setVolumeBg(self, imBg):
        if imBg is not None:
            assert(imBg.dtype == np.uint8)
            imBg = imBg.astype(np.float)
            imBg = (imBg - np.amin(imBg)) / (np.amax(imBg) - np.amin(imBg))
            self.imBg = imBg
            self.setupVolumes()

    def setVolumeFg(self, imFg):
        if imFg is not None:
            assert(imFg.dtype == np.uint8)
            imFg = imFg.astype(np.float)
            imFg = (imFg - np.amin(imFg)) / (np.amax(imFg) - np.amin(imFg))
            self.imFg = imFg
            self.setupVolumes()

    def setMatrix(self, matrix):
        assert(matrix.shape == (4, 4))
        self.matrix = matrix
        self.setupVolumes()


class ManualRegistrationWidget(QWidget):
    '''ManualRegistrationWidget
    A tool for interactive image registration
    '''

    DEFAULT_FONT = QFont('Inconsolata', 12)
    STATUS_FONT = QFont('Inconsolata', 10)

    def __init__(self, parent=None):
        super(ManualRegistrationWidget, self).__init__(parent)
        self.setupUi()
        self.setupConnection()
        self.imFixed = None
        self.imMoving = None
        self.statsFixed = None
        self.statsMoving = None
        self.minFixed = 0
        self.maxFixed = 255
        self.minMoving = 0
        self.maxMoving = 255
        self.transformMatrix = np.array((
            (0.0, 0.0, -1.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0)
        ))
        self.updateMatrixText()

    def setupConnection(self):
        self.buttonLoadFixed.clicked.connect(self.onLoadFixed)
        self.buttonLoadMoving.clicked.connect(self.onLoadMoving)
        self.comboAxes.currentIndexChanged.connect(self.directionChange)
        self.buttonClear.clicked.connect(self.onClear)
        self.buttonApply.clicked.connect(self.onApply)
        self.buttonMinMaxFixed.clicked.connect(self.onMinMaxFixed)
        self.buttonMinMaxMoving.clicked.connect(self.onMinMaxMoving)
        self.buttonLoadMatrix.clicked.connect(self.onLoadMatrix)
        self.buttonSaveMatrix.clicked.connect(self.onSaveMatrix)

    def setupUi(self):
        '''Setup GUI components'''
        # set font
        self.setFont(self.DEFAULT_FONT)
        self.setupControls()
        self.overlayDisplay = OverlayDisplayWidget(parent=self)
        self.overlayDisplay.setMinimumSize(500, 500)
        self.overlayDisplay.setSizePolicy(QSizePolicy.Expanding,
                                          QSizePolicy.Expanding)
        self.multiplyDisplay = MultiplyDisplayWidget(parent=self)
        self.multiplyDisplay.setMinimumSize(500, 500)
        self.multiplyDisplay.setSizePolicy(QSizePolicy.Expanding,
                                           QSizePolicy.Expanding)
        self.displayTab = QTabWidget(self)
        self.displayTab.addTab(self.overlayDisplay, 'Overlay Display')
        self.displayTab.addTab(self.multiplyDisplay, 'Multiply Display')
        # layout
        hlayout = QHBoxLayout()
        hlayout.addWidget(self.controlPanel)
        hlayout.addWidget(self.displayTab)
        self.setLayout(hlayout)

    def setupControls(self):
        self.controlPanel = QFrame()
        self.controlPanel.setMinimumSize(400, 500)
        #
        # create controls
        # first row: buttons and axes selection
        self.buttonLoadFixed = QPushButton('Load Fixed')
        self.buttonLoadMoving = QPushButton('Load Moving')
        self.comboAxes = QComboBox()
        self.comboAxes.addItems(['X-Y', 'X-Z', 'Y-Z'])
        self.buttonMinMaxFixed = QPushButton(r'B&&C Fixed')
        self.buttonMinMaxMoving = QPushButton(r'B&&C Moving')
        self.buttonMinMaxFixed.setEnabled(False)
        self.buttonMinMaxMoving.setEnabled(False)
        # second row: Transformation box
        groupTransform = QGroupBox('Transformation')
        label1 = QLabel('X Angle')
        label2 = QLabel('Y Angle')
        label3 = QLabel('Z Angle')
        self.spinXAngle = QDoubleSpinBox()
        self.spinXAngle.setRange(-90.0, 90.0)
        self.spinYAngle = QDoubleSpinBox()
        self.spinYAngle.setRange(-90.0, 90.0)
        self.spinZAngle = QDoubleSpinBox()
        self.spinZAngle.setRange(-90.0, 90.0)
        label4 = QLabel('X Offset')
        label5 = QLabel('Y Offset')
        label6 = QLabel('Z Offset')
        self.spinXOffset = QDoubleSpinBox()
        self.spinXOffset.setRange(-1000.0, 1000.0)
        self.spinYOffset = QDoubleSpinBox()
        self.spinYOffset.setRange(-1000.0, 1000.0)
        self.spinZOffset = QDoubleSpinBox()
        self.spinZOffset.setRange(-1000.0, 1000.0)
        self.buttonClear = QPushButton('Clear')
        self.buttonApply = QPushButton('Apply')
        # third row: Output matrix
        labelMatrix = QLabel('Transformation matrix')
        self.editMatrix = QTextEdit()
        self.editMatrix.setReadOnly(True)
        self.buttonLoadMatrix = QPushButton('Load Matrix')
        self.buttonSaveMatrix = QPushButton('Save Matrix')
        #
        # layout controls
        # 1st row
        hlayoutButtons1 = QHBoxLayout()
        hlayoutButtons1.addWidget(self.buttonLoadFixed)
        hlayoutButtons1.addWidget(self.buttonLoadMoving)
        hlayoutButtons1.addWidget(self.comboAxes)
        hlayoutButtons2 = QHBoxLayout()
        hlayoutButtons2.addWidget(self.buttonMinMaxFixed)
        hlayoutButtons2.addWidget(self.buttonMinMaxMoving)
        # 2nd row
        hlayoutTransform1 = QHBoxLayout()
        hlayoutTransform1.addWidget(label1)
        hlayoutTransform1.addWidget(self.spinXAngle)
        hlayoutTransform1.addWidget(label4)
        hlayoutTransform1.addWidget(self.spinXOffset)
        hlayoutTransform2 = QHBoxLayout()
        hlayoutTransform2.addWidget(label2)
        hlayoutTransform2.addWidget(self.spinYAngle)
        hlayoutTransform2.addWidget(label5)
        hlayoutTransform2.addWidget(self.spinYOffset)
        hlayoutTransform3 = QHBoxLayout()
        hlayoutTransform3.addWidget(label3)
        hlayoutTransform3.addWidget(self.spinZAngle)
        hlayoutTransform3.addWidget(label6)
        hlayoutTransform3.addWidget(self.spinZOffset)
        hlayoutTransform4 = QHBoxLayout()
        hlayoutTransform4.addStretch()
        hlayoutTransform4.addWidget(self.buttonClear)
        hlayoutTransform4.addWidget(self.buttonApply)
        vlayoutTransform = QVBoxLayout()
        vlayoutTransform.addLayout(hlayoutTransform1)
        vlayoutTransform.addLayout(hlayoutTransform2)
        vlayoutTransform.addLayout(hlayoutTransform3)
        vlayoutTransform.addLayout(hlayoutTransform4)
        groupTransform.setLayout(vlayoutTransform)
        # 3rd row
        hlayout1 = QHBoxLayout()
        hlayout1.addStretch()
        hlayout1.addWidget(self.buttonLoadMatrix)
        hlayout1.addWidget(self.buttonSaveMatrix)
        # final layout
        vlayout = QVBoxLayout()
        vlayout.addLayout(hlayoutButtons1)
        vlayout.addLayout(hlayoutButtons2)
        vlayout.addWidget(groupTransform)
        vlayout.addWidget(labelMatrix)
        vlayout.addWidget(self.editMatrix)
        vlayout.addLayout(hlayout1)
        self.controlPanel.setLayout(vlayout)
        self.controlPanel.setSizePolicy(QSizePolicy.Fixed,
                                        QSizePolicy.Expanding)

    def initializeInteractor(self):
        self.overlayDisplay.initialize()
        self.multiplyDisplay.initialize()

    @pyqtSlot(int)
    def directionChange(self, direction):
        self.overlayDisplay.setDirection(direction)
        self.overlayDisplay.render()
        self.overlayDisplay.update()

    @pyqtSlot()
    def onLoadFixed(self):
        fname = QFileDialog.getOpenFileName(self, 'Load fixed image',
                                            DATA_FOLDER,
                                            filter='TIFF Images (*.tiff *.tif)')
        fname = str(fname)
        if not fname:
            return
        self.imFixed = self.normalizeImage(self.load3DTIFFImage(fname))
        self.statsFixed = ImageStat(self.imFixed)
        self.minFixed, self.maxFixed = self.statsFixed.extrema
        self.overlayDisplay.setVolumeBg(self.imFixed,
                                        self.minFixed, self.maxFixed)
        self.overlayDisplay.render()
        self.buttonMinMaxFixed.setEnabled(True)
        self.multiplyDisplay.setVolumeBg(self.imFixed)
        self.multiplyDisplay.render()

    @pyqtSlot()
    def onLoadMoving(self):
        fname = QFileDialog.getOpenFileName(self, 'Load fixed image',
                                            DATA_FOLDER,
                                            filter='TIFF Images (*.tiff *.tif)')
        fname = str(fname)
        if not fname:
            return
        self.imMoving = self.normalizeImage(self.load3DTIFFImage(fname))
        self.statsMoving = ImageStat(self.imMoving)
        self.minMoving, self.maxMoving = self.statsMoving.extrema
        self.overlayDisplay.setVolumeFg(self.imMoving,
                                        self.minMoving, self.maxMoving)
        self.overlayDisplay.setMatrix(self.transformMatrix)
        self.overlayDisplay.render()
        self.buttonMinMaxMoving.setEnabled(True)
        self.multiplyDisplay.setVolumeFg(self.imMoving)
        self.multiplyDisplay.render()

    @pyqtSlot()
    def onClear(self):
        self.spinXAngle.setValue(0.0)
        self.spinXOffset.setValue(0.0)
        self.spinYAngle.setValue(0.0)
        self.spinYOffset.setValue(0.0)
        self.spinZAngle.setValue(0.0)
        self.spinZOffset.setValue(0.0)

    @pyqtSlot()
    def onApply(self):
        xAngle = self.spinXAngle.value() / 180.0 * np.pi
        xOffset = self.spinXOffset.value()
        yAngle = self.spinYAngle.value() / 180.0 * np.pi
        yOffset = self.spinYOffset.value()
        zAngle = self.spinZAngle.value() / 180.0 * np.pi
        zOffset = self.spinZOffset.value()
        # form rotation matrices
        Rx = np.eye(4)
        Rx[1, 1] = np.cos(xAngle)
        Rx[2, 2] = np.cos(xAngle)
        Rx[1, 2] = -np.sin(xAngle)
        Rx[2, 1] = np.sin(xAngle)
        Ry = np.eye(4)
        Ry[0, 0] = np.cos(yAngle)
        Ry[2, 2] = np.cos(yAngle)
        Ry[0, 2] = np.sin(yAngle)
        Ry[2, 0] = -np.sin(yAngle)
        Rz = np.eye(4)
        Rz[0, 0] = np.cos(zAngle)
        Rz[1, 1] = np.cos(zAngle)
        Rz[0, 1] = -np.sin(zAngle)
        Rz[1, 0] = np.sin(zAngle)
        # final transform
        T = np.dot(Rz, np.dot(Ry, Rx))
        T[0, 3] = xOffset
        T[1, 3] = yOffset
        T[2, 3] = zOffset
        self.transformMatrix = np.dot(T, self.transformMatrix)
        self.updateMatrixText()
        self.overlayDisplay.setMatrix(self.transformMatrix)
        self.onClear()

    @pyqtSlot()
    def onMinMaxFixed(self):
        self.mmDialogFixed = MinMaxDialog(self.minFixed, self.maxFixed,
                                          self.statsFixed, self)
        self.connect(self.mmDialogFixed, SIGNAL('minMaxChanged()'),
                     self.minMaxChangedFixed)
        self.mmDialogFixed.exec_()
        self.disconnect(self.mmDialogFixed, SIGNAL('minMaxChanged()'),
                        self.minMaxChangedFixed)
        del self.mmDialogFixed
        self.mmDialogFixed = None

    @pyqtSlot()
    def onMinMaxMoving(self):
        self.mmDialogMoving = MinMaxDialog(self.minMoving, self.maxMoving,
                                           self.statsMoving, self)
        self.connect(self.mmDialogMoving, SIGNAL('minMaxChanged()'),
                     self.minMaxChangedMoving)
        self.mmDialogMoving.exec_()
        self.disconnect(self.mmDialogMoving, SIGNAL('minMaxChanged()'),
                        self.minMaxChangedMoving)
        del self.mmDialogMoving
        self.mmDialogMoving = None

    @pyqtSlot()
    def onSaveMatrix(self):
        fname = QFileDialog.getSaveFileName(self, 'Save matrix to npy file',
                                            './transform_matrix.npy',
                                            filter='Numpy Array Files (*.npy)')
        fname = str(fname)
        if fname:
            np.save(fname, self.transformMatrix)

    @pyqtSlot()
    def onLoadMatrix(self):
        fname = QFileDialog.getOpenFileName(self, 'Open matrix from npy file',
                                            './transform_matrix.npy',
                                            filter='Numpy Array Files (*.npy)')
        fname = str(fname)
        if fname:
            self.transformMatrix = np.load(fname)
        self.updateMatrixText()
        self.overlayDisplay.setMatrix(self.transformMatrix)
        self.multiplyDisplay.setMatrix(self.transformMatrix)

    @pyqtSlot()
    def minMaxChangedFixed(self):
        self.minFixed, self.maxFixed = self.mmDialogFixed.results
        self.overlayDisplay.setBgMinMax(self.minFixed, self.maxFixed)
        self.overlayDisplay.update()

    @pyqtSlot()
    def minMaxChangedMoving(self):
        self.minMoving, self.maxMoving = self.mmDialogMoving.results
        self.overlayDisplay.setFgMinMax(self.minMoving, self.maxMoving)
        self.overlayDisplay.update()

    def updateMatrixText(self):
        msg = np.array_str(self.transformMatrix,
                           precision=2, suppress_small=True)
        self.editMatrix.setText(msg)

    @staticmethod
    def normalizeImage(im):
        scaledImage = (im - np.amin(im)) / (np.amax(im) - np.amin(im))
        return (scaledImage*255.0).astype(np.uint8)

    @staticmethod
    def load3DTIFFImage(fname):
        imList = fi.read_multipage(fname)
        if not imList:
            return None
        nx, ny = imList[0].shape
        nz = len(imList)
        im = np.zeros((nx, ny, nz))
        for idx in xrange(nz):
            im[:, :, idx] = imList[idx]
        return im.astype(np.float)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = ManualRegistrationWidget()
    w.show()
    w.initializeInteractor()
    sys.exit(app.exec_())
