from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog,
QDialogButtonBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
QLabel, QLineEdit, QMenu, QMenuBar, QPushButton, QSpinBox, QTextEdit,
QVBoxLayout,QMainWindow, QMessageBox, QAction,QSlider,QSizePolicy)
from PyQt5.QtCore import Qt
import sys
from PyQt5.QtGui import QIcon
import numpy as np
from keras.models import load_model
from matplotlib import colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import vtk
class Dialog(QDialog):
    NumGridRows = 3
    NumButtons = 4

    def __init__(self):
        """
         Initialize and set up the dialog. This is called by __init__ and should not be called directly
        """
        super(Dialog, self).__init__()
        self.createFormGroupBox()
        
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.forces)
        self.buttonBox.rejected.connect(self.reject)
        
        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.formGroupBox)
        self.mainLayout.addWidget(self.buttonBox)
        #self.mainlayout_save = self.mainLayout
        self.setLayout(self.mainLayout)
        
        self.setWindowTitle("Topology Optimization via TL")
        
    def createFormGroupBox(self):
        """
         Create group box to display the form. This is used in the GUI to add a list of models
        """
        self.formGroupBox = QGroupBox("Form layout")
        ####### Load the Machine learning models
        self.Simple_4080_model = load_model('Simple_4080.h5')
        self.Simple_80160_model = load_model('Simple_80160.h5')
        self.Simple_120160_model = load_model('Simple_120160.h5')
        self.Simple_120240_model = load_model('Simple_120240.h5')
        self.Simple_160320_model = load_model('Simple_160320.h5')
        self.Simple_200400_model = load_model('Simple_200400.h5')
        self.Curved_model = load_model('Curved.h5')
        self.Frame_model = load_model('Frame.h5')
        self.Lshape_model = load_model('Lshape.h5')
        self.Lshapewithhole_model = load_model('Lshape with hole.h5')
        self.Simple_408010_model = load_model('Simple_408010.h5')
        self.Simple_8016010_model = load_model('Simple_8016010.h5')
        self.With1hole_model = load_model('With 1 hole.h5')
        self.With2holes_model = load_model('With 2 holes.h5')
        self.Cube_model = load_model('Cube.h5')
        self.Cubewith2holes_model = load_model('Cube with 2 holes.h5')
        self.Domewith2holes_model = load_model('Dome with 2 holes.h5')
        layout = QFormLayout()
        ###### Add Methods
        self.Method = QComboBox()
        self.Method.addItem("TL")
        self.Method.addItem("TL + SIMP")
        layout.addRow(QLabel("Method:"), self.Method)
        ###### Add Dimensions
        self.Dimension = QComboBox()
        self.Dimension.addItem("2D")
        self.Dimension.addItem("3D")
        layout.addRow(QLabel("Dimension:"), self.Dimension)
        ###### Add Domain Topologies
        self.domain = QComboBox()
        layout.addRow(QLabel("Domain:"), self.domain)
        self.Dimension.activated[str].connect(self.domains)
        ###### Add Resolutions
        self.Resolution = QComboBox()
        layout.addRow(QLabel("Resolution:"), self.Resolution)
        self.domain.activated[str].connect(self.Resolutions)
        ###### Add Boundary Conditions
        self.Bounday_conditions = QComboBox()
        layout.addRow(QLabel("Bounday Conditions:"), self.Bounday_conditions)
        self.domain.activated[str].connect(self.Boundayconditions)
        self.formGroupBox.setLayout(layout)
        
    
    
    def domains(self, text):
        """
         Sets the domains to the given text. This is called when the user presses the domains button
         
         Args:
         	 text: Text that was pressed
        """
        # This method is used to set the domain of the current page
        if text =='2D':
            self.domain.clear()
            self.domain.addItem("Simple")
            self.domain.addItem("Lshape")
            self.domain.addItem("Lshape with hole")
            self.domain.addItem("Frame")
            self.domain.addItem("Curved")
        else:
            self.domain.clear()
            self.domain.addItem("Simple")
            self.domain.addItem("Simple with 1 hole")
            self.domain.addItem("Simple with 2 holes")
            self.domain.addItem("Cube")
            self.domain.addItem("Cube with 2 holes")
            self.domain.addItem("Dome with 2 holes")
            
    def Resolutions(self, text):
        """
        Sets the resolutions for the selected dimension. This is called when the user clicks on the Resolution combobox
        
        Args:
        text: The shape of the domain
        """   
        # The resolution of the current image.
        if text =='Simple' and self.Dimension.currentText() =='2D':
            self.Resolution.clear()
            self.Resolution.addItem("40 x 80")
            self.Resolution.addItem("80 x 160")
            self.Resolution.addItem("120 x 160")
            self.Resolution.addItem("120 x 240")
            self.Resolution.addItem("160 x 320")
            self.Resolution.addItem("200 x 400")
        elif text=='Lshape' and self.Dimension.currentText() =='2D':
            self.Resolution.clear()
            self.Resolution.addItem("120 x 240")
        elif text=='Lshape with hole' and self.Dimension.currentText() =='2D':
            self.Resolution.clear()
            self.Resolution.addItem("120 x 240")
        elif text=='Frame' or text =='Curved':
            self.Resolution.clear()
            self.Resolution.addItem("120 x 240")
            
        elif text =='Simple' and self.Dimension.currentText() =='3D':
            self.Resolution.clear()
            self.Resolution.addItem("40 x 80 x 10")
            self.Resolution.addItem("80 x 160 x 10")
        elif text=='Simple with 1 hole' or text =='Simple with 2 holes':
            self.Resolution.clear()
            self.Resolution.addItem("40 x 80 x 10")
        elif text == 'Dome with 2 holes' or text == 'Cube with 2 holes' or text == 'Cube':
            self.Resolution.clear()
            self.Resolution.addItem("40 x 40 x 40")
            
    def Boundayconditions(self,text):
        if self.domain.currentText() == 'Lshape' or self.domain.currentText() == 'Lshape with hole':
            self.Bounday_conditions.clear()
            self.Bounday_conditions.addItem('Cantilever')
            self.Bounday_conditions.addItem('Constrained Cantilever')
        elif self.domain.currentText() == 'Dome with 2 holes' or self.domain.currentText() == 'Cube with 2 holes' or self.domain.currentText() == 'Cube':
            self.Bounday_conditions.clear()
            self.Bounday_conditions.addItem('Four simple supports')
        else:
            self.Bounday_conditions.clear()
            self.Bounday_conditions.addItem('Cantilever')
            self.Bounday_conditions.addItem('Simply Supported')
            self.Bounday_conditions.addItem('Constrained Cantilever')
            
        
            
            
            
            
            
        

    def forces(self):
        # Create textbox
        self.buttonBox.setParent(None)
        self.counter = 1
        self.textbox_layout = QVBoxLayout()
        if self.Dimension.currentText() == '3D':
            ############ Initialize the plot
            colors = vtk.vtkNamedColors()
            self.points = vtk.vtkPoints()
            self.polydata = vtk.vtkPolyData()
            cubeSource = vtk.vtkCubeSource()
        
            self.glyph3D = vtk.vtkGlyph3D()
            self.glyph3D.SetSourceConnection(cubeSource.GetOutputPort())
            self.mapper = vtk.vtkPolyDataMapper()
            self.actor = vtk.vtkActor()
            self.actor2 = vtk.vtkActor()
            self.actor3 = vtk.vtkActor()
            self.actor4 = vtk.vtkActor()
            self.renderer = vtk.vtkRenderer()
            self.renderWindow = vtk.vtkRenderWindow()
            self.renderWindow.AddRenderer(self.renderer)
            self.renderWindow.SetSize(600,600)
            self.renderWindowInteractor = vtk.vtkRenderWindowInteractor()
            self.renderWindowInteractor.SetRenderWindow(self.renderWindow)
            
            self.actor.GetProperty().SetColor(colors.GetColor3d("BurlyWood"))
            self.renderer.AddActor(self.actor)
            self.renderer.AddActor(self.actor2)
            self.renderer.AddActor(self.actor3)
            self.renderer.AddActor(self.actor4)
            self.renderer.SetBackground(colors.GetColor3d("SlateGray")) # Background Slate Gray
            
            
            ####### ADD Widget
            self.mainlayout_save = QVBoxLayout()
            
            self.mainlayout_save = self.mainLayout
            #self.Force_x = QLineEdit(self,placeholderText="Force in x direction [-100,100].")
            self.Forcex_label = QLabel("Force in x direction [-100,100]")
            self.Force_x = QSlider(Qt.Horizontal, self)
            self.Force_x.setMinimum(-100)
            self.Force_x.setMaximum(100)
            self.Force_x.setPageStep(1)
            self.Force_x.setValue(50)
            #self.Force_y = QLineEdit(self,placeholderText="Force in y direction [-100,100].")
            self.Forcey_label = QLabel("Force in y direction [-100,100]")
            self.Force_y = QSlider(Qt.Horizontal, self)
            self.Force_y.setMinimum(-100)
            self.Force_y.setMaximum(100)
            self.Force_y.setPageStep(1)
            self.Force_y.setValue(-50)
            #self.Force_z = QLineEdit(self,placeholderText="Force in z direction [-100,100].")
            self.Forcez_label = QLabel("Force in z direction [-100,100]")
            self.Force_z = QSlider(Qt.Horizontal, self)
            self.Force_z.setMinimum(-100)
            self.Force_z.setMaximum(100)
            self.Force_z.setPageStep(1)
            self.Force_z.setValue(50)
            ####### Slider for Positions
            self.nely,self.nelx,self.nelz = self.Extract_Resolusion(self.Resolution.currentText(),3)
            if self.domain.currentText() == 'Simple':
                if self.Bounday_conditions.currentText() == 'Cantilever':
                #self.Position_x = QLineEdit(self,placeholderText="Position in x direction.")
                    self.Position_x = QSlider(Qt.Horizontal, self)
                    self.Position_x.setMinimum(10)
                    self.Position_x.setMaximum(self.nelx)
                    self.Position_x.setPageStep(1)
                    self.Position_x.setValue(self.nelx-1)
                    self.Positionx_label = QLabel("Position in x direction [10,"+str(self.nelx)+"]")
                    #self.Position_y = QLineEdit(self,placeholderText="Position in y direction.")
                    self.Position_y = QSlider(Qt.Horizontal, self)
                    self.Position_y.setMinimum(1)
                    self.Position_y.setMaximum(self.nely)
                    self.Position_y.setPageStep(1) 
                    self.Position_y.setValue(self.nely-1)
                    self.Positiony_label = QLabel("Position in y direction [1,"+str(self.nely)+"]")
                    #self.Position_z = QLineEdit(self,placeholderText="Position in z direction.")
                    self.Position_z = QSlider(Qt.Horizontal, self)
                    self.Position_z.setMinimum(1)
                    self.Position_z.setMaximum(self.nelz)
                    self.Position_z.setPageStep(1)
                    self.Position_z.setValue(self.nelz-1)
                    self.Positionz_label = QLabel("Position in z direction [1,"+str(self.nelz)+"]")
                elif self.Bounday_conditions.currentText() == 'Simply Supported':
                    self.Position_x = QSlider(Qt.Horizontal, self)
                    self.Position_x.setMinimum(10)
                    self.Position_x.setMaximum(self.nelx-10)
                    self.Position_x.setPageStep(1)
                    self.Position_x.setValue(self.nelx-15)
                    self.Positionx_label = QLabel("Position in x direction [10,"+str(self.nelx-10)+"]")
                    self.Position_y = QSlider(Qt.Horizontal, self)
                    self.Position_y.setMinimum(int(np.round(self.nely/2)))
                    self.Position_y.setMaximum(self.nely)
                    self.Position_y.setValue(self.nely-1)
                    self.Position_y.setPageStep(1) 
                    self.Positiony_label = QLabel("Position in y direction [,"+str(np.round(self.nely/2))+str(self.nely)+"]")
                    self.Position_z = QSlider(Qt.Horizontal, self)
                    self.Position_z.setMinimum(1)
                    self.Position_z.setMaximum(self.nelz)
                    self.Position_z.setValue(self.nelz-1)
                    self.Position_z.setPageStep(1) 
                    self.Positionz_label = QLabel("Position in z direction [1,"+str(self.nelz)+"]")
                else:
                    self.Position_x = QSlider(Qt.Horizontal, self)
                    self.Position_x.setMinimum(1)
                    self.Position_x.setMaximum(np.round(self.nelx/2))
                    self.Position_x.setPageStep(1)
                    self.Position_x.setValue(np.round(self.nelx/2)-1)
                    self.Positionx_label = QLabel("Position in x direction [1,"+str(np.round(self.nelx/2))+"]")
                    self.Position_y = QSlider(Qt.Horizontal, self)
                    self.Position_y.setMinimum(int(np.round(self.nely/2)))
                    self.Position_y.setMaximum(self.nely)
                    self.Position_y.setValue(self.nely-1)
                    self.Position_y.setPageStep(1) 
                    self.Positiony_label = QLabel("Position in y direction [,"+str(np.round(self.nely/2))+","+str(self.nely)+"]")
                    self.Position_z = QSlider(Qt.Horizontal, self)
                    self.Position_z.setMinimum(1)
                    self.Position_z.setMaximum(self.nelz)
                    self.Position_z.setValue(self.nelz-1)
                    self.Position_z.setPageStep(1) 
                    self.Positionz_label = QLabel("Position in z direction [1,"+str(self.nelz)+"]")
            elif self.domain.currentText() == 'Cube':
                self.Position_x = QSlider(Qt.Horizontal, self)
                self.Position_x.setMinimum(int(np.round(self.nelx/4)))
                self.Position_x.setMaximum(int(np.round(3*self.nelx/4)))
                self.Position_x.setValue(int(np.round(3*self.nelx/4)-1))
                self.Position_x.setPageStep(1)  
                self.Positionx_label = QLabel("Position in x direction ["+str(np.round(self.nelx/4))+","+str(np.round(3*self.nelx/4))+"]")
                self.Position_y = QSlider(Qt.Horizontal, self)
                self.Position_y.setMinimum(int(np.round(self.nely/2)))
                self.Position_y.setMaximum(self.nely)
                self.Position_y.setValue(self.nely-1)
                self.Position_y.setPageStep(1) 
                self.Positiony_label = QLabel("Position in y direction ["+str(np.round(self.nely/2))+","+str(self.nely)+"]")
                self.Position_z = QSlider(Qt.Horizontal, self)
                self.Position_z.setMinimum(int(np.round(self.nelz/4)))
                self.Position_z.setMaximum(int(np.round(3*self.nelz/4)))
                self.Position_z.setValue(int(np.round(3*self.nelz/4)-1))
                self.Position_z.setPageStep(1) 
                self.Positionz_label = QLabel("Position in z direction ["+str(np.round(self.nelz/4))+","+str(np.round(3*self.nelz/4))+"]") 
            elif self.domain.currentText() == 'Simple with 1 hole':
                if self.Bounday_conditions.currentText() == 'Cantilever':
                #self.Position_x = QLineEdit(self,placeholderText="Position in x direction.")
                    self.Position_x = QSlider(Qt.Horizontal, self)
                    self.Position_x.setMinimum(self.nelx/2)
                    self.Position_x.setMaximum(self.nelx)
                    self.Position_x.setValue(self.nelx-1)
                    self.Position_x.setPageStep(1)  
                    self.Positionx_label = QLabel("Position in x direction ["+str(self.nelx/2)+","+str(self.nelx)+"]")
                    #self.Position_y = QLineEdit(self,placeholderText="Position in y direction.")
                    self.Position_y = QSlider(Qt.Horizontal, self)
                    self.Position_y.setMinimum(1)
                    self.Position_y.setMaximum(self.nely)
                    self.Position_y.setValue(self.nely-1)
                    self.Position_y.setPageStep(1) 
                    self.Positiony_label = QLabel("Position in y direction [1,"+str(self.nely)+"]")
                    #self.Position_z = QLineEdit(self,placeholderText="Position in z direction.")
                    self.Position_z = QSlider(Qt.Horizontal, self)
                    self.Position_z.setMinimum(1)
                    self.Position_z.setMaximum(self.nelz)
                    self.Position_z.setValue(self.nelz-1)
                    self.Position_z.setPageStep(1) 
                    self.Positionz_label = QLabel("Position in z direction [1,"+str(self.nelz)+"]")
                elif self.Bounday_conditions.currentText() == 'Simply Supported':
                    self.Position_x = QSlider(Qt.Horizontal, self)
                    self.Position_x.setMinimum(int(self.nelx/2))
                    self.Position_x.setMaximum(self.nelx-10)
                    self.Position_x.setPageStep(1)
                    self.Position_x.setValue(self.nelx-15)
                    self.Positionx_label = QLabel("Position in x direction ["+str(self.nelx/2)+","+str(self.nelx-10)+"]")
                    self.Position_y = QSlider(Qt.Horizontal, self)
                    self.Position_y.setMinimum(int(np.round(self.nely/2)))
                    self.Position_y.setMaximum(self.nely)
                    self.Position_y.setPageStep(1)
                    self.Position_y.setValue(self.nely-1)
                    self.Positiony_label = QLabel("Position in y direction [,"+str(np.round(self.nely/2))+str(self.nely)+"]")
                    self.Position_z = QSlider(Qt.Horizontal, self)
                    self.Position_z.setMinimum(1)
                    self.Position_z.setMaximum(self.nelz)
                    self.Position_z.setValue(self.nelz-1)
                    self.Position_z.setPageStep(1) 
                    self.Positionz_label = QLabel("Position in z direction [1,"+str(self.nelz)+"]")
                else:
                    self.Forcey_label = QLabel("Force in y direction [-100,-1]")
                    self.Force_y = QSlider(Qt.Horizontal, self)
                    self.Force_y.setMinimum(-100)
                    self.Force_y.setMaximum(-1)
                    self.Force_y.setValue(-50)
                    self.Force_y.setPageStep(1)
                    self.Position_x = QSlider(Qt.Horizontal, self)
                    self.Position_x.setMinimum(int(self.nelx/2))
                    self.Position_x.setMaximum(self.nelx-10)
                    self.Position_x.setPageStep(1)
                    self.Position_x.setValue(self.nelx-15)
                    self.Positionx_label = QLabel("Position in x direction ["+str(self.nelx/2)+","+str(self.nelx-10)+"]")
                    self.Position_y = QSlider(Qt.Horizontal, self)
                    self.Position_y.setMinimum(int(np.round(self.nely/2)))
                    self.Position_y.setMaximum(self.nely)
                    self.Position_y.setPageStep(1)
                    self.Position_y.setValue(self.nely-1)
                    self.Positiony_label = QLabel("Position in y direction [,"+str(np.round(self.nely/2))+","+str(self.nely)+"]")
                    self.Position_z = QSlider(Qt.Horizontal, self)
                    self.Position_z.setMinimum(1)
                    self.Position_z.setMaximum(self.nelz)
                    self.Position_z.setPageStep(1)
                    self.Position_z.setValue(self.nelz-1)
                    self.Positionz_label = QLabel("Position in z direction [1,"+str(self.nelz)+"]")
                    
            elif self.domain.currentText() == 'Simple with 2 holes':
                if self.Bounday_conditions.currentText() == 'Cantilever':
                #self.Position_x = QLineEdit(self,placeholderText="Position in x direction.")
                    self.Position_x = QSlider(Qt.Horizontal, self)
                    self.Position_x.setMinimum(73)
                    self.Position_x.setMaximum(self.nelx)
                    self.Position_x.setPageStep(1)  
                    self.Position_x.setValue(self.nelx-1)
                    self.Positionx_label = QLabel("Position in x direction [73,"+str(self.nelx)+"]")
                    #self.Position_y = QLineEdit(self,placeholderText="Position in y direction.")
                    self.Position_y = QSlider(Qt.Horizontal, self)
                    self.Position_y.setMinimum(1)
                    self.Position_y.setMaximum(self.nely)
                    self.Position_y.setPageStep(1) 
                    self.Position_y.setValue(self.nely-1)
                    self.Positiony_label = QLabel("Position in y direction [1,"+str(self.nely)+"]")
                    #self.Position_z = QLineEdit(self,placeholderText="Position in z direction.")
                    self.Position_z = QSlider(Qt.Horizontal, self)
                    self.Position_z.setMinimum(1)
                    self.Position_z.setMaximum(self.nelz)
                    self.Position_z.setPageStep(1)
                    self.Position_z.setValue(self.nelz-1)
                    self.Positionz_label = QLabel("Position in z direction [1,"+str(self.nelz)+"]")
                else:
                    self.Position_x = QSlider(Qt.Horizontal, self)
                    self.Position_x.setMinimum(int(np.round(self.nelx/3 + 1)))
                    self.Position_x.setMaximum(int(np.round(2*self.nelx/3 - 1)))
                    self.Position_x.setPageStep(1) 
                    self.Position_x.setValue(int(np.round(2*self.nelx/3 - 1)-1))
                    self.Positionx_label = QLabel("Position in x direction ["+str(np.round(self.nelx/3 + 1))+","+str(np.round(2*self.nelx/3 - 1))+"]")
                    self.Position_y = QSlider(Qt.Horizontal, self)
                    self.Position_y.setMinimum(int(np.round(self.nely/2)))
                    self.Position_y.setMaximum(self.nely)
                    self.Position_y.setPageStep(1)
                    self.Position_y.setValue(self.nely-1)
                    self.Positiony_label = QLabel("Position in y direction [,"+str(np.round(self.nely/2))+str(self.nely)+"]")
                    self.Position_z = QSlider(Qt.Horizontal, self)
                    self.Position_z.setMinimum(1)
                    self.Position_z.setMaximum(self.nelz)
                    self.Position_z.setPageStep(1)
                    self.Position_z.setValue(self.nelz-1)
                    self.Positionz_label = QLabel("Position in z direction [1,"+str(self.nelz)+"]")
            elif self.domain.currentText() == 'Dome with 2 holes':
                self.Position_x = QSlider(Qt.Horizontal, self)
                #self.Position_x.setMinimum(np.round(self.nelx/4))
                #self.Position_x.setMaximum(np.round(3*self.nelx/4))
                #self.Position_x.setPageStep(1)  
                self.Positionx_label = QLabel("")
                self.Position_y = QSlider(Qt.Horizontal, self)
                self.Position_y.setMinimum(30)
                self.Position_y.setMaximum(self.nely)
                self.Position_y.setValue(self.nely-1)
                self.Position_y.setPageStep(1) 
                self.Positiony_label = QLabel("Position in y direction [30,"+str(self.nely)+"] - SHOULD BE DETERMINED FIRST!")
                self.Position_y.valueChanged[int].connect(self.Determine_positionx_for_DOME)
                self.Position_z = QSlider(Qt.Horizontal, self)
                self.Position_z.setMinimum(20)
                self.Position_z.setMaximum(20)
                self.Position_z.setPageStep(1)
                self.Position_z.setValue(20)
                self.Positionz_label = QLabel("Position in z direction = 20")
            elif self.domain.currentText() == 'Cube with 2 holes':
                self.Position_x = QSlider(Qt.Horizontal, self)
                self.Position_x.setMinimum(int(np.round(self.nelx/4)))
                self.Position_x.setMaximum(int(np.round(3*self.nelx/4)))
                self.Position_x.setPageStep(1)
                self.Position_x.setValue(int(np.round(3*self.nelx/4)-1))
                self.Positionx_label = QLabel("Position in x direction ["+str(np.round(self.nelx/4))+","+str(np.round(3*self.nelx/4))+"]")
                self.Position_y = QSlider(Qt.Horizontal, self)
                self.Position_y.setMinimum(int(np.round(self.nely/2)))
                self.Position_y.setMaximum(self.nely)
                self.Position_y.setPageStep(1)
                self.Position_y.setValue(self.nely-1)
                self.Positiony_label = QLabel("Position in y direction ["+str(np.round(self.nely/2))+","+str(self.nely)+"]")
                self.Position_z = QSlider(Qt.Horizontal, self)
                self.Position_z.setMinimum(int(np.round(self.nelz/4)))
                self.Position_z.setMaximum(int(np.round(3*self.nelz/4)))
                self.Position_z.setPageStep(1) 
                self.Position_z.setValue(int(np.round(3*self.nelz/4))-1)
                self.Positionz_label = QLabel("Position in z direction ["+str(np.round(self.nelz/4))+","+str(np.round(3*self.nelz/4))+"]") 
            
            self.Position_x.valueChanged.connect(self.Plot_structure)
            self.Position_y.valueChanged.connect(self.Plot_structure)
            self.Force_y.valueChanged.connect(self.Plot_structure)
            self.Force_x.valueChanged.connect(self.Plot_structure)
            self.Position_z.valueChanged.connect(self.Plot_structure)
            self.Force_z.valueChanged.connect(self.Plot_structure)
            self.mainlayout_save.addWidget(self.Forcex_label)
            self.mainlayout_save.addWidget(self.Force_x)
            self.mainlayout_save.addWidget(self.Forcey_label)
            self.mainlayout_save.addWidget(self.Force_y)
            self.mainlayout_save.addWidget(self.Forcez_label)
            self.mainlayout_save.addWidget(self.Force_z)
            self.mainlayout_save.addWidget(self.Positionx_label)
            self.mainlayout_save.addWidget(self.Position_x)
            self.mainlayout_save.addWidget(self.Positiony_label)
            self.mainlayout_save.addWidget(self.Position_y)
            self.mainlayout_save.addWidget(self.Positionz_label)
            self.mainlayout_save.addWidget(self.Position_z)
            

            self.setLayout(self.mainlayout_save)
            self.counter = self.counter*2
            
            
            
        else:
            ####### Initialize the Plot
            plt.ion() # Ensure that redrawing is possible
            self.fig,self.ax = plt.subplots()
            self.mainlayout_save = self.mainLayout 
            self.nely,self.nelx,self.nelz = self.Extract_Resolusion(self.Resolution.currentText(),2)
            self.arr1 = self.ax.arrow(0, 0, 0, 0, color='r', width=0,head_width=0, head_length=0)
            self.arr2 = self.ax.arrow(0, 0, 0, 0, color='r', width=0,head_width=0, head_length=0)
            #self.Force_x = QLineEdit(self,placeholderText="Force in x direction [-100,100].")
            
            #self.Force_y = QLineEdit(self,placeholderText="Force in y direction [-100,100].")
            self.Forcex_label = QLabel("Force in x direction [-100,100]")
            self.Force_x = QSlider(Qt.Horizontal, self)
            self.Force_x.setMinimum(-100)
            self.Force_x.setMaximum(100)
            self.Force_x.setPageStep(1)
            self.Force_x.setValue(50)
            self.Forcey_label = QLabel("Force in y direction [-100,100]")
            self.Force_y = QSlider(Qt.Horizontal, self)
            self.Force_y.setMinimum(-100)
            self.Force_y.setMaximum(100)
            self.Force_y.setPageStep(1)
            self.Force_y.setValue(-50)
            if self.domain.currentText() == 'Simple':
                if self.Bounday_conditions.currentText() == 'Cantilever':
                #self.Position_x = QLineEdit(self,placeholderText="Position in x direction.")
                    self.Position_x = QSlider(Qt.Horizontal, self)
                    self.Position_x.setMinimum(int(self.nelx/2))
                    self.Position_x.setMaximum(self.nelx)
                    self.Position_x.setPageStep(1)  
                    self.Positionx_label = QLabel("Position in x direction ["+str(self.nelx/2)+","+str(self.nelx)+"]")
                    #self.Position_y = QLineEdit(self,placeholderText="Position in y direction.")
                    self.Position_y = QSlider(Qt.Horizontal, self)
                    self.Position_y.setMinimum(6)
                    self.Position_y.setMaximum(self.nely-5)
                    self.Position_y.setPageStep(1) 
                    self.Positiony_label = QLabel("Position in y direction [6,"+str(self.nely-5)+"]")
                    self.Position_x.setValue(self.nelx-1)
                    self.Position_y.setValue(self.nely-10)
                else:
                    self.Position_x = QSlider(Qt.Horizontal, self)
                    self.Position_x.setMinimum(int(self.nelx/2))
                    self.Position_x.setMaximum(self.nelx-2)
                    self.Position_x.setPageStep(1)  
                    self.Positionx_label = QLabel("Position in x direction ["+str(self.nelx/2)+","+str(self.nelx-2)+"]")
                    self.Position_y = QSlider(Qt.Horizontal, self)
                    self.Position_y.setMinimum(5)
                    self.Position_y.setMaximum(self.nely-5)
                    self.Position_y.setPageStep(1) 
                    self.Positiony_label = QLabel("Position in y direction [5,"+str(self.nely-5)+"]")
                    self.Position_x.setValue(self.nelx-4)
                    self.Position_y.setValue(self.nely-10)
                
            elif self.domain.currentText() == 'Lshape' or self.domain.currentText() == 'Lshape with hole':
                if self.Bounday_conditions.currentText() == 'Cantilever':
                #self.Position_x = QLineEdit(self,placeholderText="Position in x direction.")
                    self.Position_x = QSlider(Qt.Horizontal, self)
                    self.Position_x.setMinimum(self.nelx-20)
                    self.Position_x.setMaximum(self.nelx)
                    self.Position_x.setPageStep(1)  
                    self.Positionx_label = QLabel("Position in x direction ["+str(self.nelx-20)+","+str(self.nelx)+"]")
                    #self.Position_y = QLineEdit(self,placeholderText="Position in y direction.")
                    self.Position_y = QSlider(Qt.Horizontal, self)
                    self.Position_y.setMinimum(self.nely-20)
                    self.Position_y.setMaximum(self.nely-4)
                    self.Position_y.setPageStep(1) 
                    self.Positiony_label = QLabel("Position in y direction ["+str(self.nely-20)+","+str(self.nely)+"]")
                    self.Position_x.setValue(self.nelx-1)
                    self.Position_y.setValue(self.nely-7)
                else:
                    self.Position_x = QSlider(Qt.Horizontal, self)
                    self.Position_x.setMinimum(self.nelx-40)
                    self.Position_x.setMaximum(self.nelx-20)
                    self.Position_x.setPageStep(1)  
                    self.Positionx_label = QLabel("Position in x direction ["+str(self.nelx-40)+","+str(self.nelx-20)+"]")
                    #self.Position_y = QLineEdit(self,placeholderText="Position in y direction.")
                    self.Position_y = QSlider(Qt.Horizontal, self)
                    self.Position_y.setMinimum(self.nely-20)
                    self.Position_y.setMaximum(self.nely-4)
                    self.Position_y.setPageStep(1) 
                    self.Positiony_label = QLabel("Position in y direction ["+str(self.nely-20)+","+str(self.nely)+"]")
                    self.Position_x.setValue(self.nelx-25)
                    self.Position_y.setValue(self.nely-6)
            elif self.domain.currentText() == 'Curved':
                if self.Bounday_conditions.currentText() == 'Cantilever':
                #self.Position_x = QLineEdit(self,placeholderText="Position in x direction.")
                    self.Position_x = QSlider(Qt.Horizontal, self)
                    self.Position_x.setMinimum(self.nelx-10)
                    self.Position_x.setMaximum(self.nelx-1)
                    self.Position_x.setPageStep(1)  
                    self.Positionx_label = QLabel("Position in x direction ["+str(self.nelx-10)+","+str(self.nelx-1)+"]")
                    #self.Position_y = QLineEdit(self,placeholderText="Position in y direction.")
                    self.Position_y = QSlider(Qt.Horizontal, self)
                    self.Position_y.setMinimum(int(0.25*self.nely) + 1)
                    self.Position_y.setMaximum(int(0.75*self.nely) - 1)
                    self.Position_y.setPageStep(1) 
                    self.Positiony_label = QLabel("Position in y direction ["+str(0.25*self.nely + 1)+","+str(0.75*self.nely - 1)+"]")
                    self.Position_x.setValue(self.nelx-4)
                    self.Position_y.setValue(int(0.75*self.nely)-4)
                else:
                    self.Position_x = QSlider(Qt.Horizontal, self)
                    self.Position_x.setMinimum(int(0.25*self.nelx))
                    self.Position_x.setMaximum(int(0.75*self.nelx))
                    self.Position_x.setPageStep(1)  
                    self.Positionx_label = QLabel("Position in x direction ["+str(0.25*self.nelx)+","+str(0.75*self.nelx)+"]")
                    self.Position_y = QSlider(Qt.Horizontal, self)
                    self.Position_y.setMinimum(int(0.25*self.nely))
                    self.Position_y.setMaximum(int(0.75*self.nely))
                    self.Position_y.setPageStep(1) 
                    self.Positiony_label = QLabel("Position in y direction ["+str(0.25*self.nely)+","+str(0.75*self.nely)+"]")
                    self.Position_x.setValue(int(0.75*self.nelx)-2)
                    self.Position_y.setValue(int(0.75*self.nely)-1)
            elif self.domain.currentText() == 'Frame':
                if self.Bounday_conditions.currentText() == 'Cantilever':
                #self.Position_x = QLineEdit(self,placeholderText="Position in x direction.")
                    self.Position_x = QSlider(Qt.Horizontal, self)
                    self.Position_x.setMinimum(int(self.nelx*0.8))
                    self.Position_x.setMaximum(self.nelx-1)
                    self.Position_x.setPageStep(1)  
                    self.Positionx_label = QLabel("Position in x direction ["+str(self.nelx*0.8)+","+str(self.nelx-1)+"]")
                    #self.Position_y = QLineEdit(self,placeholderText="Position in y direction.")
                    self.Position_y = QSlider(Qt.Horizontal, self)
                    self.Position_y.setMinimum(1)
                    self.Position_y.setMaximum(self.nely - 1)
                    self.Position_y.setPageStep(1) 
                    self.Positiony_label = QLabel("Position in y direction [1,"+str(self.nely - 1)+"]")
                    self.Position_x.setValue(self.nelx-4)
                    self.Position_y.setValue(self.nely-5)
                else:
                    self.Position_x = QSlider(Qt.Horizontal, self)
                    self.Position_x.setMinimum(int(0.25*self.nelx))
                    self.Position_x.setMaximum(int(0.75*self.nelx))
                    self.Position_x.setPageStep(1)  
                    self.Positionx_label = QLabel("Position in x direction ["+str(0.25*self.nelx)+","+str(0.75*self.nelx)+"]")
                    self.Position_y = QSlider(Qt.Horizontal, self)
                    self.Position_y.setMinimum(1)
                    self.Position_y.setMaximum(int(0.2*self.nely))
                    self.Position_y.setPageStep(1) 
                    self.Positiony_label = QLabel("Position in y direction [1,"+str(0.2*self.nely)+"]")
                    self.Position_x.setValue(int(0.75*self.nelx)-2)
                    self.Position_y.setValue(int(0.2*self.nely)-2)
            
                
                    
                    
                  
            #self.Position_x = QLineEdit(self,placeholderText="Position in x direction.")
            if self.Method.currentText()=='TL':
                self.Position_x.valueChanged.connect(self.Plot_structure)
                self.Position_y.valueChanged.connect(self.Plot_structure)
                self.Force_y.valueChanged.connect(self.Plot_structure)
                self.Force_x.valueChanged.connect(self.Plot_structure)
            else:
                self.Position_x.sliderReleased.connect(self.Plot_structure)
                self.Position_y.sliderReleased.connect(self.Plot_structure)
                self.Force_x.sliderReleased.connect(self.Plot_structure)
                self.Force_y.sliderReleased.connect(self.Plot_structure)
            #self.Position_y = QLineEdit(self,placeholderText="Position in y direction.")
            self.mainlayout_save.addWidget(self.Forcex_label)
            self.mainlayout_save.addWidget(self.Force_x)
            self.mainlayout_save.addWidget(self.Forcey_label)
            self.mainlayout_save.addWidget(self.Force_y)
            self.mainlayout_save.addWidget(self.Positionx_label)
            self.mainlayout_save.addWidget(self.Position_x)
            self.mainlayout_save.addWidget(self.Positiony_label)
            self.mainlayout_save.addWidget(self.Position_y)
            self.setLayout(self.mainlayout_save)
            self.counter = self.counter*2 + 1
        self.button = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Reset | QDialogButtonBox.Cancel)
        self.button.accepted.connect(self.forces)
        self.button.button(QDialogButtonBox.Reset).clicked.connect(self.Reset_Button)
        self.button.rejected.connect(self.reject)
        self.mainlayout_save.addWidget(self.button)
        #self.start_button = QPushButton('Start', self)
        #self.start_button.setToolTip('Find the optimized structure!!!') ### Message shown when the curser is on the button 
        #self.start_button.clicked.connect(self.Extract_data)
        #self.mainlayout_save.addWidget(self.start_button)
        self.setLayout(self.mainlayout_save)

    def Determine_positionx_for_DOME(self,value):
        self.Positionx_label.setParent(None)
        self.Position_x.setParent(None)
        self.button.setParent(None)
        #self.start_button.setParent(None)
        self.Position_x = QSlider(Qt.Horizontal, self)
        self.Position_x.setMinimum(int(np.round(-np.sqrt(400-(value-20)^2)+20)))
        self.Position_x.setMaximum(int(np.round(np.sqrt(400-(value-20)^2)+20)))
        self.Position_x.setPageStep(1) 
        self.Position_x.setValue(20)
        self.Positionx_label = QLabel("Position in x direction ["+str(np.round(-np.sqrt(400-(value-20)^2)+20))+","+str(np.round(np.sqrt(400-(value-20)^2)+20))+"]")
        self.Position_x.valueChanged.connect(self.Plot_structure)
        self.mainlayout_save.addWidget(self.Positionx_label)
        self.mainlayout_save.addWidget(self.Position_x)
        self.button = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Reset | QDialogButtonBox.Cancel)
        self.button.accepted.connect(self.forces)
        self.button.button(QDialogButtonBox.Reset).clicked.connect(self.Reset_Button)
        self.button.rejected.connect(self.reject)
        self.mainlayout_save.addWidget(self.button)
        self.start_button = QPushButton('Start', self)
        self.start_button.setToolTip('Find the optimized structure!!!') ### Message shown when the curser is on the button 
        self.start_button.clicked.connect(self.Extract_data)
        #self.mainlayout_save.addWidget(self.start_button)
        self.setLayout(self.mainlayout_save)
        

    def Reset_Button(self):
        self.button.setParent(None)
        
        #self.start_button.setParent(None)
        if self.counter%2 == 0:
            self.Force_x.setParent(None)
            self.Force_y.setParent(None)
            self.Force_z.setParent(None)
            self.Forcex_label.setParent(None)
            self.Forcey_label.setParent(None)
            self.Forcez_label.setParent(None)
            self.Position_x.setParent(None)
            self.Position_y.setParent(None)
            self.Position_z.setParent(None)
            self.Positionx_label.setParent(None)
            self.Positiony_label.setParent(None)
            self.Positionz_label.setParent(None)
            
        else:
            self.Force_x.setParent(None)
            self.Force_y.setParent(None)
            self.Forcex_label.setParent(None)
            self.Forcey_label.setParent(None)
            self.Position_x.setParent(None)
            self.Position_y.setParent(None)
            self.Positionx_label.setParent(None)
            self.Positiony_label.setParent(None)
            self.fig.clear()
            plt.close(self.fig)
        
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.forces)
        self.buttonBox.rejected.connect(self.reject)    
        self.mainLayout.addWidget(self.buttonBox)
        
        self.setLayout(self.mainLayout)    
        
    def Extract_data(self):
        '''
        Extract the data from the user's choices
        '''
        if self.Dimension.currentText() == '3D':
            self.forcex_value = int(self.Force_x.value())
            
            self.forcey_value = int(self.Force_y.value())
            self.forcez_value = int(self.Force_z.value())
            self.Positionx_value = int(self.Position_x.value())
            self.Positiony_value = int(self.Position_y.value())
            self.Positionz_value = int(self.Position_z.value())
            self.nely,self.nelx,self.nelz = self.Extract_Resolusion(self.Resolution.currentText(),3)
            X_input = np.zeros([self.nely,self.nelx,self.nelz,5])
            if self.domain.currentText() == 'Simple' or self.domain.currentText() == 'Simple with 2 holes' or self.domain.currentText() == 'Simple with 1 hole':     
                X_input[:,:,:,0] = np.ones([self.nely,self.nelx,self.nelz])
                if self.domain.currentText() == 'Simple with 2 holes':
                    for ely in range(self.nely):
                        for elx in range(self.nelx):
                            if np.sqrt(np.power((ely+1-self.nely/2),2)+np.power((elx+1-self.nelx/5),2))<self.nely/5:  
                                X_input[ely,elx,:,0]=0
                            elif np.sqrt(np.power((ely+1-self.nely/2),2)+np.power((elx+1-4*self.nelx/5),2))<self.nely/5: 
                                X_input[ely,elx,:,0]=0
                elif self.domain.currentText() == 'Simple with 1 hole':
                    for ely in range(self.nely):
                        for elx in range(self.nelx):
                            if np.sqrt(np.power((ely+1-self.nely/2),2)+np.power((elx+1-self.nelx/3),2))<self.nely/4:  
                                X_input[ely,elx,:,0]=0
                    
                    
                if self.Bounday_conditions.currentText() == 'Cantilever':
                    if self.Positiony_value==0 and self.Positionz_value==0:
                        X_input[self.Positiony_value,self.Positionx_value-1,self.Positionz_value,2] = self.forcex_value
                        X_input[self.Positiony_value,self.Positionx_value-1,self.Positionz_value,3] = self.forcey_value
                        X_input[self.Positiony_value,self.Positionx_value-1,self.Positionz_value,4] = self.forcez_value
                    elif self.Positiony_value==0 and self.Positionz_value!=0:
                        X_input[self.Positiony_value,self.Positionx_value-1,self.Positionz_value-1,2] = self.forcex_value
                        X_input[self.Positiony_value,self.Positionx_value-1,self.Positionz_value-1,3] = self.forcey_value
                        X_input[self.Positiony_value,self.Positionx_value-1,self.Positionz_value-1,4] = self.forcez_value
                    elif self.Positiony_value!=0 and self.Positionz_value==0:
                        X_input[self.Positiony_value-1,self.Positionx_value-1,self.Positionz_value,2] = self.forcex_value
                        X_input[self.Positiony_value-1,self.Positionx_value-1,self.Positionz_value,3] = self.forcey_value
                        X_input[self.Positiony_value-1,self.Positionx_value-1,self.Positionz_value,4] = self.forcez_value
                    else:   
                        X_input[self.Positiony_value-1,self.Positionx_value-1,self.Positionz_value-1,2] = self.forcex_value
                        X_input[self.Positiony_value-1,self.Positionx_value-1,self.Positionz_value-1,3] = self.forcey_value
                        X_input[self.Positiony_value-1,self.Positionx_value-1,self.Positionz_value-1,4] = self.forcez_value
                    X_input[:,0:2,:,1] = -1
                elif self.Bounday_conditions.currentText() == 'Simply Supported':
                    if self.Positionz_value!=0:
                        X_input[self.Positiony_value-1,self.Positionx_value-1,self.Positionz_value-1,2] = self.forcex_value
                        X_input[self.Positiony_value-1,self.Positionx_value-1,self.Positionz_value-1,3] = self.forcey_value
                        X_input[self.Positiony_value-1,self.Positionx_value-1,self.Positionz_value-1,4] = self.forcez_value
                    else:   
                        X_input[self.Positiony_value-1,self.Positionx_value-1,self.Positionz_value,2] = self.forcex_value
                        X_input[self.Positiony_value-1,self.Positionx_value-1,self.Positionz_value,3] = self.forcey_value
                        X_input[self.Positiony_value-1,self.Positionx_value-1,self.Positionz_value,4] = self.forcez_value
                    X_input[0:2,0:2,:,1] = -1
                    X_input[0,self.nelx-1,:,1] = -1
                else:
                    if self.Positionz_value!=0:
                        X_input[self.Positiony_value-1,self.Positionx_value-1,self.Positionz_value-1,2] = self.forcex_value
                        X_input[self.Positiony_value-1,self.Positionx_value-1,self.Positionz_value-1,3] = self.forcey_value
                        X_input[self.Positiony_value-1,self.Positionx_value-1,self.Positionz_value-1,4] = self.forcez_value
                    else:   
                        X_input[self.Positiony_value-1,self.Positionx_value-1,self.Positionz_value,2] = self.forcex_value
                        X_input[self.Positiony_value-1,self.Positionx_value-1,self.Positionz_value,3] = self.forcey_value
                        X_input[self.Positiony_value-1,self.Positionx_value-1,self.Positionz_value,4] = self.forcez_value
                    X_input[:,0,:,1] = -1
                    X_input[0:2,self.nelx-1,:,1] = -1
            
            elif self.domain.currentText() == 'Cube' or self.domain.currentText() == 'Cube with 2 holes' or self.domain.currentText() == 'Dome with 2 holes':
                X_input[:,:,:,0] = np.ones([self.nely,self.nelx,self.nelz])
                if self.domain.currentText() == 'Dome with 2 holes':
                    for ely in range(self.nely):
                        for elx in range(self.nelx):
                            for elz in range(self.nelz):
                                if np.sqrt(np.power((ely+1-self.nely/2),2)+np.power((elx+1-self.nelx/2),2)+np.power((elz+1-self.nelz/2),2))>20 and ely<19:  
                                    X_input[ely,elx,elz,0]=0
                                elif np.sqrt(np.power((ely+1-self.nely/2),2)+np.power((elx+1-self.nelx/2),2))<5: 
                                    X_input[ely,elx,:,0]=0
                                elif np.sqrt(np.power((ely+1-self.nely/2),2)+np.power((elz+1-self.nelz/2),2))<5:
                                    X_input[ely,:,elz,0]=0
                                    
                elif self.domain.currentText() == 'Cube with 2 holes':
                    for ely in range(self.nely):
                        for elx in range(self.nelx):
                            for elz in range(self.nelz):  
                                if np.sqrt(np.power((ely+1-self.nely/2),2)+np.power((elx+1-self.nelx/2),2))<5: 
                                    X_input[ely,elx,:,0]=0
                                elif np.sqrt(np.power((ely+1-self.nely/2),2)+np.power((elz+1-self.nelz/2),2))<5:
                                    X_input[ely,:,elz,0]=0
                    
                if self.Positionz_value!=0:
                    X_input[self.Positiony_value-1,self.Positionx_value-1,self.Positionz_value-1,2] = self.forcex_value
                    X_input[self.Positiony_value-1,self.Positionx_value-1,self.Positionz_value-1,3] = self.forcey_value
                    X_input[self.Positiony_value-1,self.Positionx_value-1,self.Positionz_value-1,4] = self.forcez_value
                else:   
                    X_input[self.Positiony_value-1,self.Positionx_value-1,self.Positionz_value,2] = self.forcex_value
                    X_input[self.Positiony_value-1,self.Positionx_value-1,self.Positionz_value,3] = self.forcey_value
                    X_input[self.Positiony_value-1,self.Positionx_value-1,self.Positionz_value,4] = self.forcez_value
                X_input[0,0,0,1] = -1
                X_input[0,self.nelx-1,0,1] = -1
                X_input[0,0,self.nelz-1,1] = -1
                X_input[0,self.nelx-1,self.nelz-1,1] = -1
                     
        else:
            self.forcex_value = int(self.Force_x.value())
            self.forcey_value = int(self.Force_y.value())
            self.Positionx_value = int(self.Position_x.value())
            self.Positiony_value = int(self.Position_y.value())
            self.nely,self.nelx,self.nelz = self.Extract_Resolusion(self.Resolution.currentText(),2)
            X_input = np.zeros([self.nely,self.nelx,5])
            if self.domain.currentText() == 'Simple':
                X_input[:,:,0] = np.ones([self.nely,self.nelx])
                #pos = 2*(self.Positiony_value+1) * (self.Positionx_value+1) 
                pos = 2*self.Positionx_value*(self.nely+1) + 2*self.Positiony_value 
                if pos%(2*(self.nely+1))==0:
                    X_input[self.nely-1,int(np.floor(pos/(2*(self.nely+1)))-2),1]=self.forcex_value
                    X_input[self.nely-1,int(np.floor(pos/(2*(self.nely+1)))-2),2]=self.forcey_value
                else:
                    X_input[int((pos%(2*(self.nely+1)))/2) -1,int(np.floor(pos/(2*(self.nely+1)))-2),1]=self.forcex_value
                    X_input[int((pos%(2*(self.nely+1)))/2) -1,int(np.floor(pos/(2*(self.nely+1)))-2),2]=self.forcey_value
               
                if self.Bounday_conditions.currentText() == 'Cantilever': 
                    X_input[:,0,3] = 1
                    X_input[:,0,4] = 1
                    
                elif self.Bounday_conditions.currentText() == 'Simply Supported':
                    X_input[self.nely-1,0,3] = 1
                    X_input[self.nely-1,0,4] = 1
                    X_input[self.nely-1,self.nelx-1,4] = 1
                elif self.Bounday_conditions.currentText() == 'Constrained Cantilever':
                    X_input[:,0,3] = 1
                    X_input[:,0,4] = 1
                    X_input[self.nely-1,self.nelx-1,4] = 1
                    
            elif self.domain.currentText() == 'Lshape' or self.domain.currentText() == 'Lshape with hole':
                #pos = 2*(self.Positiony_value+1) * (self.Positionx_value+1)
                pos = 2*self.Positionx_value*(self.nely+1) + 2*self.Positiony_value 
                passive = np.zeros([self.nely,self.nelx])
                if self.domain.currentText() == 'Lshape':
                    for j in range(self.nely):
                        for i in range(self.nelx):
                            if j>(self.nely*0.5-1) and i<(self.nelx*0.5-1):
                                passive[j,i] = 1
                            X_input[j,i,0] = 1 -passive[j,i]
                elif self.domain.currentText() == 'Lshape with hole':
                    for j in range(self.nely):
                        for i in range(self.nelx):
                            if j>(self.nely*0.5-1) and i<(self.nelx*0.5-1):
                                passive[j,i] = 1
                            if np.power((i+1-(0.75*self.nelx)),2) +np.power((j+1-(0.25*self.nely)),2) <400:
                                passive[j,i] = 1
                            X_input[j,i,0] = 1 -passive[j,i]
            
                if pos%(2*(self.nely+1))==0:
                    X_input[self.nely-1,int(np.floor(pos/(2*(self.nely+1)))-2),1]=self.forcex_value
                    X_input[self.nely-1,int(np.floor(pos/(2*(self.nely+1)))-2),2]=self.forcey_value
                else:
                    X_input[int((pos%(2*(self.nely+1)))/2) -1,int(np.floor(pos/(2*(self.nely+1)))-2),1]=self.forcex_value
                    X_input[int((pos%(2*(self.nely+1)))/2) -1,int(np.floor(pos/(2*(self.nely+1)))-2),2]=self.forcey_value
               
                if self.Bounday_conditions.currentText() == 'Cantilever': 
                    X_input[:,0,3] = 1
                    X_input[:,0,4] = 1
                elif self.Bounday_conditions.currentText() == 'Constrained Cantilever':
                    X_input[:,0,3] = 1
                    X_input[:,0,4] = 1
                    X_input[self.nely-1,self.nelx-1,4] = 1
                    
            elif self.domain.currentText() == 'Frame' or self.domain.currentText() == 'Curved': 
                pos = 2*(self.Positiony_value+1) + 2* (self.Positionx_value)*(self.nely+1)
                passive = np.zeros([self.nely,self.nelx])
                if self.domain.currentText() == 'Curved':
                    for i in range(self.nelx):
                        for j in range(self.nely):
                            if ((np.power((i+1-self.nelx/2),2)/np.power((self.nelx/2),2)) + (np.power((j+1-1*self.nely/4),2)/np.power((self.nely/4),2)))>1 and j<(1*self.nely/4-1):
                                passive[j,i] = 1
                            if ((np.power((i+1-self.nelx/2),2)/np.power((self.nelx/2 - 8),2)) + (np.power((j+1-self.nely),2)/np.power((self.nely/4),2)))<1 and j>(3*self.nely/4-1):
                                passive[j,i] = 1
                            X_input[j,i,0] = 1 -passive[j,i]
                elif self.domain.currentText() == 'Frame':
                    for i in range(self.nelx):
                        for j in range(self.nely):
                            if (0.2*self.nelx-1)<i and i<(0.8*self.nelx-1) and (0.2*self.nely-1)<j and j<(0.8*self.nely-1):       
                                passive[j,i] = 1
                            X_input[j,i,0] = 1 -passive[j,i]
                
                if pos%(2*(self.nely+1))==0:
                    X_input[self.nely-1,int(np.floor(pos/(2*(self.nely+1)))-2),1]=self.forcex_value
                    X_input[self.nely-1,int(np.floor(pos/(2*(self.nely+1)))-2),2]=self.forcey_value
                else:
                    X_input[int((pos%(2*(self.nely+1)))/2) -1,int(np.floor(pos/(2*(self.nely+1)))-2),1]=self.forcex_value
                    X_input[int((pos%(2*(self.nely+1)))/2) -1,int(np.floor(pos/(2*(self.nely+1)))-2),2]=self.forcey_value
               
                if self.Bounday_conditions.currentText() == 'Cantilever': 
                    X_input[:,0,3] = 1
                    X_input[:,0,4] = 1
                    
                elif self.Bounday_conditions.currentText() == 'Simply Supported':
                    X_input[self.nely-1,0,3] = 1
                    X_input[self.nely-1,0,4] = 1
                    X_input[self.nely-1,self.nelx-1,4] = 1
                elif self.Bounday_conditions.currentText() == 'Constrained Cantilever':
                    X_input[:,0,3] = 1
                    X_input[:,0,4] = 1
                    X_input[self.nely-1,self.nelx-1,4] = 1
           
        return X_input
                
                
    def Prediction(self):
        '''
        Predict the optimum structure based on the input info
        '''
        X_input = self.Extract_data()
        if self.Dimension.currentText() == '3D':
            X_input = np.reshape(X_input,[1,self.nely,self.nelx,self.nelz,5])

            if self.domain.currentText() == 'Simple':
                X_input = self.Down_sampling3D(X_input)
                if self.Resolution.currentText() == '40 x 80 x 10':
                    Y_optimized = self.Simple_408010_model.predict(X_input, verbose=0)
                else:
                    Y_optimized = self.Simple_8016010_model.predict(X_input, verbose=0)
            elif self.domain.currentText() == 'Simple with 1 hole':
                X_input = self.Down_sampling3D(X_input)
                Y_optimized = self.With1hole_model.predict(X_input, verbose=0)
            elif self.domain.currentText() == 'Simple with 2 holes':
                X_input = self.Down_sampling3D(X_input)
                Y_optimized = self.With2holes_model.predict(X_input, verbose=0)
            elif self.domain.currentText() == 'Cube':
                X_input = self.Down_sampling3D_Cube(X_input)
                Y_optimized = self.Cube_model.predict(X_input, verbose=0)
            elif self.domain.currentText() == 'Cube with 2 holes':
                X_input = self.Down_sampling3D_Cube(X_input)
                Y_optimized = self.Cubewith2holes_model.predict(X_input, verbose=0)
            elif self.domain.currentText() == 'Dome with 2 holes':
                X_input = self.Down_sampling3D_Cube(X_input)
                Y_optimized = self.Domewith2holes_model.predict(X_input, verbose=0)
        else:
            
            X_input = np.reshape(X_input,[1,self.nely,self.nelx,5])
            X_input = self.Down_sampling2D(X_input)
            if self.domain.currentText() == 'Simple':
                if self.Resolution.currentText() == '40 x 80':
                    Y_optimized = self.Simple_4080_model.predict(X_input, verbose=0)
                elif self.Resolution.currentText() == '80 x 160':
                    Y_optimized = self.Simple_80160_model.predict(X_input, verbose=0)
                elif self.Resolution.currentText() == '120 x 160':
                    Y_optimized = self.Simple_120160_model.predict(X_input, verbose=0)
                elif self.Resolution.currentText() == '120 x 240':
                    Y_optimized = self.Simple_120240_model.predict(X_input, verbose=0)
                elif self.Resolution.currentText() == '160 x 320':
                    Y_optimized = self.Simple_160320_model.predict(X_input, verbose=0)
                elif self.Resolution.currentText() == '200 x 400':
                    Y_optimized = self.Simple_200400_model.predict(X_input, verbose=0)
            elif self.domain.currentText() == 'Lshape':
                Y_optimized = self.Lshape_model.predict(X_input, verbose=0)

            elif self.domain.currentText() == 'Lshape with hole':
                Y_optimized = self.Lshapewithhole_model.predict(X_input, verbose=0)

            elif self.domain.currentText() == 'Curved':
                Y_optimized = self.Curved_model.predict(X_input, verbose=0)

            elif self.domain.currentText() == 'Frame':
                Y_optimized = self.Frame_model.predict(X_input, verbose=0)
            Y_optimized = np.reshape(Y_optimized,(self.nely,self.nelx))

            #Y_optimized = Y_optimized.T.reshape(self.nely * self.nelx)
            #Y_optimized = Y_optimized.reshape((self.nelx, self.nely)).T
            '''
            for i in range(self.nelx):
                for j in range(self.nely):
                    if Y_optimized[j,i]>0.3 and Y_optimized[j,i]<0.5:
                        Y_optimized[j,i] = 0
                    if Y_optimized[j,i]<0.3:
                        Y_optimized[j,i] = 0
                    if Y_optimized[j,i]>0.5:
                        Y_optimized[j,i]=1
              '''          
            
        return Y_optimized
    
    
    def Plot_structure(self):
        Y_optimized = self.Prediction()
        if self.Dimension.currentText() == '2D':
            self.arr1.remove()
            self.arr2.remove()
            if self.Method.currentText() == 'TL + SIMP':
                Y_SIMP = self.Top_SIMP2D(Y_optimized)
                #Y_SIMP = np.flipud(Y_SIMP)
                self.ax.axis([-self.nely/2, self.nelx+self.nely/2, 3*self.nely/2, -self.nely/2])
                im = self.ax.imshow(Y_SIMP, cmap='gray',interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
                im.set_array(Y_SIMP)
                self.arr1 = self.ax.arrow(self.Positionx_value, self.Positiony_value, self.forcex_value*self.nely/200, 0, color='r', width=self.nely/40,head_width=self.nely/20, head_length=self.nely/40)
                self.arr2 = self.ax.arrow(self.Positionx_value, self.Positiony_value, 0, self.forcey_value*self.nely/200, color='r', width=self.nely/40,head_width=self.nely/20, head_length=self.nely/40)
                
                self.fig.canvas.draw()
                plt.show()
            else:
                Y_optimized = np.round(Y_optimized)
                #Y_optimized = np.flipud(Y_optimized)
                '''
                for i in range(self.nelx):
                    for j in range(self.nely):
                        if Y_optimized[j,i]<0.3:
                            Y_optimized[j,i] = 0
                '''  
                self.ax.axis([-self.nely/2, self.nelx+self.nely/2,3*self.nely/2, -self.nely/2])
                im = self.ax.imshow(-Y_optimized.reshape((self.nely, self.nelx)), cmap='gray',interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
                im.set_array(-Y_optimized.reshape((self.nely, self.nelx)))
                #im = self.ax.imshow(-Y_optimized.reshape((self.nelx,self.nely)).T, cmap='gray',interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
                #im.set_array(-Y_optimized.reshape((self.nelx,self.nely)).T)
                self.arr1 = self.ax.arrow(self.Positionx_value,  self.Positiony_value, self.forcex_value*self.nely/200, 0, color='r', width=self.nely/40,head_width=self.nely/20, head_length=self.nely/40)
                self.arr2 = self.ax.arrow(self.Positionx_value,  self.Positiony_value, 0, self.forcey_value*self.nely/200, color='r', width=self.nely/40,head_width=self.nely/20, head_length=self.nely/40)
                self.fig.canvas.draw()
                plt.show()
            
            
        else:
            Y_optimized = Y_optimized.reshape([self.nely,self.nelx,self.nelz])
            
            self.Plot_3D_structures(Y_optimized)
            '''
            ax = self.fig.gca(projection='3d')
            ax.unit_cube()
            ax.autoscale_view()
            #ax.set_frame_on(True)
            #ax.update(ax.voxels(X_input, facecolors='grey', edgecolor='k'))
            ax.voxels(Y_optimized, facecolors='grey', edgecolor='k')
            plt.show()
            self.fig.show(ax)
            '''
    def Extract_Resolusion(self,s,dim):
        i=0
        nelx = []
        nely = []
        nelz = []
        while s[i]!=' ':
            nelx.append(s[i])
            i = i+1
        nelx = ''.join(nelx)
        nelx = int(nelx)
        
        i = i+3
        
        if dim == 3:
            while s[i]!=' ':
                nely.append(s[i])
                i = i+1
            nely = ''.join(nely)
            nely = int(nely)
            i = i+3
            for j in range(i,len(s)):
                nelz.append(s[j])
            nelz = ''.join(nelz)
            nelz = int(nelz)
        else:
            for j in range(i,len(s)):
                nely.append(s[j])
            nely = ''.join(nely)
            nely = int(nely)
            nelz.append('0')
            nelz = ''.join(nelz)
            nelz = int(nelz)

        return nelx,nely,nelz

    
    def Down_sampling2D(self,x):
        y = np.zeros([np.size(x,axis=0),40,80,5])
        
        ##### First Channel
        y[:,0:40,0:80,0] = np.ones((40,80))
        ##### Forth Channel
        y[:,0:15,0,3] = x[:,0:15,0,3]
        y[:,15:40,0,3] = x[:,(np.size(x,axis=1)-25):np.size(x,axis=1),0,3]
        y[:,39,79,3] = x[:,np.size(x,axis=1)-1,np.size(x,axis=2)-1,3]
        
        ##### Fifth Channel
        y[:,0:15,0,4] = x[:,0:15,0,4]
        y[:,15:40,0,4] = x[:,(np.size(x,axis=1)-25):np.size(x,axis=1),0,4]
        y[:,39,79,4] = x[:,np.size(x,axis=1)-1,np.size(x,axis=2)-1,4]
        
        
        
        for k in range(np.size(x,axis=0)):
            ##### Second Channel
            xx = x[k,:,:,1]
            (i,j) = xx.nonzero()
            if i:
                posx = np.floor(i*40/np.size(x,axis=1))
                posy = np.floor(j*80/np.size(x,axis=2))
                y[k,int(posx),int(posy),1] = x[k,int(i),int(j),1] 
            
            ##### Third Channel
            xx = x[k,:,:,1]
            (i,j) = xx.nonzero()
            if i:
                posx = np.floor(i*40/np.size(x,axis=1))
                posy = np.floor(j*80/np.size(x,axis=2))
                y[k,int(posx),int(posy),2] = x[k,int(i),int(j),2]
            
        return y
   

    def Down_sampling3D(self,x):
        y = np.zeros([np.size(x,axis=0),20,40,10,5])
        
        ##### First Channel
        y[:,0:20,0:40,0:10,0] = np.ones((20,40,10))
        ##### Second Channel
        y[:,0:2,0:2,0:5,1] = x[:,0:2,0:2,0:5,1]
        y[:,0:2,0:2,5:10,1] = x[:,0:2,0:2,(np.size(x,axis=3)-5):np.size(x,axis=3),1]
        
        y[:,0,39,0:5,1] = x[:,0,np.size(x,axis=2)-1,0:5,1]
        y[:,0,39,5:10,1] = x[:,0,np.size(x,axis=2)-1,(np.size(x,axis=3)-5):np.size(x,axis=3),1]
        
        y[:,0:10,0:2,0:5,1] = x[:,0:10,0:2,0:5,1]
        y[:,10:20,0:2,5:10,1] = x[:,(np.size(x,axis=1)-10):np.size(x,axis=1),0:2,(np.size(x,axis=3)-5):np.size(x,axis=3),1]
        
        y[:,0:10,1,0:5,1] = x[:,0:10,1,0:5,1]
        y[:,10:20,1,5:10,1] = x[:,(np.size(x,axis=1)-10):np.size(x,axis=1),1,(np.size(x,axis=3)-5):np.size(x,axis=3),1]
        
        y[:,0:2,39,0:5,1] = x[:,0:2,np.size(x,axis=2)-1,0:5,1]
        y[:,0:2,39,5:10,1] = x[:,0:2,np.size(x,axis=2)-1,(np.size(x,axis=3)-5):np.size(x,axis=3),1]
        
        
        
        for jj in range(np.size(x,axis=0)):
            ##### Third Channel
            xx = x[jj,:,:,:,2]
            (i,j,k) = xx.nonzero()
            if i:
                posx = np.floor(i*20/np.size(x,axis=1))
                posy = np.floor(j*40/np.size(x,axis=2))
                posz = np.floor(k*10/np.size(x,axis=3))
                y[jj,int(posx),int(posy),int(posz),2] = x[jj,int(i),int(j),int(k),2] 
            
            ##### Forth Channel
            xx = x[jj,:,:,:,3]
            (i,j,k) = xx.nonzero()
            if i:
                posx = np.floor(i*20/np.size(x,axis=1))
                posy = np.floor(j*40/np.size(x,axis=2))
                posz = np.floor(k*10/np.size(x,axis=3))
                y[jj,int(posx),int(posy),int(posz),3] = x[jj,int(i),int(j),int(k),3]
            ##### Fifth channel
            xx = x[jj,:,:,:,4]
            (i,j,k) = xx.nonzero()
            if i:
                posx = np.floor(i*20/np.size(x,axis=1))
                posy = np.floor(j*40/np.size(x,axis=2))
                posz = np.floor(k*10/np.size(x,axis=3))
                y[jj,int(posx),int(posy),int(posz),4] = x[jj,int(i),int(j),int(k),4]
            
        return y

    def Down_sampling3D_Cube(self,x):
        y = np.zeros([np.size(x, axis=0), 20, 40, 10, 5])

        ##### First Channel
        y[:, 0:20, 0:40, 0:10, 0] = np.ones((20, 40, 10))
        ##### Second Channel
        y[:, 0:2, 0:2, 0:10, 1] = -1
        y[:, 0, 39, 0:10, 1] = -1

        for jj in range(np.size(x, axis=0)):
            ##### Third Channel
            xx = x[jj, :, :, :, 2]
            (i, j, k) = xx.nonzero()
            if i:
                posx = np.floor(i * 20 / np.size(x, axis=1))
                posy = np.floor(j * 40 / np.size(x, axis=2))
                posz = np.floor(k * 10 / np.size(x, axis=3))
                y[jj, int(posx), int(posy), int(posz), 2] = x[jj, int(i), int(j), int(k), 2]

                ##### Forth Channel
            xx = x[jj, :, :, :, 3]
            (i, j, k) = xx.nonzero()
            if i:
                posx = np.floor(i * 20 / np.size(x, axis=1))
                posy = np.floor(j * 40 / np.size(x, axis=2))
                posz = np.floor(k * 10 / np.size(x, axis=3))
                y[jj, int(posx), int(posy), int(posz), 3] = x[jj, int(i), int(j), int(k), 3]
            ##### Fifth channel
            xx = x[jj, :, :, :, 4]
            (i, j, k) = xx.nonzero()
            if i:
                posx = np.floor(i * 20 / np.size(x, axis=1))
                posy = np.floor(j * 40 / np.size(x, axis=2))
                posz = np.floor(k * 10 / np.size(x, axis=3))
                y[jj, int(posx), int(posy), int(posz), 4] = x[jj, int(i), int(j), int(k), 4]

        return y



    def Top_SIMP2D(self,Y_Predicted):
        '''
        The SIMP method used in out TL + SIMP. The output of our model is given as input to a SIMP solver to create a high quality and well connected
        sructures.

        This SIMP code is taken from: https://www.topopt.mek.dtu.dk/apps-and-software
        '''
        # Default input parameters
        nelx=self.nelx
        nely=self.nely
        volfrac=0.5
        rmin=1.5
        penal=3.0
        ft=0# ft==0 -> sens, ft==1 -> dens
        
        # Max and min stiffness
        Emin=1e-9
        Emax=1.0
        # dofs:
        ndof = 2*(nelx+1)*(nely+1)
        # Allocate design variables (as array), initialize and allocate sens.
        x=volfrac * np.ones(nely*nelx,dtype=float)
        xold=x.copy()
        xPhys=x.copy()
        Y_Predicted = Y_Predicted.T.reshape(nely*nelx)
        g=0 # must be initialized to use the NGuyen/Paulino OC approach
        dc=np.zeros((nely,nelx), dtype=float)
        # FE: Build the index vectors for the for coo matrix format.
        KE=self.lk()
        edofMat=np.zeros((nelx*nely,8),dtype=int)
        
        for elx in range(nelx):
            for ely in range(nely):
                el = ely+elx*nely
                n1=(nely+1)*elx+ely
                n2=(nely+1)*(elx+1)+ely
                edofMat[el,:]=np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])
                
                
        # Construct the index pointers for the coo format
        iK = np.kron(edofMat,np.ones((8,1))).flatten()
        jK = np.kron(edofMat,np.ones((1,8))).flatten() 
        # Filter: Build (and assemble) the index+data vectors for the coo matrix format
        nfilter=int(nelx*nely*((2*(np.ceil(rmin)-1)+1)**2))
        iH = np.zeros(nfilter)
        jH = np.zeros(nfilter)
        sH = np.zeros(nfilter)
        cc=0
        
        for i in range(nelx):
            for j in range(nely):
                row=i*nely+j
                kk1=int(np.maximum(i-(np.ceil(rmin)-1),0))
                kk2=int(np.minimum(i+np.ceil(rmin),nelx))
                ll1=int(np.maximum(j-(np.ceil(rmin)-1),0))
                ll2=int(np.minimum(j+np.ceil(rmin),nely))
                for k in range(kk1,kk2):
                    for l in range(ll1,ll2):
                        col=k*nely+l
                        fac=rmin-np.sqrt(((i-k)*(i-k)+(j-l)*(j-l)))
                        iH[cc]=row
                        jH[cc]=col
                        sH[cc]=np.maximum(0.0,fac)
                        cc=cc+1
                        
                        
        # Finalize assembly and convert to csc format
        H=coo_matrix((sH,(iH,jH)),shape=(nelx*nely,nelx*nely)).tocsc()	
        Hs=H.sum(1)
        
        # Solution and RHS vectors
        f=np.zeros((ndof,1))
        u=np.zeros((ndof,1))
        dofs=np.arange(2*(nelx+1)*(nely+1))
        
        fx = self.forcex_value
        fy = self.forcey_value
        #Positionx_value = 0.75*nelx
        #Positiony_value = nely
        pos = 2*self.Positionx_value*(nely+1) + 2*self.Positiony_value 
        f[int(pos-2),0] = fx
        f[int(pos-1),0] = fy
        
        
        if self.Bounday_conditions.currentText() == 'Contilever':
            fixed=dofs[0:2*(nely+1)]            
        elif self.Bounday_conditions.currentText() == 'Simply Supported':          
            fixed=np.union1d(np.array([2*nely,2*nely+1]),np.array([2*(nelx+1)*(nely+1)-1]))
        else:
            fixed=np.union1d(dofs[0:2*(nely+1)],np.array([2*(nelx+1)*(nely+1)-1]))

        free=np.setdiff1d(dofs,fixed)
        loop=0
        change=1
        dv = np.ones(nely*nelx)
        dc = np.ones(nely*nelx)
        ce = np.ones(nely*nelx)
        
        while change>0.01 and loop<10:
        
            loop=loop+1
            if loop ==1:
                xPhys = Y_Predicted
                x = Y_Predicted
            # Setup and solve FE problem
            sK=((KE.flatten()[np.newaxis]).T*(Emin+(xPhys)**penal*(Emax-Emin))).flatten(order='F')
            K = coo_matrix((sK,(iK,jK)),shape=(ndof,ndof)).tocsc()
            # Remove constrained dofs from matrix
            K = K[free,:][:,free]
            # Solve system 
            u[free,0]=spsolve(K,f[free,0])    
            # Objective and sensitivity
            ce[:] = (np.dot(u[edofMat].reshape(nelx*nely,8),KE) * u[edofMat].reshape(nelx*nely,8) ).sum(1)
            obj=( (Emin+xPhys**penal*(Emax-Emin))*ce ).sum()
            dc[:]=(-penal*xPhys**(penal-1)*(Emax-Emin))*ce
            dv[:] = np.ones(nely*nelx)
        	# Sensitivity filtering:
            
                
            if ft==0:
                dc[:] = np.asarray((H*(x*dc))[np.newaxis].T/Hs)[:,0] / np.maximum(0.001,x)
            elif ft==1:
                dc[:] = np.asarray(H*(dc[np.newaxis].T/Hs))[:,0]
                dv[:] = np.asarray(H*(dv[np.newaxis].T/Hs))[:,0]
        	# Optimality criteria
            xold[:]=x
            (x[:],g)=self.oc(nelx,nely,x,volfrac,dc,dv,g)
            # Filter design variables
            if ft==0:   
                xPhys[:]=x
            elif ft==1:	
                xPhys[:]=np.asarray(H*x[np.newaxis].T/Hs)[:,0]
            # Compute the change by the inf. norm
            change=np.linalg.norm(x.reshape(nelx*nely,1)-xold.reshape(nelx*nely,1),np.inf)
            
            # Plot to screen
            self.ax.axis([-nely / 2, nelx + nely / 2, 3*nely/2, -nely/2])
            im = self.ax.imshow(-xPhys.reshape((nelx,nely)).T, cmap='gray',interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
            im.set_array(-xPhys.reshape((nelx,nely)).T)
        
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.show()
            # Write iteration history to screen (req. Python 2.6 or newer)
            #print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(loop,obj,(g+volfrac*nelx*nely)/(nelx*nely),change))
        print("DONE!")        
        return -xPhys.reshape((nelx,nely)).T

      
    # Optimality criterion
    def oc(self,nelx,nely,x,volfrac,dc,dv,g):
        l1=0
        l2=1e9
        move=0.2
        # reshape to perform vector operations
        xnew=np.zeros(nelx*nely)
        while (l2-l1)/(l1+l2)>1e-3:
            lmid=0.5*(l2+l1) + 0.0001
            xnew[:]= np.maximum(0.0,np.maximum(x-move,np.minimum(1.0,np.minimum(x+move,x*np.sqrt(-dc/dv/lmid)))))
            gt=g+np.sum((dv*(xnew-x)))
            if gt>0 :
                l1=lmid
            else:
                l2=lmid
        return (xnew,gt) 

    #element stiffness matrix
    def lk(self):
        E=1
        nu=0.3
        k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
        KE = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ]);
        return (KE)           




    def Plot_3D_structures(self,Y_optimized): 
        self.renderer.RemoveActor(self.actor2)
        self.renderer.RemoveActor(self.actor3)
        self.renderer.RemoveActor(self.actor4)
        self.points = vtk.vtkPoints()
        a = np.where(Y_optimized>0.5)
        for i in range(np.size(a[0])):
            self.points.InsertNextPoint(a[0][i],a[1][i],a[2][i])
            
        
        self.polydata.SetPoints(self.points)
        self.polydata.Modified()

        self.glyph3D.SetInputData(self.polydata)
        self.glyph3D.Update()

        self.mapper.SetInputConnection(self.glyph3D.GetOutputPort())
        self.mapper.Update()
        self.actor.SetMapper(self.mapper)
        self.actor2,self.actor3,self.actor4 = self.Plot_3D_Arrow()
        self.renderer.AddActor(self.actor2)
        self.renderer.AddActor(self.actor3)
        self.renderer.AddActor(self.actor4)
        self.renderWindow.Render()
        self.renderWindowInteractor.Start()
        
        
        
    def Plot_3D_Arrow(self):
        USER_MATRIX = True
        colors = vtk.vtkNamedColors()
           # Set the background color.
        colors.SetColor("BkgColor", [26, 51, 77, 255])
        
        # Generate a random start and end point
        startPoint = [0] * 3
        rng = vtk.vtkMinimalStandardRandomSequence()
        rng.SetSeed(8775070)  # For testing.
        
        startPoint[0] = self.nely - self.Positiony_value
        startPoint[1] = self.Positionx_value
        startPoint[2] = self.Positionz_value
        #################################### First Arrow
        # Create an arrow.
        arrowSource = vtk.vtkArrowSource()
        endPoint = [0] * 3
        endPoint[0] = self.nely - self.Positiony_value + self.forcey_value*self.nely/200 
        endPoint[1] = self.Positionx_value
        endPoint[2] = self.Positionz_value
        # Compute a basis
        normalizedX = [0] * 3
        normalizedY = [0] * 3
        normalizedZ = [0] * 3
        
        # The X axis is a vector from start to end
        vtk.vtkMath.Subtract(endPoint, startPoint, normalizedX)
        length = vtk.vtkMath.Norm(normalizedX)
        vtk.vtkMath.Normalize(normalizedX)
        
        # The Z axis is an arbitrary vector cross X
        arbitrary = [0] * 3
        for i in range(0, 3):
            rng.Next()
            arbitrary[i] = rng.GetRangeValue(-10, 10)
        vtk.vtkMath.Cross(normalizedX, arbitrary, normalizedZ)
        vtk.vtkMath.Normalize(normalizedZ)
        
        # The Y axis is Z cross X
        vtk.vtkMath.Cross(normalizedZ, normalizedX, normalizedY)
        matrix = vtk.vtkMatrix4x4()
        
        # Create the direction cosine matrix
        matrix.Identity()
        for i in range(0, 3):
            matrix.SetElement(i, 0, normalizedX[i])
            matrix.SetElement(i, 1, normalizedY[i])
            matrix.SetElement(i, 2, normalizedZ[i])
        
        # Apply the transforms
        transform = vtk.vtkTransform()
        transform.Translate(startPoint)
        transform.Concatenate(matrix)
        transform.Scale(length, length, length)
        
        # Transform the polydata
        transformPD = vtk.vtkTransformPolyDataFilter()
        transformPD.SetTransform(transform)
        transformPD.SetInputConnection(arrowSource.GetOutputPort())
        
        # Create a mapper and actor for the arrow
        mapper = vtk.vtkPolyDataMapper()
        actor = vtk.vtkActor()
        if USER_MATRIX:
            mapper.SetInputConnection(arrowSource.GetOutputPort())
            actor.SetUserMatrix(transform.GetMatrix())
        else:
            mapper.SetInputConnection(transformPD.GetOutputPort())
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(colors.GetColor3d("Cyan"))
        
        ################### Second Arrow
        arrowSource2 = vtk.vtkArrowSource()
        endPoint2 = [0] * 3
        endPoint2[0] = self.nely - self.Positiony_value
        endPoint2[1] = self.Positionx_value + self.forcex_value*self.nely/200
        endPoint2[2] = self.Positionz_value
        # Compute a basis
        normalizedX2 = [0] * 3
        normalizedY2 = [0] * 3
        normalizedZ2 = [0] * 3
        
        # The X axis is a vector from start to end
        vtk.vtkMath.Subtract(endPoint2, startPoint, normalizedX2)
        length2 = vtk.vtkMath.Norm(normalizedX2)
        vtk.vtkMath.Normalize(normalizedX2)
        
        # The Z axis is an arbitrary vector cross X
        arbitrary2 = [0] * 3
        for i in range(0, 3):
            rng.Next()
            arbitrary2[i] = rng.GetRangeValue(-10, 10)
        vtk.vtkMath.Cross(normalizedX2, arbitrary2, normalizedZ2)
        vtk.vtkMath.Normalize(normalizedZ2)
        
        # The Y axis is Z cross X
        vtk.vtkMath.Cross(normalizedZ2, normalizedX2, normalizedY2)
        matrix2 = vtk.vtkMatrix4x4()
        
        # Create the direction cosine matrix
        matrix2.Identity()
        for i in range(0, 3):
            matrix2.SetElement(i, 0, normalizedX2[i])
            matrix2.SetElement(i, 1, normalizedY2[i])
            matrix2.SetElement(i, 2, normalizedZ2[i])
        
        # Apply the transforms
        transform2 = vtk.vtkTransform()
        transform2.Translate(startPoint)
        transform2.Concatenate(matrix2)
        transform2.Scale(length2, length2, length2)
        
        # Transform the polydata
        transformPD2 = vtk.vtkTransformPolyDataFilter()
        transformPD2.SetTransform(transform2)
        transformPD2.SetInputConnection(arrowSource2.GetOutputPort())
        
        # Create a mapper and actor for the arrow
        mapper2 = vtk.vtkPolyDataMapper()
        actor2 = vtk.vtkActor()
        if USER_MATRIX:
            mapper2.SetInputConnection(arrowSource2.GetOutputPort())
            actor2.SetUserMatrix(transform2.GetMatrix())
        else:
            mapper2.SetInputConnection(transformPD2.GetOutputPort())
        actor2.SetMapper(mapper2)
        actor2.GetProperty().SetColor(colors.GetColor3d("Bisque"))
        
        ################### Third Arrow
        arrowSource3 = vtk.vtkArrowSource()
        endPoint3 = [0] * 3
        endPoint3[0] = self.nely - self.Positiony_value
        endPoint3[1] = self.Positionx_value
        endPoint3[2] = self.Positionz_value + self.forcez_value*self.nely/200
        # Compute a basis
        normalizedX3 = [0] * 3
        normalizedY3 = [0] * 3
        normalizedZ3 = [0] * 3
        
        # The X axis is a vector from start to end
        vtk.vtkMath.Subtract(endPoint3, startPoint, normalizedX3)
        length3 = vtk.vtkMath.Norm(normalizedX3)
        vtk.vtkMath.Normalize(normalizedX3)
        
        # The Z axis is an arbitrary vector cross X
        arbitrary3 = [0] * 3
        for i in range(0, 3):
            rng.Next()
            arbitrary3[i] = rng.GetRangeValue(-10, 10)
        vtk.vtkMath.Cross(normalizedX3, arbitrary3, normalizedZ3)
        vtk.vtkMath.Normalize(normalizedZ3)
        
        # The Y axis is Z cross X
        vtk.vtkMath.Cross(normalizedZ3, normalizedX3, normalizedY3)
        matrix3 = vtk.vtkMatrix4x4()
        
        # Create the direction cosine matrix
        matrix3.Identity()
        for i in range(0, 3):
            matrix3.SetElement(i, 0, normalizedX3[i])
            matrix3.SetElement(i, 1, normalizedY3[i])
            matrix3.SetElement(i, 2, normalizedZ3[i])
        
        # Apply the transforms
        transform3 = vtk.vtkTransform()
        transform3.Translate(startPoint)
        transform3.Concatenate(matrix3)
        transform3.Scale(length3, length3, length3)
        
        # Transform the polydata
        transformPD3 = vtk.vtkTransformPolyDataFilter()
        transformPD3.SetTransform(transform3)
        transformPD3.SetInputConnection(arrowSource3.GetOutputPort())
        
        # Create a mapper and actor for the arrow
        mapper3 = vtk.vtkPolyDataMapper()
        actor3 = vtk.vtkActor()
        if USER_MATRIX:
            mapper3.SetInputConnection(arrowSource3.GetOutputPort())
            actor3.SetUserMatrix(transform3.GetMatrix())
        else:
            mapper3.SetInputConnection(transformPD3.GetOutputPort())
        actor3.SetMapper(mapper3)
        actor3.GetProperty().SetColor(colors.GetColor3d("Navy"))
        return actor,actor2,actor3


if __name__ == '__main__':
    app = QApplication(sys.argv)
    dialog = Dialog()
    sys.exit(dialog.exec_())