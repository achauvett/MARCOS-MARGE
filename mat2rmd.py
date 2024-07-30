import os
import sys
#*****************************************************************************
# Get the directory of the current script
main_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(main_directory)
parent_directory = os.path.dirname(parent_directory)

# Define the subdirectories you want to add to sys.path
subdirs = ['MaRGE', 'marcos_client']

# Add the subdirectories to sys.path
for subdir in subdirs:
    full_path = os.path.join(parent_directory, subdir)
    sys.path.append(full_path)
#******************************************************************************
import numpy as np
import experiment as ex
import scipy.signal as sig
from scipy.stats import linregress
import configs.hw_config as hw # Import the scanner hardware config
import configs.units as units
import seq.mriBlankSeq as blankSeq  # Import the mriBlankSequence for any new sequence.

from datetime import date
from datetime import datetime
import ismrmrd
import ismrmrd.xsd
import datetime
import ctypes
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from scipy.io import loadmat

#*********************************************************************************
#*********************************************************************************
#*********************************************************************************

class Converter():
    def __init__(self):
        super().__init__()
        self.acq = ismrmrd.Acquisition()
        self.img = ismrmrd.Image()
        self.header= ismrmrd.xsd.ismrmrdHeader() 
        
        
    def loadmat (self, fil):
        
        """
        Open a file dialog to select a .mat file and return its path.

        Returns:
            str: The path of the selected .mat file.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        default_dir = "C:/Users/Portatil PC 6/PycharmProjects/pythonProject1/Results"

        # Open the file dialog and prompt the user to select a .mat file
        file_name, _ = QFileDialog.getOpenFileName(self, "Select a .mat file", default_dir, "MAT Files (*.mat)",
                                                   options=options)

        mat = loadmat(file_name)
        try:
            data = mat['dataFull'] #name for RARE seq
        except KeyError:
            try:
                data = mat['data_full']  #name for GRE seq
            except KeyError:
                raise KeyError("Neither 'dataFull' nor 'data_full' found in the .mat file")
        return data
        
        
        
dset = ismrmrd.Dataset(path, f'/dataset', True) # Create the dataset
        
        nScans = self.mapVals['nScans']
        nPoints = np.array(self.mapVals['nPoints'])
        etl = self.mapVals['etl']
        nRD = self.nPoints[0]
        nPH = self.nPoints[1]
        nSL = self.nPoints[2]
        ind = self.getIndex(self.etl, nPH, self.sweepMode)
        nRep = (nPH//etl)*nSL
        bw = self.mapVals['bw']
        
        axesOrientation = self.axesOrientation
        axesOrientation_list = axesOrientation.tolist()

        read_dir = [0, 0, 0]
        phase_dir = [0, 0, 0]
        slice_dir = [0, 0, 0]

        read_dir[axesOrientation_list.index(0)] = 1
        phase_dir[axesOrientation_list.index(1)] = 1
        slice_dir[axesOrientation_list.index(2)] = 1
        
        # Experimental Conditions field
        exp = ismrmrd.xsd.experimentalConditionsType() 
        magneticFieldStrength = hw.larmorFreq*1e6/hw.gammaB
        exp.H1resonanceFrequency_Hz = hw.larmorFreq

        self.header.experimentalConditions = exp 

        # Acquisition System Information field
        sys = ismrmrd.xsd.acquisitionSystemInformationType() 
        sys.receiverChannels = 1 
        self.header.acquisitionSystemInformation = sys


        # Encoding field can be filled if needed
        encoding = ismrmrd.xsd.encodingType()  
        encoding.trajectory = ismrmrd.xsd.trajectoryType.CARTESIAN
              
        
        dset.write_xml_header(self.header.toXML()) # Write the header to the dataset
                
        
        
        new_data = np.zeros((nPH * nSL * nScans, nRD + 2*self.addRdPoints))
        new_data = np.reshape(self.dataFullmat, (nScans, nSL, nPH, nRD+ 2*self.addRdPoints))
        
        counter=0  
        for scan in range(nScans):
            for slice_idx in range(nSL):
                for phase_idx in range(nPH):
                    
                    line = new_data[scan, slice_idx, phase_idx, :]
                    line2d = np.reshape(line, (1, nRD+2*self.addRdPoints))
                    acq = ismrmrd.Acquisition.from_array(line2d, None)
                    
                    index_in_repetition = phase_idx % etl
                    current_repetition = (phase_idx // etl) + (slice_idx * (nPH // etl))
                    
                    acq.clearAllFlags()
                    
                    if index_in_repetition == 0: 
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_CONTRAST)
                    elif index_in_repetition == etl - 1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_CONTRAST)
                    
                    if ind[phase_idx]== 0:
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_PHASE)
                    elif ind[phase_idx] == nPH - 1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_PHASE)
                    
                    if slice_idx == 0:
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_SLICE)
                    elif slice_idx == nSL - 1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
                        
                    if int(current_repetition) == 0:
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_REPETITION)
                    elif int(current_repetition) == nRep - 1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_REPETITION)
                        
                    if scan == 0:
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_AVERAGE)
                    elif scan == nScans-1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_AVERAGE)
                    
                    
                    counter += 1 
                    
                    # +1 to start at 1 instead of 0
                    acq.idx.repetition = int(current_repetition + 1)
                    acq.idx.kspace_encode_step_1 = ind[phase_idx]+1 # phase
                    acq.idx.slice = slice_idx + 1
                    acq.idx.contrast = index_in_repetition + 1
                    acq.idx.average = scan + 1 # scan
                    
                    acq.scan_counter = counter
                    acq.discard_pre = self.addRdPoints
                    acq.discard_post = self.addRdPoints
                    acq.sample_time_us = 1/bw
                    acq.position=(ctypes.c_float * 3)(*self.dfov) 
                    
                    acq.read_dir = (ctypes.c_float * 3)(*read_dir)
                    acq.phase_dir = (ctypes.c_float * 3)(*phase_dir)
                    acq.slice_dir = (ctypes.c_float * 3)(*slice_dir)
                    
                    dset.append_acquisition(acq) # Append the acquisition to the dataset
                        
                        
        image=self.mapVals['image3D']
        image_reshaped = np.reshape(image, (nSL, nPH, nRD))
        
        for slice_idx in range (nSL): ## image3d does not have scan dimension
            
            image_slice = image_reshaped[slice_idx, :, :]
            img = ismrmrd.Image.from_array(image_slice)
            
            img.field_of_view = (ctypes.c_float * 3)(*(self.fov)*10) # mm
            img.position = (ctypes.c_float * 3)(*self.dfov)
            
            img.data_type= 8 ## COMPLEX FLOAT
            img.image_type = 5 ## COMPLEX
            
            
            
            img.read_dir = (ctypes.c_float * 3)(*read_dir)
            img.phase_dir = (ctypes.c_float * 3)(*phase_dir)
            img.slice_dir = (ctypes.c_float * 3)(*slice_dir)
            
            dset.append_image(f"image_raw", img) # Append the image to the dataset
                
        
        dset.close()    

 mat = loadmat('C:\\Users\\Axelle\\Desktop\\MARCOS MARGE\\MaRGE\\experiments\\acquisitions\\None\\2024.06.18.15.29\\Phantom\\Left\\mat\\RARE.2024.06.18.15.30.00.289.mat')
    kspacemat = mat['kSpace3D']