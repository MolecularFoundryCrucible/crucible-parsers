#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 17:45:48 2026

@author: mkywall
"""
from crucible.parsers import BaseParser

import os
import logging
from datetime import datetime
from crucible import BaseDataset



logger = logging.getLogger(__name__)

#%%

def read_dm_image(source_file):
    """Read image data from a DM3 or DM4 file using ncempy.
    
    Returns a 2D numpy array and the image dimensionality.
    """
    import warnings
    try:
        import numpy as np
        import ncempy.io as nio
    except:
        raise ImportError('The following packages are required for the DM parser to work:\
                            numpy, pillow, ncempy.io, py4DSTEM (0.13.17), matplotlib')
    
    f = nio.dm.fileDM(source_file, on_memory=True)
    f.parseHeader()
    ds = f.getDataset(0)
    img = ds['data']
    ndim = len(img.shape)
    
    if ndim == 2:
        return np.asarray(img), ndim
    elif ndim == 3:
        # Take the first slice
        return np.asarray(img[0, :, :]), ndim
    elif ndim == 4:
        # Take the first slice along both extra dimensions
        return np.asarray(img[0, 0, :, :]), ndim
    else:
        warnings.warn(f"Unexpected dimensionality: {ndim}")
        return None, ndim


class DigitalMicrographParser(BaseParser):

    _measurement = None
    _data_format = None
    _instrument_name = None

    def __init__(self, files_to_upload=None, project_id=None,
                 metadata=None, keywords=None, mfid=None,
                 measurement=None, dataset_name=None,
                 session_name=None, public=False, instrument_name=None,
                 data_format=None):
        """
        Initialize the parser with dataset properties.

        Args:
            files_to_upload (list, optional): Files to upload
            project_id (str, optional): Crucible project ID
            metadata (dict, optional): Scientific metadata
            keywords (list, optional): Keywords for the dataset
            mfid (str, optional): Unique dataset identifier
            measurement (str, optional): Measurement type
            dataset_name (str, optional): Human-readable dataset name
            session_name (str, optional): Session name for grouping datasets
            public (bool, optional): Whether dataset is public. Defaults to False.
            instrument_name (str, optional): Instrument name
            data_format (str, optional): Data format type
        """
        # Use parser's defaults if not provided
        if measurement is None:
            measurement = self._measurement
        if data_format is None:
            data_format = self._data_format
        if instrument_name is None:
            instrument_name = self._instrument_name

        # Dataset properties
        self.project_id      = project_id
        self.files_to_upload = files_to_upload or []
        self.mfid            = mfid
        self.measurement     = measurement
        self.dataset_name    = dataset_name
        self.session_name    = session_name
        self.public          = public
        self.instrument_name = instrument_name
        self.data_format     = data_format
        self.source_folder   = os.getcwd()
        self.thumbnail       = None

        # Initialize with user-provided metadata/keywords
        self.scientific_metadata = metadata or {}
        self.keywords = keywords or []
        self._client = None

        # Call parser-specific extraction (Template Method Pattern)
        self.parse()

        return

    def parse(self):
        """
        Parse Digital Micrograph Files and extract metadata.

        This is a hook method that subclasses should override to implement
        their specific parsing logic. The base implementation does nothing
        (generic upload with no parsing).

        Subclasses should:
        - Read and parse domain-specific file formats
        - Call self.add_metadata() to merge extracted metadata with user-provided metadata
        - Call self.add_keywords() to add domain-specific keywords to user-provided keywords
        - Update self.files_to_upload if needed (e.g., add related files)
        - Set self.thumbnail if generating a visualization
        - Access all instance variables (self.mfid, self.project_id, etc.)

        Example:
            def parse(self):
                # Parse files
                input_file = self.files_to_upload[0]
                metadata = self._parse_file(input_file)

                # Add to instance
                self.add_metadata(metadata)
                self.add_keywords(["domain", "specific"])
                self.thumbnail = self._generate_thumbnail()
        """
            
        # Validate input
        supported_filetypes = ['dm3', 'dm4']

        if not self.files_to_upload:
            raise ValueError("No input files provided")
        
        data_file = os.path.abspath(self.files_to_upload[0])
        dm_version = [ext for ext in supported_filetypes if data_file.endswith(ext)]

        if len(dm_version) == 0:
            raise ValueError("Input file does not have a digital micrograph extension (dm3/dm4)")
        
        self.data_format = dm_version[0]


        # Add Extracted Metadata
        data_file_metadata = self.get_scientific_metadata(data_file)
        self.add_metadata(data_file_metadata)

        # Add DM extracted keywords
        extracted_keywords = self.extract_keywords(data_file_metadata)
        self.add_keywords(extracted_keywords)

        # Update Measurement 
        self.measurement = data_file_metadata.get('Microscope Info.Illumination Mode')

        # Generate thumbnail
        self.thumbnail = self.generate_dm_thumbnail(data_file)



    @staticmethod
    def get_scientific_metadata(data_file):
        try:
            import ncempy.io as nio
        except:
            raise ImportError('The following packages are required for the DM parser to work:\
                               pillow, ncempy.io, py4DSTEM (0.13.17), matplotlib')

        metaData = {}
        with nio.dm.fileDM(data_file, on_memory=True) as dm1:
            # Only keep the most useful tags as meta data
            for kk, ii in dm1.allTags.items():
                # Most useful starting tags
                prefix1 = 'ImageList.{}.ImageTags.'.format(dm1.numObjects)
                prefix2 = 'ImageList.{}.ImageData.'.format(dm1.numObjects)
                pos1 = kk.find(prefix1)
                pos2 = kk.find(prefix2)
                if pos1 > -1:
                    sub = kk[pos1 + len(prefix1):]
                    metaData[sub] = ii
                elif pos2 > -1:
                    sub = kk[pos2 + len(prefix2):]
                    metaData[sub] = ii

                # Remove unneeded keys
                for jj in list(metaData):
                    if jj.find('frame sequence') > -1:
                        del metaData[jj]
                    elif jj.find('Private') > -1:
                        del metaData[jj]
                    elif jj.find('Reference Images') > -1:
                        del metaData[jj]
                    elif jj.find('Frame.Intensity') > -1:
                        del metaData[jj]
                    elif jj.find('Area.Transform') > -1:
                        del metaData[jj]
                    elif jj.find('Parameters.Objects') > -1:
                        del metaData[jj]
                    elif jj.find('Device.Parameters') > -1:
                        del metaData[jj]

            # Store the X and Y pixel size, offset and unit
            try:
                metaData['PhysicalSizeX'] = metaData['Calibrations.Dimension.1.Scale']
                metaData['PhysicalSizeXOrigin'] = metaData['Calibrations.Dimension.1.Origin']
                metaData['PhysicalSizeXUnit'] = metaData['Calibrations.Dimension.1.Units']
                metaData['PhysicalSizeY'] = metaData['Calibrations.Dimension.2.Scale']
                metaData['PhysicalSizeYOrigin'] = metaData['Calibrations.Dimension.2.Origin']
                metaData['PhysicalSizeYUnit'] = metaData['Calibrations.Dimension.2.Units']
            except:
                metaData['PhysicalSizeX'] = 1
                metaData['PhysicalSizeXOrigin'] = 0
                metaData['PhysicalSizeXUnit'] = ''
                metaData['PhysicalSizeY'] = 1
                metaData['PhysicalSizeYOrigin'] = 0
                metaData['PhysicalSizeYUnit'] = ''
            #metaData = dm1.getMetadata(0)
            metaData['FileName'] = data_file
            return metaData
        
    @staticmethod
    def extract_keywords(metadata_dict):
        dm_keywords = [metadata_dict.get(key, None) for key in metadata_dict if 'Mode' in key]
        return dm_keywords

    
    @staticmethod
    def generate_dm_thumbnail(dm_path):
        """Open a DM3/DM4 file, extract a 2D image, and save a thumbnail PNG.
        
        Returns filepath on success, None on failure.
        """
        from crucible.config import get_cache_dir
        import logging

        try:
            from PIL import Image
            import matplotlib.pyplot as plt
            import py4DSTEM as py4d
        except:
            raise ImportError('The following packages are required for the DM parser to work:\
                               pillow, ncempy.io, py4DSTEM (0.13.17), matplotlib')
        
        # Suppress matplotlib's verbose output
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

        # Get cache directory and create thumbnails_upload subdirectory
        cache_dir = get_cache_dir()
        thumbnail_dir = os.path.join(cache_dir, 'thumbnails_upload')
        os.makedirs(thumbnail_dir, exist_ok=True)
        
        # Create file path
        fname = os.path.splitext(os.path.basename(dm_path))[0]
        file_path = os.path.join(thumbnail_dir, f'{fname}.png')
        
        imarr, ndim = read_dm_image(dm_path)
        if ndim < 2:
            return None
        
        fig, ax = py4d.show(imarr, scaling = 'log', returnfig=True)
        fig.savefig(file_path, bbox_inches='tight', pad_inches=0.05, dpi=100)
        plt.close(fig)

        # Resize to thumbnail
        im = Image.open(file_path)
        im.thumbnail((200, 200))
        im = im.convert('RGB')
        im.save(file_path, format='PNG')

        return file_path
        

