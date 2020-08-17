import mmcv
import numpy as np
import pandas as pd
import os
from glob import glob
from .builder import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class MyDataset(BaseDataset):

    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_dir = '/data/Jack12/skin_lesion/'
        all_image_path = glob(os.path.join(data_dir, '*', '*.jpg'))
        imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
        lesion_type_dict = {
            'nv': 'Melanocytic nevi',
            'mel': 'dermatofibroma',
            'bkl': 'Benign keratosis-like lesions ',
            'bcc': 'Basal cell carcinoma',
            'akiec': 'Actinic keratoses',
            'vasc': 'Vascular lesions',
            'df': 'Dermatofibroma'
        }

        data_infos = []

        df_original = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata.csv'))
        df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
        df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
        df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes

        df_original['duplicates'].value_counts()
        for index in range(len(df_original)):
            X = df_original['path']
            y = df_original['cell_type_idx']

            if self.transform:
                X = self.transform(X)

            data_infos.append((X, y))
        return X, y