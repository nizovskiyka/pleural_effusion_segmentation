from typing import List
import glob
import os
import sys
import tempfile
import zipfile
import scipy
import nibabel as nib
import SimpleITK as sitk
import numpy as np

class Preprocessor:
    '''Prepeocessing .dcm and .nii.gz 3D files to .npy 2D'''

    def load_dicom(self, directory: str) -> np.ndarray:
        '''Load dicom from an input dir as a 3D np array;
        directory arg shall contain .dcm files inside.'''
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(directory)
        reader.SetFileNames(dicom_names)
        image_itk = reader.Execute()
        image_zyx = sitk.GetArrayFromImage(image_itk).astype(np.int16)
        return image_zyx

    def load_mask(self, mask_path: str) -> np.ndarray:
        '''Load mask from an input path as a 3D np array;
        mask_path arg shall point to a single .nii file.'''
        mask = nib.load(mask_path)
        mask = mask.get_fdata().transpose(2, 0, 1)
        mask = scipy.ndimage.rotate(mask, 90, (1, 2))
        return mask

    def get_img_paths(self, img_dir: str) -> List[str]:
        '''Get list of all unique dicom study dirs thar are subdirs of img_dir'''
        dicom_dirs = set()
        for root, _, files in os.walk(img_dir):
            for item in files:
                if item.endswith(".dcm"):
                    dicom_dirs.add(root)
        return list(dicom_dirs)

    def img_path2mask_path(self, img_path: str, masks_path: str) -> str:
        '''Get mask path for current study, based on the dataset structure.'''
        img_list = img_path.split(os.path.sep)
        masks_list = masks_path.split(os.path.sep)
        img_list[:-4] = masks_list
        del img_list[-3:]
        img_path = "/".join(img_list)
        return glob.glob(f"{img_path}/*")[0]

    def process_dataset(self, input_dir: str, output_dir: str) -> None:
        '''Convert raw data to numpy horizontal slices for 2D training'''

        with tempfile.TemporaryDirectory() as tmpdir, zipfile.ZipFile(input_dir, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
            unzipped_path = os.path.join(tmpdir, "unzipped")
            img_zip_path = os.path.join(tmpdir, "subset", "subset_img.zip")
            mask_zip_path = os.path.join(tmpdir, "subset", "subset_masks.zip")
            os.mkdir(unzipped_path)
            with zipfile.ZipFile(img_zip_path, 'r') as zip_ref2:
                zip_ref2.extractall(unzipped_path)
            with zipfile.ZipFile(mask_zip_path, 'r') as zip_ref2:
                zip_ref2.extractall(unzipped_path)
            imgs_dir = os.path.join(unzipped_path, "subset")
            masks_dir = os.path.join(unzipped_path, "subset_masks")
            image_paths = self.get_img_paths(imgs_dir)
            if not os.path.exists(os.path.join(output_dir, "images")):
                os.makedirs(os.path.join(output_dir, "images"))
            if not os.path.exists(os.path.join(output_dir, "masks")):
                os.makedirs(os.path.join(output_dir, "masks"))
            for sample_path in image_paths:
                mask_path = self.img_path2mask_path(sample_path, masks_dir)
                dicom_data = self.load_dicom(sample_path)
                mask_data = self.load_mask(mask_path)
                filename = mask_path.split(os.path.sep)[6][:-7]
                for i in range(dicom_data.shape[0]):
                    img_slice = dicom_data[i, :, :]
                    mask_slice = mask_data[i, :, :]
                    np.save(os.path.join(output_dir, f"images/{filename}-{i}.npy"), img_slice)
                    np.save(os.path.join(output_dir, f"masks/{filename}-{i}.npy"), mask_slice)

if __name__ == "__main__":
    # INPUT_PATH = sys.argv[1]
    # OUTPUT_PATH = sys.argv[2]
    INPUT_PATH = "/raw_data/subset.zip"
    OUTPUT_PATH = "/npy_dataset"
    if not os.path.exists(os.path.join(OUTPUT_PATH)):
        os.makedirs(OUTPUT_PATH)
    pr = Preprocessor()
    pr.process_dataset(INPUT_PATH, OUTPUT_PATH)
