import pydicom
import numpy as np
import argparse
import os

view_dict = np.load('view_dictionary.npy', allow_pickle=True).item()
misc_class = 'ö2d'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, help='Path to folder containing DICOM files')
    parser.add_argument('--output_folder', type=str, help='Path where generated dataset will be written to')
    args = parser.parse_args()

    files = load_filenames(args.input_folder)

    for f in files:
        img, label, uid, hr, fps, iid = get_data_from_dicom(f)
        if img is None:
            continue
        img = convert_to_grayscale(img)
        class_id = get_class_id(label)
        file_name = str(uid) + '_iid' + str(iid) + '_cls' + str(class_id) + '_hr' + str(hr) + '_fps' + str(fps) + '.npy'
        folder_name = os.path.join(args.output_folder, str(class_id), '')
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        np.save(os.path.join(folder_name, file_name), img, allow_pickle=True)


def load_filenames(path):
    files = []
    for dirName, _, fileList in os.walk(path):
        for filename in fileList:
            files.append(os.path.join(dirName, filename))
    return files


def get_data_from_dicom(dicom_path):
    data = pydicom.read_file(dicom_path)
    try:
        if hasattr(data, 'pixel_array'):
            img = data.pixel_array
            if img is None:
                raise ValueError('No image data')
        else:
            raise ValueError('No image data')
        label = get_label(data)
        if checkAttr(data, 'PatientID'):
            uid = data.PatientID
        else:
            uid = -1
        if checkAttr(data, 'HeartRate'):
            hr = data.HeartRate
        else:
            hr = -1
        if checkAttr(data, 'CineRate'):
            fps = data.CineRate
        else:
            fps = -1
        if checkAttr(data, 'InstanceNumber'):
            iid = data.InstanceNumber
        else:
            iid = -1
        return img, label, uid, hr, fps, iid
    except ValueError as ve:
        print("Failed reading Dicom file {} with error {}.".format(dicom_path, ve))
        return None, None, None, None, None, None

def get_label(dicom_data):
    label = None
    dicom_tag1 = 'ReferringPhysicianName'
    dicom_tag2 = 'PerformingPhysicianName'

    if checkAttr(dicom_data, dicom_tag1):
        label = str(getattr(dicom_data, dicom_tag1))
    elif checkAttr(dicom_data, dicom_tag2):
        label = str(getattr(dicom_data, dicom_tag2))
    else:
        raise ValueError('No label found')
    return label

def checkAttr(dicom_data, dicom_tag):
    return hasattr(dicom_data, dicom_tag) and (str(getattr(dicom_data, dicom_tag)) != '')

def get_class_id(label):
    label = label.strip()
    if (label in view_dict):
        class_id = view_dict.get(label)
    else:
        class_id = view_dict.get(misc_class)
    return class_id

def convert_to_grayscale(img):
    return np.average(img, -1).astype(np.uint8)


if __name__ == '__main__':
    main()