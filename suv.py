import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union
from warnings import warn

import numpy as np
import pydicom
import SimpleITK as sitk  # type: ignore


def convert_dcm_dir_to_suv(dcm_dir: Path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dcm_dir))
    reader.SetFileNames(dicom_names)
    img = reader.Execute()
    ds = pydicom.read_file(dicom_names[0])  # type: ignore
    (weight, injected_dose, decay_time, halflife) = extract_suv_metadata_from_dicom(ds)
    suv_img = calc_suv(img, weight, injected_dose, decay_time, halflife)
    return suv_img


def get_acquisition_date_time(ds):
    if hasattr(ds, "AcquisitionDateTime"):
        acq_time_str = ds.AcquisitionDateTime
    else:
        acq_time_str = ds.AcquisitionDate + ds.AcquisitionTime
    fmtstr = "%Y%m%d%H%M%S"
    if "." in acq_time_str:
        fmtstr += ".%f"
    AcquisitionDateTime = datetime.strptime(acq_time_str, fmtstr)
    return AcquisitionDateTime


def extract_suv_metadata_from_dicom(ds: pydicom.Dataset):
    """extracts variables for calculating SUVs

    Args:
        ds (pydicom.Dataset): dicom datasest with metadata

    Returns:
        tuple of metadata: (
        AcquisitionDateTime, # datetime
        PatientWeight, # float, in kg
        RadionuclideTotalDose, # float, in Bq
        RadiopharmaceuticalStartDateTime, # datetime
        RadionuclideHalfLife, # float, in seconds
    )
    """
    # extract and check metadata
    if "ATTN" not in ds.CorrectedImage:
        raise ValueError("Image is not attenuation corrected")
    if "DECY" not in ds.CorrectedImage:
        raise ValueError("image is not decay corrected")
    if ds.Units != "BQML":
        raise ValueError("Image is not in BQML unites")

    assert (
        len(ds.RadiopharmaceuticalInformationSequence) == 1
    ), f"Cannot handle {len(ds.RadiopharmaceuticalInformationSequence)} radiopharmacauticals"

    radiopharmacauticalInfo = ds.RadiopharmaceuticalInformationSequence[0]
    halflife = float(radiopharmacauticalInfo.RadionuclideHalfLife)  # seconds
    weight = float(ds.PatientWeight)  # in kg
    # Radionuclide Total Dose is NOT corrected for residual dose in syringe, which is ignored here ...
    injected_dose = float(radiopharmacauticalInfo.RadionuclideTotalDose)
    if ds.Modality == "NM":
        injected_dose *= 1e6  # convert to Bq from MBq for NM scans

    AcquisitionDateTime = get_acquisition_date_time(ds)

    if hasattr(radiopharmacauticalInfo, "RadiopharmaceuticalStartDateTime"):
        start_datetime = radiopharmacauticalInfo.RadiopharmaceuticalStartDateTime
    else:  # assume same day as scan acquisition
        start_datetime = (
            ds.AcquisitionDate + radiopharmacauticalInfo.RadiopharmaceuticalStartTime
        )
    fmtstr = "%Y%m%d%H%M%S"
    if "." in start_datetime:
        fmtstr += ".%f"
    RadiopharmaceuticalStartDateTime = datetime.strptime(start_datetime, fmtstr)

    # series_time = ds.SeriesDate + ds.SeriesTime
    # fmtstr = "%Y%m%d%H%M%S"
    # if "." in series_time:
    #     fmtstr += ".%f"
    # SeriesDateTime = datetime.strptime(series_time, fmtstr)

    # if SeriesDateTime > AcquisitionDateTime:
    #     raise ValueError("Series Date must be before Acquisition Date")

    # decay_time = SeriesDateTime - RadiopharmaceuticalStartDateTime # seconds
    if ds.DecayCorrection == "START":
        decay_time = AcquisitionDateTime - RadiopharmaceuticalStartDateTime  # seconds
        if AcquisitionDateTime < RadiopharmaceuticalStartDateTime:
            warn(
                f"AcquisitionDateTime {AcquisitionDateTime} is before RadiopharmaceuticalStartDateTime {RadiopharmaceuticalStartDateTime}, assuming that dates are wrong and correcting based only on times"
            )
            if AcquisitionDateTime.time() < RadiopharmaceuticalStartDateTime.time():
                raise ValueError(
                    f"AcquisitionDateTime {AcquisitionDateTime} is before RadiopharmaceuticalStartDateTime {RadiopharmaceuticalStartDateTime}"
                )
            decay_time = datetime.combine(
                datetime.today(), AcquisitionDateTime.time()
            ) - datetime.combine(
                datetime.today(), RadiopharmaceuticalStartDateTime.time()
            )

    elif ds.DecayCorrection == "ADMIN":
        decay_time = timedelta(0)  # seconds
    else:
        raise ValueError(
            f"image is not decay corrected to {ds.DecayCorrection} is not supported"
        )

    return (
        weight,
        injected_dose,
        decay_time.total_seconds(),
        halflife,
    )


def calc_suv(
    img: Union[np.ndarray, sitk.Image],
    weight: float,
    injected_dose: float,
    decay_time: float,
    halflife: float,
    inplace=False,
) -> Union[np.ndarray, sitk.Image]:
    """Converts PT or SPECT image into SUV units

    Args:
        img (Union[np.ndarray, sitk.Image]): The PT or SPECT image. This image must be attenuation and decay corrected
        weight (float): patient weight in kg
        injected_dose (float): injected dose in Bq
        decay_time (float): decay time in seconds, decay time is time from injection time to decay correction time
        halflife (float): halflife in seconds of radioisotope
        inplace (bool, optional): If true, returns the same 'img' object with new values, otherwise returns a new object of the same class as 'img'. Defaults to False.

    Returns:
        Union[np.ndarray, sitk.Image]: the PT or SPECT image in SUV units
    """
    if weight <= 0.0:
        raise ValueError(f"PatientWeight {weight} is not positive")

    decayed_dose = injected_dose * (2 ** (-decay_time / halflife))
    SUVbwScaleFactor = weight * 1000 / decayed_dose
    if inplace:
        if (isinstance(img, sitk.Image) and img.GetPixelID() != sitk.sitkFloat32) or (
            isinstance(img, np.ndarray) and not np.issubdtype(img.dtype, np.floating)
        ):
            warn(
                "Image is not floating point, it will probably not work to convert in place"
            )
        img *= SUVbwScaleFactor
        return img

    if isinstance(img, sitk.Image) and img.GetPixelID() != sitk.sitkFloat32:
        img = sitk.Cast(img, sitk.sitkFloat32)
    elif isinstance(img, np.ndarray) and not np.issubdtype(img.dtype, np.floating):
        img = img.astype(float)
    suv_img = img * SUVbwScaleFactor
    return suv_img


class BAMF_STARGUIDE_SENSITIVITY:
    # cnt/min/uCi
    peak113kev = 195
    peak208kev = 213
    combo = 408


# The GE Starguide is a special case, since it doesn't currently support the required fields in dicom.
# We have to use some apriori knowledge to convert to suv
def starguide_convert_counts_to_bqml(
    img: sitk.Image,
    ds: pydicom.Dataset,
    camera_sensitivity=408,
    default_scan_duration_ms=239615,
) -> sitk.Image:
    """convets starguide image to bqml units

    Args:
        img (sitk.Image): starguide SPECT image in counts
        ds: pydicom.Dataset, dicom file to extract metadata from
        camera_sensitivity (int, optional): this is specific to each scanner and energy window. Defaults to 408.
        default_scan_duration_ms (int, optional): scan duration can be found in the dicom ImageComments tag, but if that is not availible use this value. Defaults to 239615.

    Returns:
        sitk.Image: The Starguide SPECT image in bqml
    """
    scan_duration_ms = default_scan_duration_ms
    if hasattr(ds, "ImageComments"):
        match = re.search(r"\$ImagingDuration\$(\d*)\$", ds.ImageComments)
        if match is None:
            warn(
                f"Could not find ImageDuration in ImageComments, using default value of {default_scan_duration_ms}"
            )
        else:
            scan_duration_ms = int(match.group(1))
    else:
        warn(
            f"DICOM has no ImageComments tag, using default value of {default_scan_duration_ms} for scan duration"
        )

    # camera_sensitivity units: cnt/min/uCi, This is specific to the scanner, it is not in the dicom
    # scan_duration_ms units milliseconds, this can be extracted from the ImageComments section of the metadata, it is specific to the scan protocol
    scan_duration_minute = scan_duration_ms / 1000 / 60  # convert to min
    uCi_count = 1 / (camera_sensitivity * scan_duration_minute)  # uCi/count
    bq_count = uCi_count * 37000  # Bq/count
    vox_mm3 = np.prod(img.GetSpacing())
    vox_ml = 0.001 * vox_mm3  # mm3 to ml
    bq_count_ml = bq_count / vox_ml
    img = sitk.Cast(img, sitk.sitkFloat32)
    # img_bq = img * bq_count # voxel unit is Bq
    img_bqml = img * bq_count_ml  # voxel unit is Bq/ml

    return img_bqml


def getAcquisitionDateTime(ds: pydicom.Dataset) -> datetime:
    if hasattr(ds, "AcquisitionDateTime"):
        acq_time_str = ds.AcquisitionDateTime
    else:
        acq_time_str = ds.AcquisitionDate + ds.AcquisitionTime
    fmtstr = "%Y%m%d%H%M%S"
    if "." in acq_time_str:
        fmtstr += ".%f"
    return datetime.strptime(acq_time_str, fmtstr)


# RadiopharmaceuticalStartDateTime
def starguide_extract_suv_metadata_from_dicom(
    ds: pydicom.Dataset,
    injection_time: datetime,
):
    """extracts some variables for calculating SUVs for Starguide scanner

    Args:
        ds (pydicom.Dataset): dicom file with metadata of scan
        injection_time (datetime): radiopharmacuetical injection datetime

    Raises:
        ValueError: if scan is not attenuation corrected

    Returns:
        Tuple(str, str): tuple of (patient weight, decay time)
    """
    if "ATTN" not in ds.CorrectedImage:
        raise ValueError("Image is not attenuation corrected")

    weight = float(ds.PatientWeight)  # in kg
    AcquisitionDateTime = getAcquisitionDateTime(ds)

    # starguide data is 'decay corrected' to acquisition time
    decay_time = AcquisitionDateTime - injection_time  # seconds

    return (weight, decay_time.total_seconds())
