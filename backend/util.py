# backend/util.py
import string
import numpy as np
# from paddleocr import TextRecognition
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')

# model = TextRecognition()

# Mapping for common OCR mistakes (O↔0, I↔1, S↔5, etc.)
dict_char_to_int = {'O': '0', 'I': '1', 'Z': '2', 'E': '3', 'A': '4', 'S': '5', 'G': '6', 'J': '7', 'B': '8'}
dict_int_to_char = {'0': 'O', '1': 'I', '2': 'Z', '3': 'E', '4': 'A', '5': 'S', '6': 'G', '7': 'J', '8': 'B'}


def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    length = len(text)
    if length != 7 and length != 8:
        print(f"{text} failed length test in license_complies_format")
        return False
    
    if  (text[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[0] in dict_char_to_int.keys()) and \
        (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[0] in dict_char_to_int.keys()) and \
        (text[2] in string.ascii_uppercase or text[2] in dict_int_to_char.keys()) and \
        (text[3] in string.ascii_uppercase or text[3] in dict_int_to_char.keys()) and \
        (text[length-2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[length-2] in dict_char_to_int.keys()) and \
        (text[length-1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[length-1] in dict_char_to_int.keys()):
        return True
    else:
        print(f"{text} failed format test in license_complies_format")
        return False

def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    print(f"{text} has starting formating")
    license_plate_ = ''
    length = len(text)
    if length == 7:
        mapping = {
            0: dict_char_to_int, 1: dict_char_to_int,
            2: dict_int_to_char, 3: dict_int_to_char,
            5: dict_char_to_int, 6: dict_char_to_int
        }
        for j in range(length):
            if j == 4:
                license_plate_ += text[j]
                continue
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]
            # print(f"{license_plate_} is being formatted")
    elif length == 8:
        mapping = {
            0: dict_char_to_int, 1: dict_char_to_int,
            2: dict_int_to_char, 3: dict_int_to_char,
            6: dict_char_to_int, 7: dict_char_to_int
        }
        for j in range(length):
            if j in [4, 5]:
                license_plate_ += text[j]
                continue
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]
            # print(f"{license_plate_} is being formatted")

    print(f"{license_plate_} is formatted")
    return license_plate_

def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    # img_np = np.array(license_plate_crop)
    # result = model.predict(input=img_np, batch_size=1)
    result = ocr.predict(license_plate_crop)
    count = 0
    for res in result:
        if res["rec_texts"]:
            text = res["rec_texts"]
            text = " ".join(text)
            text = text.upper().replace(' ', '')
            text = text.replace('-', '')
            text = text.replace('.', '')
            text = text.replace(',', '')
            scores = res['rec_scores']
            score = float(np.mean(np.array(scores)))
            print(f"Plate: {text} | Score: {score:.3f}")
            if license_complies_format(text):
                return format_license(text), score
            # return text, score
    return None, None


def get_car_deep(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        car_id = vehicle_track_ids[j].track_id
        ltrb = vehicle_track_ids[j].to_ltrb()
        xcar1 = ltrb[0]
        ycar1 = ltrb[1]
        xcar2 = ltrb[2]
        ycar2 = ltrb[3]

        # xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1

def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1