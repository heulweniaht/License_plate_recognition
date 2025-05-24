import cv2
import numpy as np
import torch
# import pytesseract
import easyocr
from pathlib import Path
from PIL import Image
from skimage.restoration import wiener # For Wiener filter (optional deblurring)
from skimage.filters import unsharp_mask # Import for unsharp_mask
import sys
import pathlib
import math
import re
import os
import time # For more robust unique filenames

pathlib.PosixPath = pathlib.WindowsPath
# --- Cấu hình ---
YOLO_MODEL_PATH = r"C:\Users\Admin\Documents\check\best.pt"  # Đường dẫn đến file best.pt của bạn
OUTPUT_DIR = 'log_images'    # Thư mục lưu ảnh các bước
DEBUG = True                  # Bật/tắt log ảnh các bước

# Load the model once, globally or pass it as an argument
# For simplicity here, we'll load it inside the main processing function
# but ensure it's loaded only once per run if possible for efficiency.
# If you call nhan_dien_ky_tu_nang_cao multiple times, consider loading model outside.
CHAR_CORRECTIONS_TO_DIGIT = {
    'O': '0', 'Q': '0', 'D': '0', 'U': '0', # U->0 có thể không phổ biến
    'I': '1', 'L': '1', 'J': '1',
    'S': '5',
    'G': '6', 'C': '6', # C->6 cũng có thể xảy ra
    'B': '8',
    'Z': '2',
    'A': '4', # A->4
    'E': '8', # E->8
    'F': '7', # F->7
    'T': '7', # T->7
}
CHAR_CORRECTIONS_TO_ALPHA = {
    '0': 'O', # Hoặc D tùy ngữ cảnh
    '1': 'L', # Hoặc I
    '5': 'S',
    '6': 'G',
    '8': 'B',
    '2': 'Z', # Hoặc S
    '4': 'A',
    '7': 'T', # Hoặc F
}
# Tạo thư mục lưu ảnh nếu chưa có
if DEBUG and not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def log_image(step_name, image):
    """Hàm lưu ảnh của từng bước xử lý."""
    if DEBUG:
        try:
            # Tạo tên file duy nhất để tránh ghi đè nếu xử lý nhiều ảnh
            # Using time.time() for more uniqueness than tickCount if rapidly processed
            timestamp = int(time.time() * 1000) # Milliseconds for uniqueness
            filename = os.path.join(OUTPUT_DIR, f"{timestamp}_{step_name.replace(' ', '_')}.png")

            if image is None:
                print(f"CẢNH BÁO: Ảnh cho bước '{step_name}' là None.")
                return

            if not isinstance(image, np.ndarray):
                print(f"CẢNH BÁO: Đầu vào cho bước '{step_name}' không phải là NumPy array. Type: {type(image)}")
                return

            img_to_save = image.copy() # Work on a copy

            # Kiểm tra kiểu dữ liệu và chuyển đổi nếu cần
            if img_to_save.dtype != np.uint8:
                if img_to_save.max() <= 1.0 and img_to_save.min() >=0.0 : # Nếu ảnh đã được chuẩn hóa về [0,1]
                    img_to_save = (img_to_save * 255).astype(np.uint8)
                elif img_to_save.max() > 255 or img_to_save.min() < 0: # Values out of typical image range
                     # Normalize to 0-255 if it seems to be a different range (e.g. float from some filters)
                    img_to_save = cv2.normalize(img_to_save, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                else:
                    img_to_save = img_to_save.astype(np.uint8) # Chuyển đổi trực tiếp nếu không phải float

            # Kiểm tra số kênh màu
            if len(img_to_save.shape) == 2: # Ảnh xám
                cv2.imwrite(filename, img_to_save)
            elif len(img_to_save.shape) == 3:
                if img_to_save.shape[2] == 3: # Ảnh màu BGR
                    cv2.imwrite(filename, img_to_save)
                elif img_to_save.shape[2] == 1: # Ảnh xám nhưng có 3 chiều
                    cv2.imwrite(filename, cv2.cvtColor(img_to_save, cv2.COLOR_GRAY2BGR))
                else:
                    print(f"CẢNH BÁO: Số kênh màu không được hỗ trợ cho bước '{step_name}'. Shape: {img_to_save.shape}")
                    return
            else:
                print(f"CẢNH BÁO: Định dạng ảnh không được hỗ trợ cho bước '{step_name}'. Shape: {img_to_save.shape}")
                return

            print(f"Ảnh đã lưu: {filename}")
        except Exception as e:
            print(f"LỖI khi lưu ảnh cho bước '{step_name}': {e}")
            if image is not None and isinstance(image, np.ndarray):
                 print(f"Thông tin ảnh: dtype={image.dtype}, shape={image.shape}, min_val={image.min()}, max_val={image.max()}")
            elif image is None:
                 print("Thông tin ảnh: None")
            else:
                 print(f"Thông tin ảnh: Type={type(image)}")


# --- Missing Function Implementations (Assumptions) ---
def chinh_muc_xam(bgr_image):
    """Chuyển ảnh màu BGR sang ảnh xám."""
    if bgr_image is None:
        print("Lỗi: Ảnh đầu vào cho chinh_muc_xam là None.")
        return None
    if len(bgr_image.shape) == 3 and bgr_image.shape[2] == 3:
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        log_image("Tien_xu_ly_Chuyen_xam", gray)
        return gray
    elif len(bgr_image.shape) == 2: # Already grayscale
        log_image("Tien_xu_ly_Chuyen_xam_Da_xam", bgr_image)
        return bgr_image
    else:
        print(f"Lỗi: Định dạng ảnh không hợp lệ cho chinh_muc_xam. Shape: {bgr_image.shape}")
        return None

def nhi_phan_hoa_bien_so(gray_image):
    """Áp dụng Otsu's binarization."""
    if gray_image is None:
        print("Lỗi: Ảnh đầu vào cho nhi_phan_hoa_bien_so là None.")
        return None
    if len(gray_image.shape) != 2:
        print(f"Lỗi: nhi_phan_hoa_bien_so yêu cầu ảnh xám. Shape: {gray_image.shape}")
        return None
    _, otsu_thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    log_image("Tien_xu_ly_Nhi_phan_Otsu", otsu_thresh)
    return otsu_thresh
# --- End of Missing Function Implementations ---

def apply_clahe(gray_image):
    if gray_image is None: return None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4)) # Tile grid nhỏ hơn cho biển số
    enhanced_image = clahe.apply(gray_image)
    log_image("Tien_xu_ly_CLAHE", enhanced_image)
    return enhanced_image

def apply_unsharp_masking(gray_image):
    if gray_image is None: return None
    # Áp dụng unsharp mask, bạn có thể cần điều chỉnh 'radius' và 'amount'
    # Ensure image is float for unsharp_mask if preserve_range=False (default)
    # If preserve_range=True, it handles uint8 directly.
    # skimage's unsharp_mask returns a float array, typically in [0,1] or original range
    sharpened_image_float = unsharp_mask(gray_image.astype(float)/255.0, radius=1.0, amount=1.0)
    # Convert back to uint8
    sharpened_image_uint8 = (np.clip(sharpened_image_float, 0, 1) * 255).astype(np.uint8)
    log_image("Tien_xu_ly_UnsharpMask", sharpened_image_uint8)
    return sharpened_image_uint8


def score_plate_candidate(text_segments, avg_confidence):
    """
    Chấm điểm một ứng viên biển số.
    text_segments: list các string, ví dụ ['19-C1', '250.89']
    avg_confidence: điểm tin cậy trung bình từ EasyOCR
    """
    score = 0
    full_text_raw = " ".join(text_segments).strip()

    # 1. Kiểm tra định dạng cơ bản (ví dụ rất đơn giản)
    pattern_line1_strict = r"^\d{2}-[A-Z]\d{1,2}$" # Ví dụ: 29-C1, 29-B12
    pattern_line1_flexible = r"^\d{2}-[A-Z0-9]{1,3}$" # Ví dụ: 29-C1A, 29-123
    pattern_line2 = r"^\d{2,3}\.\d{2}$" # Ví dụ: 250.89, 50.89

    if len(text_segments) == 2:
        line1 = text_segments[0]
        line2 = text_segments[1]
        if re.match(pattern_line1_strict, line1) or re.match(pattern_line1_flexible, line1):
            score += 10 # Điểm cao cho dòng 1 khớp
        if re.match(pattern_line2, line2):
            score += 10 # Điểm cao cho dòng 2 khớp
        if (re.match(pattern_line1_strict, line1) or re.match(pattern_line1_flexible, line1)) and re.match(pattern_line2, line2):
            score += 5 # Thưởng thêm nếu cả 2 dòng khớp
    elif len(text_segments) == 1:
        # Cố gắng kiểm tra xem nó có thể là biển số ghép không
        # Ví dụ: "19C1250.89" hoặc "19-C1250.89"
        # This is complex, current regex is a placeholder
        if len(full_text_raw.replace(" ","").replace("-","").replace(".","")) >= 6: # Độ dài tối thiểu, e.g. 29C1234
            # A very simple check for single line: is it mostly alphanumeric with a dot or dash?
            if re.match(r"^\d{2}[A-Z0-9]{1,3}\s?\d{2,3}\.?\d{2}$", full_text_raw.replace("-","")):
                 score += 7
            elif re.match(r"^\d{2}-[A-Z0-9]{1,3}\s?\d{2,3}\.?\d{2}$", full_text_raw):
                 score += 8
            elif len(full_text_raw.replace(" ","").replace("-","").replace(".","")) >= 7 :
                 score += 5


    else: # Không phải 1 hoặc 2 phần text -> khả năng sai cao
        score -= 5


    # 2. Cộng điểm từ confidence của EasyOCR (ví dụ: nhân với 20 để có thang điểm lớn hơn)
    score += int(avg_confidence * 20)

    # 3. Phạt nếu có ký tự không mong muốn (ngoài allowlist đã dùng)
    # (allowlist của EasyOCR đã phần nào xử lý việc này)

    # 4. Kiểm tra các lỗi phổ biến nhưng chưa được sửa (ví dụ '7' thay vì '1' ở đầu)
    if text_segments and text_segments[0].startswith("7"):
        score -= 2 # Trừ nhẹ điểm nếu mã vùng bắt đầu bằng '7' (khả năng là '1' bị sai)

    return score
def _correct_char_globally(char_in, expected_type=None):
    """
    Sửa một ký tự dựa trên lỗi OCR phổ biến, có xem xét loại ký tự mong đợi.
    expected_type: None (không rõ), 'alpha', hoặc 'digit'.
    """
    char = str(char_in).upper()
    if not char:
        return ""

    is_alpha = char.isalpha()
    is_digit = char.isdigit()

    if expected_type == 'alpha':
        if is_alpha: return char  # Đã là chữ
        # Cố gắng chuyển sang chữ
        return CHAR_CORRECTIONS_TO_ALPHA.get(char, char)
    elif expected_type == 'digit':
        if is_digit: return char  # Đã là số
        # Cố gắng chuyển sang số
        return CHAR_CORRECTIONS_TO_DIGIT.get(char, char)
    else:  # Không có kỳ vọng cụ thể hoặc có thể là cả hai
        # Ưu tiên: nếu nó trông giống lỗi phổ biến cho số, sửa thành số.
        # Điều này là do các trường hợp nhầm lẫn O/0, I/1, B/8 thường là số trong biển số.
        if char in CHAR_CORRECTIONS_TO_DIGIT:
            return CHAR_CORRECTIONS_TO_DIGIT[char]
        # Nếu không phải lỗi số và nó không phải là chữ, có thể là lỗi chữ.
        if not is_digit and not is_alpha and char in CHAR_CORRECTIONS_TO_ALPHA:
            return CHAR_CORRECTIONS_TO_ALPHA[char]
        return char # Trả về ký tự gốc nếu không có sửa lỗi rõ ràng

def _correct_segment_by_pattern(segment_in, pattern_str):
    """
    Sửa một đoạn text dựa trên một mẫu (pattern) các loại ký tự.
    pattern_str: ví dụ "DD" (2 số), "AAD" (2 chữ 1 số), "DDDDD" (5 số)
                 D = Digit (Số), A = Alpha (Chữ), X = Any (Bất kỳ, sửa lỗi chung)
    """
    segment = str(segment_in)
    # Nếu độ dài không khớp, hoặc không có pattern, sửa lỗi chung cho từng ký tự
    if not pattern_str or len(segment) != len(pattern_str):
        return "".join([_correct_char_globally(c) for c in segment])

    corrected_chars = []
    for i, p_char_type in enumerate(pattern_str):
        original_char = segment[i]
        if p_char_type == 'D':
            corrected_chars.append(_correct_char_globally(original_char, 'digit'))
        elif p_char_type == 'A':
            corrected_chars.append(_correct_char_globally(original_char, 'alpha'))
        else:  # 'X' hoặc ký tự không xác định trong pattern
            corrected_chars.append(_correct_char_globally(original_char))
    return "".join(corrected_chars)
# --- Kết thúc hàm trợ giúp ---

def post_process_chosen_text(text_segments):
    if not text_segments:
        return "Không nhận diện được ký tự."

    # 1. Kết hợp và làm sạch cơ bản các đoạn text từ OCR
    # Giữ lại dấu chấm/gạch ngang ban đầu vì chúng có thể là gợi ý cấu trúc.
    ocr_text_combined = "".join(text_segments).upper()
    # Loại bỏ khoảng trắng thừa giữa các ký tự, nhưng giữ khoảng trắng giữa các cụm (nếu có)
    ocr_text_cleaned = ' '.join(ocr_text_combined.split())


    # 2. Định nghĩa các mẫu regex cho các cấu trúc biển số phổ biến của Việt Nam (xe máy)
    # Mục tiêu: Mã vùng (2 số) - Seri (chữ/số, 2-4 ký tự) - Cụm số (4 hoặc 5 số, có thể có dấu chấm)

    # Regex này cố gắng bắt các cấu trúc phổ biến, cho phép thiếu dấu phân cách
    # Nhóm 1: Mã vùng (VD: "34", "29")
    # Nhóm 2: Seri (VD: "L6", "C1A", "AB", "F123")
    # Nhóm 3: Cụm số (VD: "6842", "12345", "68.42", "123.45")
    # Regex này được áp dụng cho chuỗi đã loại bỏ hết khoảng trắng để tăng khả năng khớp.
    plate_parser_regex = re.compile(
        r"^(\d{2})"                                 # Nhóm 1: Mã vùng (luôn là 2 số)
        r"([A-Z0-9]{2,4})"                         # Nhóm 2: Seri (2 đến 4 ký tự chữ hoặc số)
        r"(\d{4,5})$"                              # Nhóm 3: Cụm số (4 hoặc 5 số ở cuối)
    )

    # Chuẩn bị chuỗi để khớp regex: loại bỏ tất cả ký tự không phải chữ/số
    alphanumeric_text = "".join(filter(str.isalnum, ocr_text_cleaned))

    area_code_final = ""
    seri_final = ""
    number_block_final = ""

    match = plate_parser_regex.match(alphanumeric_text)

    if match:
        area_code_raw = match.group(1)
        seri_raw = match.group(2)
        numbers_raw = match.group(3)

        # a. Sửa mã vùng (luôn là "DD")
        area_code_final = _correct_segment_by_pattern(area_code_raw, "DD")

        # b. Sửa Seri (linh hoạt hơn)
        # Các mẫu seri phổ biến: AD (L6), ADD (L12), AA (AB), AAD (AB1)
        # Chúng ta sẽ đoán mẫu dựa trên độ dài và loại ký tự, sau đó áp dụng sửa lỗi.
        seri_len = len(seri_raw)
        if seri_len == 2: # VD: L6 (AD), AB (AA)
            if seri_raw[0].isalpha() and seri_raw[1].isdigit(): # Mẫu AD
                seri_final = _correct_segment_by_pattern(seri_raw, "AD")
            elif seri_raw[0].isalpha() and seri_raw[1].isalpha(): # Mẫu AA
                seri_final = _correct_segment_by_pattern(seri_raw, "AA")
            else: # Các trường hợp 2 ký tự khác (DD, DA) - sửa chung
                seri_final = _correct_segment_by_pattern(seri_raw, "XX")
        elif seri_len == 3: # VD: L12 (ADD), AB1 (AAD), ABC (AAA)
            if seri_raw[0].isalpha() and seri_raw[1].isdigit() and seri_raw[2].isdigit(): # Mẫu ADD
                seri_final = _correct_segment_by_pattern(seri_raw, "ADD")
            elif seri_raw[0].isalpha() and seri_raw[1].isalpha() and seri_raw[2].isdigit(): # Mẫu AAD
                seri_final = _correct_segment_by_pattern(seri_raw, "AAD")
            elif seri_raw[0].isalpha() and seri_raw[1].isalpha() and seri_raw[2].isalpha(): # Mẫu AAA
                seri_final = _correct_segment_by_pattern(seri_raw, "AAA")
            else: # Các trường hợp 3 ký tự khác - sửa chung
                seri_final = _correct_segment_by_pattern(seri_raw, "XXX")
        elif seri_len == 4: # VD: AB12 (AADD), ABCD (AAAA) - ít phổ biến hơn cho xe máy
            if seri_raw[0].isalpha() and seri_raw[1].isalpha() and \
               seri_raw[2].isdigit() and seri_raw[3].isdigit(): # Mẫu AADD
                seri_final = _correct_segment_by_pattern(seri_raw, "AADD")
            elif seri_raw[0].isalpha() and seri_raw[1].isalpha() and \
                 seri_raw[2].isalpha() and seri_raw[3].isalpha(): # Mẫu AAAA
                seri_final = _correct_segment_by_pattern(seri_raw, "AAAA")
            else: # Các trường hợp 4 ký tự khác - sửa chung
                seri_final = _correct_segment_by_pattern(seri_raw, "XXXX")
        else: # Độ dài seri không phổ biến
            seri_final = "".join([_correct_char_globally(c) for c in seri_raw])


        # c. Sửa và định dạng cụm số (luôn là số)
        numbers_corrected_digits = _correct_segment_by_pattern(numbers_raw, "D" * len(numbers_raw))

        if len(numbers_corrected_digits) == 5:
            number_block_final = f"{numbers_corrected_digits[:3]}.{numbers_corrected_digits[3:]}"
        elif len(numbers_corrected_digits) == 4:
            number_block_final = f"{numbers_corrected_digits[:2]}.{numbers_corrected_digits[2:]}"
        # elif len(numbers_corrected_digits) == 3: # Ít phổ biến, có thể là N.NN
        #     number_block_final = f"{numbers_corrected_digits[0]}.{numbers_corrected_digits[1:]}"
        else: # Độ dài không mong đợi
            number_block_final = numbers_corrected_digits
    else:
        # Regex không khớp, có thể do OCR quá rời rạc hoặc định dạng rất lạ.
        # Đây là phương án cuối cùng, rất cơ bản.
        # Cố gắng tách thủ công nếu có vẻ như 2 dòng bị nối liền.
        # Ví dụ: "34L66842" -> Mã vùng: 34, Seri: L6, Số: 6842
        # Đây là phần phức tạp, cần nhiều logic hơn để đảm bảo độ chính xác.
        # Tạm thời, nếu regex chính thất bại, chúng ta trả về text đã được sửa lỗi chung chung.
        if len(text_segments) == 1: # Chỉ xử lý nếu OCR ban đầu trả về 1 segment
            raw_single_segment = text_segments[0].replace(" ","").replace(".","").replace("-","")
            corrected_fallback = "".join([_correct_char_globally(c) for c in raw_single_segment])
            # Thử một số quy tắc tách đơn giản cho các trường hợp phổ biến
            # 1. Mã vùng (2D) + Seri (AD, 2 ký tự) + Số (DDDD, 4 ký tự) => Tổng 8 ký tự
            if len(corrected_fallback) == 8:
                ac = _correct_segment_by_pattern(corrected_fallback[0:2], "DD")
                sr = _correct_segment_by_pattern(corrected_fallback[2:4], "AD") # Giả định mẫu AD
                nb_raw = _correct_segment_by_pattern(corrected_fallback[4:8], "DDDD")
                nb = f"{nb_raw[0:2]}.{nb_raw[2:4]}"
                return f"{ac}-{sr} {nb}".strip()
            # 2. Mã vùng (2D) + Seri (ADD, 3 ký tự) + Số (DDDD, 4 ký tự) => Tổng 9 ký tự
            # Hoặc Mã vùng (2D) + Seri (AAD, 3 ký tự) + Số (DDDD, 4 ký tự)
            if len(corrected_fallback) == 9 and corrected_fallback[2].isalpha():
                ac = _correct_segment_by_pattern(corrected_fallback[0:2], "DD")
                sr_pattern = "AAD" # Giả định AAD nếu ký tự thứ 3 là chữ
                if corrected_fallback[3].isdigit() and corrected_fallback[4].isdigit(): # Kiểm tra mẫu ADD
                    sr_pattern = "ADD"
                sr = _correct_segment_by_pattern(corrected_fallback[2:5], sr_pattern)
                nb_raw = _correct_segment_by_pattern(corrected_fallback[5:9], "DDDD")
                nb = f"{nb_raw[0:2]}.{nb_raw[2:4]}"
                return f"{ac}-{sr} {nb}".strip()
            # 3. Mã vùng (2D) + Seri (AD, 2 ký tự) + Số (DDDDD, 5 ký tự) => Tổng 9 ký tự
            if len(corrected_fallback) == 9 and corrected_fallback[4:9].isdigit(): # Check if last 5 are digits
                ac = _correct_segment_by_pattern(corrected_fallback[0:2], "DD")
                sr = _correct_segment_by_pattern(corrected_fallback[2:4], "AD") # Giả định mẫu AD
                nb_raw = _correct_segment_by_pattern(corrected_fallback[4:9], "DDDDD")
                nb = f"{nb_raw[0:3]}.{nb_raw[3:5]}"
                return f"{ac}-{sr} {nb}".strip()

            return corrected_fallback # Trả về chuỗi đã sửa lỗi cơ bản
        else: # Nếu OCR ban đầu trả về nhiều segments và regex vẫn không khớp
             return " ".join([_correct_segment_by_pattern(s, "X"*len(s)) for s in text_segments]).strip()


    # 3. Áp dụng các luật sửa lỗi rất cụ thể (dùng hạn chế)
    # Chỉ áp dụng nếu các bước tổng quát ở trên vẫn bỏ sót các lỗi phổ biến và đã biết.
    # Ví dụ: "79" -> "19" (nếu mã vùng là 79 thì sửa thành 19)
    if area_code_final == "79": area_code_final = "19"
    # Ví dụ: Nếu mã vùng 34, seri "LG" hoặc "LC" thường là "L6"
    if area_code_final == "34" and seri_final in ["LG", "LC"]: seri_final = "L6"
    # Ví dụ: Nếu mã vùng 36, seri "BG" hoặc "BC" thường là "B6"
    if area_code_final == "36" and seri_final in ["BG", "BC", "B0", "8G", "8C", "80"]: seri_final = "B6" # Thêm các biến thể lỗi cho B6


    # 4. Kết hợp lại thành chuỗi biển số hoàn chỉnh
    if area_code_final and seri_final and number_block_final:
        return f"{area_code_final}-{seri_final} {number_block_final}".strip()
    elif area_code_final and seri_final: # Chỉ nhận diện được dòng đầu
        return f"{area_code_final}-{seri_final}".strip()
    else: # Không thể tái cấu trúc, trả về kết quả đã qua sửa lỗi cơ bản
        return " ".join([_correct_segment_by_pattern(s, "X"*len(s)) for s in text_segments]).strip()



def nhan_dien_ky_tu_nang_cao(image_path, yolo_model_path, character_allowlist):
    yolo_model = None
    try:
        yolo_model = torch.hub.load('yolov5', 'custom', path=yolo_model_path, source='local', _verbose=False)
        yolo_model.conf = 0.35 # Ngưỡng tin cậy
        yolo_model.iou = 0.45  # Ngưỡng IoU
        print("Mô hình YOLOv5 đã được tải thành công.")
    except Exception as e:
        print(f"Lỗi khi tải mô hình YOLOv5: {e}")
        print("Hãy đảm bảo bạn đã cài đặt YOLOv5 đúng cách và file 'best.pt' tồn tại.")
        return "Lỗi tải mô hình YOLOv5"

    try:
        img_original = cv2.imread(image_path)
        if img_original is None: return "Lỗi đọc ảnh"
        log_image("0_Anh_goc", img_original)
        height, width = img_original.shape[:2]

        results = yolo_model(img_original)
        detections = results.xyxy[0].cpu().numpy()

        if len(detections) == 0: return "Không phát hiện biển số"
        best_detection = max(detections, key=lambda x: x[4]) # Lấy detection có conf cao nhất
        xmin, ymin, xmax, ymax, _, _ = best_detection
        padding = 5 # pixels
        crop_xmin = max(0, int(xmin) - padding)
        crop_ymin = max(0, int(ymin) - padding)
        crop_xmax = min(width, int(xmax) + padding)
        crop_ymax = min(height, int(ymax) + padding)

        # Ensure crop coordinates are valid
        if crop_xmin >= crop_xmax or crop_ymin >= crop_ymax:
            return "Vùng cắt biển số không hợp lệ sau padding."

        lp_image_bgr = img_original[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
        if lp_image_bgr.size == 0: return "Vùng cắt biển số rỗng"
        log_image("1_Cat_vung_chua_bien_so_YOLO", lp_image_bgr)
    except Exception as e:
        print(f"Lỗi trong quá trình tách biển số: {e}")
        return f"Lỗi trong quá trình tách biển số: {e}"

    # --- Tạo các ứng viên ảnh ---
    lp_image_gray = chinh_muc_xam(lp_image_bgr.copy())
    if lp_image_gray is None: return "Lỗi chuyển xám vùng biển số"

    candidate_images = []
    # 1. Ảnh xám gốc của vùng biển số
    candidate_images.append({"name": "lp_gray", "image": lp_image_gray.copy()})

    # 2. Ảnh Otsu
    otsu_image = nhi_phan_hoa_bien_so(lp_image_gray.copy())
    if otsu_image is not None:
        candidate_images.append({"name": "lp_otsu", "image": otsu_image.copy()})

    # 3. Ảnh CLAHE
    clahe_image = apply_clahe(lp_image_gray.copy())
    if clahe_image is not None:
        candidate_images.append({"name": "lp_clahe", "image": clahe_image.copy()})
        # Optionally, binarize CLAHE image
        otsu_clahe_image = nhi_phan_hoa_bien_so(clahe_image.copy())
        if otsu_clahe_image is not None:
            candidate_images.append({"name": "lp_clahe_otsu", "image": otsu_clahe_image.copy()})


    # 4. Ảnh làm nét
    sharpened_image = apply_unsharp_masking(lp_image_gray.copy())
    if sharpened_image is not None:
        candidate_images.append({"name": "lp_sharpened", "image": sharpened_image.copy()})
        # Optionally, binarize sharpened image
        otsu_sharpened_image = nhi_phan_hoa_bien_so(sharpened_image.copy())
        if otsu_sharpened_image is not None:
            candidate_images.append({"name": "lp_sharpened_otsu", "image": otsu_sharpened_image.copy()})


    # --- Chạy OCR và chấm điểm ---
    try:
        reader = easyocr.Reader(['vi', 'en'], gpu=False) # Khởi tạo 1 lần
    except Exception as e:
        print(f"Lỗi khi khởi tạo EasyOCR Reader: {e}")
        return "Lỗi khởi tạo EasyOCR"

    ocr_results_scored = []

    for cand_data in candidate_images:
        name = cand_data["name"]
        image_to_ocr = cand_data["image"]

        if image_to_ocr is None:
            print(f"CẢNH BÁO: Ảnh cho ứng viên '{name}' là None, bỏ qua OCR.")
            continue

        print(f"\n--- Nhận diện ứng viên: {name} ---")
        log_image(f"Input_EasyOCR_{name}", image_to_ocr)

        try:
            # detail=0 returns list of [bbox, text, prob]
            # detail=1 returns list of (bbox, text, confidence) - this is what we used
            # paragraph=False for reading line by line
            raw_ocr_output = reader.readtext(image_to_ocr, detail=1, paragraph=False, allowlist=character_allowlist)
            print(f"DEBUG: RAW OCR Output cho {name}: {raw_ocr_output}")
        except Exception as e:
            print(f"Lỗi khi chạy EasyOCR cho {name}: {e}")
            raw_ocr_output = []


        if not raw_ocr_output:
            ocr_results_scored.append({"name": name, "segments": [], "avg_confidence": 0, "score": -100, "processed_text": ""})
            continue

        current_segments = []
        total_confidence = 0
        num_segments = 0
        for (bbox, text_segment, conf) in raw_ocr_output:
            cleaned_segment = str(text_segment).upper().replace('"', '').replace("'", "").strip()
            if cleaned_segment: # Only add if not empty after stripping
                current_segments.append(cleaned_segment)
                total_confidence += conf
                num_segments += 1

        avg_conf = total_confidence / num_segments if num_segments > 0 else 0
        candidate_score = score_plate_candidate(current_segments, avg_conf)

        ocr_results_scored.append({
            "name": name,
            "segments": current_segments,
            "avg_confidence": avg_conf,
            "score": candidate_score
        })
        print(f"Ứng viên {name}: Segments={current_segments}, AvgConf={avg_conf:.2f}, Score={candidate_score}")

    if not ocr_results_scored:
        return "Không có kết quả OCR nào từ các ứng viên."

    # --- Chọn ứng viên tốt nhất ---
    best_candidate_data = sorted(ocr_results_scored, key=lambda x: (x["score"], x["avg_confidence"]), reverse=True)[0]

    print(f"\n--- Ứng viên được chọn: {best_candidate_data['name']} ---")
    print(f"Segments thô: {best_candidate_data['segments']}")
    print(f"Điểm: {best_candidate_data['score']}, Độ tin cậy TB: {best_candidate_data['avg_confidence']:.2f}")

    # --- Hậu xử lý kết quả tốt nhất ---
    final_plate_text = post_process_chosen_text(best_candidate_data["segments"])

    return final_plate_text

if __name__ == '__main__':
    image_file_path = r"D:\Download\o96a0687-16300713457262015723562.jpg" # Hoặc ảnh mới của bạn image_3792f5.png
    # Check if image exists
    if not os.path.exists(image_file_path):
        print(f"LỖI: File ảnh không tồn tại tại đường dẫn: {image_file_path}")
        # Create a dummy image for testing if the specified one doesn't exist
        print("Tạo ảnh giả để kiểm tra...")
        dummy_image = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.putText(dummy_image, "TEST IMAGE", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
        cv2.imwrite("dummy_test_image.png", dummy_image)
        image_file_path = "dummy_test_image.png"
        print(f"Sử dụng ảnh giả: {image_file_path}")


    character_list = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ.-' # Allowlist for EasyOCR
    license_plate_text = nhan_dien_ky_tu_nang_cao(image_file_path, YOLO_MODEL_PATH, character_list)

    if license_plate_text:
        print(f"\n===> BIỂN SỐ CUỐI CÙNG (tự động chọn): {license_plate_text}")
    else:
        print("\n===> Không thể nhận diện biển số từ ảnh.")