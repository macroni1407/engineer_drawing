from paddleocr import PaddleOCRVL

ocr = None

def get_ocr():

    global ocr

    if ocr is None:
        ocr = PaddleOCRVL()

    return ocr