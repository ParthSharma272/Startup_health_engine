import os
from typing import Optional, Union
from src.utils.logger_config import logger

# Import OCR and Image/PDF processing libraries
try:
    import pytesseract
    from PIL import Image
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfinterp import PDFResourceManager
    from pdfminer.converter import TextConverter
    from pdfminer.layout import LAParams
    from io import StringIO
except ImportError as e:
    logger.error(f"Missing required OCR/PDF libraries: {e}. Please install them using 'pip install -r requirements.txt'")
    # Define dummy functions or raise an error if libraries are critical
    pytesseract = None
    Image = None
    pdfminer_extract_text = None
    PDFPage = None
    PDFResourceManager = None
    # PDFPageInterpreter = None # Not directly used in final extract_text_from_pdf
    TextConverter = None
    LAParams = None
    StringIO = None


class KPIExtractor: # Renaming to DocumentTextExtractor might be more accurate for this phase

    def __init__(self):

        if pytesseract is None:
            logger.critical("pytesseract library not found. OCR functionality will be disabled.")
            self.tesseract_available = False
        else:
            try:
                # Check if Tesseract executable is in PATH
                pytesseract.get_tesseract_version()
                self.tesseract_available = True
                logger.info(f"Tesseract OCR engine found (version: {pytesseract.get_tesseract_version()}).")
            except pytesseract.TesseractNotFoundError:
                logger.critical("Tesseract OCR engine not found in PATH. Please install it. OCR functionality will be disabled.")
                self.tesseract_available = False
        logger.info("KPIExtractor initialized.")

    def _extract_text_from_image(self, image_path: str) -> Optional[str]:

        if not self.tesseract_available:
            logger.error("Tesseract not available. Cannot perform OCR on image.")
            return None
        if Image is None:
            logger.error("Pillow (PIL) library not found. Cannot open image files.")
            return None

        try:
            img = Image.open(image_path)
            # You can add language configuration, e.g., lang='eng+hin' for English and Hindi
            text = pytesseract.image_to_string(img, lang='eng')
            logger.info(f"Successfully extracted text from image: {image_path}")
            return text
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            return None
        except Exception as e:
            logger.error(f"Error during OCR for image {image_path}: {e}", exc_info=True)
            return None

    def _extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:

        if pdfminer_extract_text is None:
            logger.error("pdfminer.six library not found. Cannot extract text from PDF.")
            return None

        try:
            # Attempt direct text extraction first
            text = pdfminer_extract_text(pdf_path, laparams=LAParams())
            if text and len(text.strip()) > 0: # Check if any text was extracted
                logger.info(f"Successfully extracted text directly from PDF: {pdf_path}")
                return text
            else:
                logger.warning(f"Direct text extraction from PDF {pdf_path} yielded no content or only whitespace. This might be a scanned PDF. For this prototype, we are not implementing PDF to Image conversion for OCR due to external dependency (poppler).")
                return None # Indicate failure if direct text extraction failed
        except FileNotFoundError:
            logger.error(f"PDF file not found: {pdf_path}")
            return None
        except Exception as e:
            logger.error(f"Error during PDF text extraction for {pdf_path}: {e}", exc_info=True)
            return None

    def extract_text_from_document(self, file_path: str) -> Optional[str]:

        extracted_content: Optional[str] = None
        file_extension = os.path.splitext(file_path)[1].lower()

        logger.info(f"Attempting to extract raw text from: {file_path} (Type: {file_extension})")

        if file_extension == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    extracted_content = f.read()
                logger.info(f"Successfully read text from file: {file_path}")
            except FileNotFoundError:
                logger.error(f"Text file not found: {file_path}")
                return None
            except Exception as e:
                logger.error(f"Error reading text file {file_path}: {e}", exc_info=True)
                return None
        elif file_extension in ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'):
            extracted_content = self._extract_text_from_image(file_path)
        elif file_extension == '.pdf':
            extracted_content = self._extract_text_from_pdf(file_path)
        else:
            logger.error(f"Unsupported file type for raw text extraction: {file_extension}")
            return None

        if not extracted_content:
            logger.error(f"No content extracted from {file_path}.")
            return None

        logger.info(f"Successfully extracted raw text from {file_path}.")
        return extracted_content

