import os
from io import BytesIO
import fitz # PyMuPDF
from PIL import Image
import pytesseract
import logging

logger = logging.getLogger(__name__)

class KPIExtractor:
    def __init__(self):
        # Ensure Tesseract is installed and accessible in your environment
        # If not, you might need to specify pytesseract.pytesseract.tesseract_cmd = '/path/to/tesseract'
        logger.info("KPIExtractor initialized. Ready for text extraction.")

    def extract_text_from_document(self, document_path: str) -> str:
        """
        Extracts text from various document types (PDF, TXT, JPG, PNG).
        """
        if not os.path.exists(document_path):
            logger.error(f"Document not found at path: {document_path}")
            raise FileNotFoundError(f"Document not found: {document_path}")

        file_extension = os.path.splitext(document_path)[1].lower()
        extracted_text = ""

        try:
            if file_extension == '.txt':
                with open(document_path, 'r', encoding='utf-8') as f:
                    extracted_text = f.read()
                logger.info(f"Text extracted from TXT file: {document_path}")
            elif file_extension == '.pdf':
                extracted_text = self._extract_text_from_pdf(document_path)
                logger.info(f"Text extracted from PDF file: {document_path}")
            elif file_extension in ['.jpg', '.jpeg', '.png']:
                extracted_text = self._extract_text_from_image(document_path)
                logger.info(f"Text extracted from image file: {document_path}")
            else:
                logger.warning(f"Unsupported file type for text extraction: {file_extension} for {document_path}")
                raise ValueError(f"Unsupported file type: {file_extension}")
        except Exception as e:
            logger.error(f"Error during text extraction from {document_path}: {e}", exc_info=True)
            raise

        return extracted_text

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extracts text from a PDF document using PyMuPDF (fitz).
        If direct text extraction is poor, it falls back to OCR.
        """
        text = ""
        try:
            document = fitz.open(pdf_path)
            for page_num in range(len(document)):
                page = document.load_page(page_num)
                page_text = page.get_text("text")
                text += page_text + "\n"
            document.close()
            logger.info(f"Successfully extracted text from PDF {pdf_path} using PyMuPDF.")
        except Exception as e:
            logger.warning(f"Direct PDF text extraction failed for {pdf_path}: {e}. Attempting OCR fallback.")
            # Fallback to OCR if direct text extraction fails or is empty
            text = self._ocr_pdf(pdf_path)
            if not text:
                logger.error(f"OCR fallback also failed for PDF {pdf_path}.")
                raise RuntimeError(f"Failed to extract text from PDF {pdf_path} even with OCR.")
        return text

    def _ocr_pdf(self, pdf_path: str) -> str:
        """
        Performs OCR on each page of a PDF document.
        """
        text = ""
        try:
            document = fitz.open(pdf_path)
            for page_num in range(len(document)):
                page = document.load_page(page_num)
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png")
                img = Image.open(BytesIO(img_bytes))
                page_text = pytesseract.image_to_string(img)
                text += page_text + "\n"
            document.close()
            logger.info(f"Successfully extracted text from PDF {pdf_path} using OCR.")
        except Exception as e:
            logger.error(f"Error during OCR of PDF {pdf_path}: {e}", exc_info=True)
            raise
        return text

    def _extract_text_from_image(self, image_path: str) -> str:
        """
        Extracts text from an image file using Tesseract OCR.
        """
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)
            logger.info(f"Successfully extracted text from image {image_path} using Tesseract OCR.")
        except Exception as e:
            logger.error(f"Error during OCR of image {image_path}: {e}", exc_info=True)
            raise
        return text

