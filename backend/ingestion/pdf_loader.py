import logging
import fitz  # PyMuPDF
from .base_loader import DocumentLoader

logger = logging.getLogger(__name__)

class PDFLoader(DocumentLoader):
    """Loader for PDF documents."""

    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a PDF file using PyMuPDF.
        
        Args:
            file_path (str): Path to the PDF file.
            
        Returns:
            str: The extracted text.
        """
        logger.info(f"Extracting text from PDF: {file_path}")
        try:
            text_chunks = []
            with fitz.open(file_path) as doc:
                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    if text:
                        text_chunks.append(text)
            
            return "\n".join(text_chunks)
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            raise
