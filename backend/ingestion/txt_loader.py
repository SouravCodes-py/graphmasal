import logging
from .base_loader import DocumentLoader

logger = logging.getLogger(__name__)

class TxtLoader(DocumentLoader):
    """Loader for plain text documents."""

    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a TXT file.
        
        Args:
            file_path (str): Path to the TXT file.
            
        Returns:
            str: The extracted text.
        """
        logger.info(f"Extracting text from TXT: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error extracting text from TXT {file_path} with latin-1: {e}")
                raise
        except Exception as e:
            logger.error(f"Error extracting text from TXT {file_path}: {e}")
            raise
