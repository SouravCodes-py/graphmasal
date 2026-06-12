import logging
from .base_loader import DocumentLoader

logger = logging.getLogger(__name__)

class MarkdownLoader(DocumentLoader):
    """Loader for Markdown documents."""

    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a Markdown file.
        Treats markdown as plain text, returning its content directly.
        
        Args:
            file_path (str): Path to the Markdown file.
            
        Returns:
            str: The extracted text.
        """
        logger.info(f"Extracting text from Markdown: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error extracting text from Markdown {file_path} with latin-1: {e}")
                raise
        except Exception as e:
            logger.error(f"Error extracting text from Markdown {file_path}: {e}")
            raise
