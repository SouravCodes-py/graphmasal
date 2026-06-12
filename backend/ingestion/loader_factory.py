import os
import logging
from .base_loader import DocumentLoader
from .pdf_loader import PDFLoader
from .docx_loader import DocxLoader
from .pptx_loader import PptxLoader
from .txt_loader import TxtLoader
from .markdown_loader import MarkdownLoader

logger = logging.getLogger(__name__)

class LoaderFactory:
    """Factory for creating document loaders based on file extensions."""
    
    @staticmethod
    def get_loader(file_path: str) -> DocumentLoader:
        """
        Get the appropriate DocumentLoader for the given file path.
        
        Args:
            file_path (str): Path to the document.
            
        Returns:
            DocumentLoader: An instance of the appropriate loader.
            
        Raises:
            ValueError: If the file type is not supported.
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == '.pdf':
            return PDFLoader()
        elif ext == '.docx':
            return DocxLoader()
        elif ext == '.pptx':
            return PptxLoader()
        elif ext == '.txt':
            return TxtLoader()
        elif ext in ['.md', '.markdown']:
            return MarkdownLoader()
        else:
            error_msg = f"Unsupported file extension: {ext}"
            logger.error(error_msg)
            raise ValueError(error_msg)
