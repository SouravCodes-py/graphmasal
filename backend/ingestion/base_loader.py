import logging
from abc import ABC, abstractmethod

class DocumentLoader(ABC):
    """
    Abstract base class for all document loaders.
    """
    
    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """
        Extract plain text from the given file.
        
        Args:
            file_path (str): The path to the file to process.
            
        Returns:
            str: The extracted plain text.
        """
        pass
