from logging import Logger
from types import ModuleType

def error_message_detail(error : Exception, error_detail : ModuleType, logger : Logger) -> str:
    """
    Extracts details error information including file name, line numberr, and the error message. 
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename

    line_number = exc_tb.tb_lineno
    error_message = f'Error occurred in python script : [{file_name}] at line number [{line_number}] : {str(error)}'
    logger.error(error_message)
    return error_message

class MyException(Exception): 
    """
    Custom exception class for handling errors
    """
    def __init__(self, error_message : Exception, error_detail : ModuleType, logger : Logger):
        """
        Initializes the Exception with a detailed error message
        """

        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail, logger)
    
    def __str__(self) -> str: 
        """
        Returns the string representation of the error message
        """
        return self.error_message

