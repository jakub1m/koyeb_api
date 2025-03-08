class FaceRecognitionException(Exception):
    def __init__(self, message="Face recognition error occurred"):
        super().__init__(message)
        
class DeleteUserException(Exception):
    def __init__(self, message="Error occurred when deleting the user"):
        super().__init__(message)
        
class ExtractUserException(Exception):
    def __init__(self, message="Error occurred when extracting users"):
        super().__init__(message)

class AddUserException(Exception):
    def __init__(self, message="Error occurred when adding the user"):
        super().__init__(message)

class DownloadPhotoException(Exception):
    def __init__(self, message="Error occurred when downloading photo from Supabase"):
        super().__init__(message)

