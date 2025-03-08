from pydantic import BaseModel

class NewUserRequest(BaseModel):
    imageUrl: str
    userId: str

class VerifyUserRequest(BaseModel):
    imageUrl: str
    userId: str
    
class DeleteUserRequest(BaseModel):
    userId: str
