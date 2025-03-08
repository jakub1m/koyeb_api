import io

from PIL import Image
from fastapi import FastAPI, HTTPException

from database import BackendDatabase
from exceptions import FaceRecognitionException, DeleteUserException, DownloadPhotoException
from requests_schema import NewUserRequest, DeleteUserRequest, VerifyUserRequest
from vector_face_recognition import SupabaseFaceDatabase

from config import (SUPABASE_BACKEND_KEY, 
                    SUPABASE_BACKEND_URL, 
                    SUPABASE_KEY, 
                    SUPABASE_URL, 
                    BUCKET, 
                    TABLE_NAME, 
                    PORT)

app = FastAPI()
database = BackendDatabase(supabase_key=SUPABASE_BACKEND_KEY, supabase_url=SUPABASE_BACKEND_URL, bucket=BUCKET)
vector_database =  SupabaseFaceDatabase(supabase_key=SUPABASE_KEY, supabase_url=SUPABASE_URL, table_name=TABLE_NAME)

@app.post("/user/add")
async def add_new_user(new_user_request: NewUserRequest):
    try:
        image = database.download_photo(path=new_user_request.imageUrl)
        vector_database.add_faces_from_image(image, new_user_request.userId)
        return
    
    except (FaceRecognitionException, DownloadPhotoException) as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/user/delete")
async def delete_user(del_user_request: DeleteUserRequest):
    try:
        vector_database.delete_user_faces(del_user_request.userId)
        return
    except DeleteUserException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user/verify")
async def verify_user(verify_user_request: VerifyUserRequest):
    try:
        print(f"UserID: {verify_user_request.userId}, ImageUrl: {verify_user_request.imageUrl}")
        image = database.download_photo(path=verify_user_request.imageUrl)
        result = vector_database.verify_face(image=image, user_id=verify_user_request.userId)
        print(result)
        return result
    
    except (FaceRecognitionException, DownloadPhotoException) as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/getAll")
async def get_all_users():
    try:
        users = vector_database.get_all_users()
        return users
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
