import numpy as np
import face_recognition

from supabase import create_client

from config import THRESHOLD
from exceptions import FaceRecognitionException, DeleteUserException, ExtractUserException, AddUserException

class SupabaseFaceDatabase:
    def __init__(self, supabase_url, supabase_key, table_name):
        self.supabase = create_client(supabase_url, supabase_key)
        self.table_name = table_name
    
    def add_faces_from_image(self, image, user_id):
        image = np.array(image)

        if image.shape[-1] == 4:  
            image = image[..., :3]
            
        face_locations = face_recognition.face_locations(image)
            
        if not face_locations:
            raise FaceRecognitionException("No faces found")

        if len(face_locations) > 1:
            raise FaceRecognitionException("Too many faces found")
            
        face_encodings = face_recognition.face_encodings(image, face_locations)  
            
        embedding_list = face_encodings[0].tolist()
                
        if user_id in self.get_all_users():
            self.supabase.table(self.table_name) \
                .update({"face_embedding": embedding_list}) \
                .eq("user_id", user_id) \
                .execute()
                    
        else:
            data = {
                "user_id": user_id,
                "face_embedding": embedding_list
            }
                
            self.supabase.table(self.table_name).insert(data).execute()
        return True

    
    def verify_face(self, image, user_id):
        image = np.array(image)
        if image.shape[-1] == 4:  
            image = image[..., :3]
        
        face_locations = face_recognition.face_locations(image)
        
        if len(face_locations) > 1:
            raise FaceRecognitionException("Too many faces found")
        
        if not face_locations:
            raise FaceRecognitionException("No faces found")
        
        face_encoding = face_recognition.face_encodings(image, face_locations)[0]
        
        user_embedding = self.supabase.from_('face_embeddings') \
                            .select('face_embedding') \
                            .eq('user_id', user_id) \
                            .execute()
        
        if not user_embedding.data:
            raise FaceRecognitionException(f"No face data found for user_id: {user_id}")
        
        stored_data = user_embedding.data[0]['face_embedding']
        stored_embedding = np.fromstring(stored_data.strip('[]'), sep=',')
        
        distance = face_recognition.face_distance([stored_embedding], face_encoding)[0]

        return {
            'match': bool(distance <= THRESHOLD),
            'confidence': float(1 - distance),
            'user_id': user_id,
            'distance': float(distance)
        }
    def get_all_users(self):
        response = self.supabase.table(self.table_name).select('user_id').execute()
        
        if hasattr(response, 'error') and response.error:
            raise ExtractUserException(response.error)
        
        users = set()
        for item in response.data:
            users.add(item['user_id'])
        
        return list(users)
    
    def delete_user_faces(self, user_id):
            response = self.supabase.table(self.table_name).delete().eq('user_id', user_id).execute()
            if hasattr(response, 'error') and response.error:
                raise DeleteUserException(response.error)
            return True
