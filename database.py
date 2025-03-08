from supabase import create_client
import io
from PIL import Image

from config import URL_2_CUT

class BackendDatabase:
    def __init__(self, supabase_url, supabase_key, bucket):
        self.supabase = create_client(supabase_url, supabase_key)
        self.bucket = bucket
        
    def download_photo(self, path):
        path = path.replace(URL_2_CUT,"")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.supabase.storage.from_(self.bucket).download(path)
                image_stream = io.BytesIO(response)
                img = Image.open(image_stream)
                return img
            except Exception as e:
                print(e)
