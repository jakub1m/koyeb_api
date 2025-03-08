import os

# face recognition settings
THRESHOLD = 0.55

# Server settings
PORT = os.getenv("PORT")

# Database settings
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TABLE_NAME = "face_embeddings"

# Backend Database settings
SUPABASE_BACKEND_URL = os.getenv("SUPABASE_BACKEND_URL")
SUPABASE_BACKEND_KEY = os.getenv("SUPABASE_BACKEND_KEY")
BUCKET = "avatars"

URL_2_CUT = os.getenv("URL_2_CUT")

