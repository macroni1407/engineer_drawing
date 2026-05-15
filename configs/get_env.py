import os
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()

# Lấy giá trị env

# gradio
GRADIO_PORT=int(os.getenv("GRADIO_PORT", 7860))

# redis env
REDIS_HOST=os.getenv("REDIS_HOST")
REDIS_PORT=int(os.getenv("REDIS_PORT", 15105))
REDIS_DECODE_RESPONSES=os.getenv("REDIS_DECODE_RESPONSES")
REDIS_USERNAME=os.getenv("REDIS_USERNAME")
REDIS_PASSWORD=os.getenv("REDIS_PASSWORD")

UPSTASH_REDIS_REST_URL=os.getenv("UPSTASH_REDIS_REST_URL")
UPSTASH_REDIS_REST_TOKEN=os.getenv("UPSTASH_REDIS_REST_TOKEN")

