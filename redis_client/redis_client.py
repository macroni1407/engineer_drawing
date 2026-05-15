# import redis
from upstash_redis import Redis
from configs import *

def use_redis():
  # return redis.Redis(
  #       host=REDIS_HOST,
  #       port=REDIS_PORT,
  #       # decode_responses=DECODE_RESPONSES,
  #       username=REDIS_USERNAME,
  #       password=REDIS_PASSWORD,
  #   )

  return Redis(
    url=UPSTASH_REDIS_REST_URL,
    token=UPSTASH_REDIS_REST_TOKEN
  )

