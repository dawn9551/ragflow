"""
redis:
  db: 1
  password: 'infini_rag_flow'
  host: 'localhost:9736'
"""

import redis


class RedisClient:
    def __init__(self, config):
        self.config = config
        self.REDIS = None
        self.init_redis()

    def init_redis(self):
        try:
            host = "localhost:9736"
            host_parts = host.split(":")
            self.REDIS = redis.StrictRedis(
                host=host_parts[0],
                port=int(host_parts[1]) if len(host_parts) > 1 else 9736,
                db=int(self.config.get("db", 1)),
                password=None,
                decode_responses=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
                retry_on_timeout=True,
                max_connections=10,
            )
            # 测试连接
            self.REDIS.ping()
        except redis.ConnectionError as e:
            raise Exception(f"无法连接到Redis服务器: {str(e)}")
        except Exception as e:
            raise Exception(f"初始化Redis客户端时发生错误: {str(e)}")


if __name__ == "__main__":
    config = {
        "db": 1,
        "password": "infini_rag_flow",
        "host": "localhost:9736",
    }
    redis_client = RedisClient(config)

    group_info = redis_client.REDIS.xinfo_groups("rag_flow_svr_queue")
    print(group_info)
