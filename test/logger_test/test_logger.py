import threading
import time
import random
from logger_factory import ThreadSafeLoggerFactory
import os


def simulate_work(thread_id):
    """模拟工作线程"""
    logger = ThreadSafeLoggerFactory.get_logger(f"Worker-{thread_id}")

    for i in range(5):
        # 模拟随机工作
        time.sleep(random.uniform(0.1, 0.5))

        # 记录不同级别的日志
        logger.debug(f"Thread {thread_id} is processing step {i}")
        logger.info(f"Thread {thread_id} completed step {i}")

        # 随机产生一些警告和错误
        if random.random() < 0.3:
            logger.warning(f"Thread {thread_id} encountered a warning in step {i}")
        if random.random() < 0.1:
            logger.error(f"Thread {thread_id} encountered an error in step {i}")
            try:
                raise Exception("Simulated error")
            except Exception as e:
                logger.exception("Detailed error information")


def test_single_thread():
    """测试单线程场景"""
    print("\n=== Testing Single Thread ===")
    logger = ThreadSafeLoggerFactory.get_logger("SingleThread")
    logger.info("Starting single thread test")
    logger.debug("This is a debug message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.info("Completed single thread test")


def test_multiple_threads():
    """测试多线程场景"""
    print("\n=== Testing Multiple Threads ===")
    threads = []
    for i in range(5):
        t = threading.Thread(target=simulate_work, args=(i,))
        threads.append(t)
        t.start()

    # 等待所有线程完成
    for t in threads:
        t.join()


def main():
    # 设置日志目录
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    ThreadSafeLoggerFactory.set_log_dir(log_dir)

    try:
        # 运行测试
        # test_single_thread()
        test_multiple_threads()

    finally:
        # 清理资源
        ThreadSafeLoggerFactory.reset()


if __name__ == "__main__":
    main()
