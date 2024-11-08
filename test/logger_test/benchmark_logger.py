import time
import threading
from logger_factory import ThreadSafeLoggerFactory
import os

def benchmark_logging(thread_id, iterations):
    """基准测试函数"""
    logger = ThreadSafeLoggerFactory.get_logger(f"Benchmark-{thread_id}")
    start_time = time.time()
    
    for i in range(iterations):
        logger.debug(f"Debug message {i} from thread {thread_id}")
        logger.info(f"Info message {i} from thread {thread_id}")
        if i % 100 == 0:
            logger.warning(f"Warning message {i} from thread {thread_id}")
            logger.error(f"Error message {i} from thread {thread_id}")
    
    end_time = time.time()
    return end_time - start_time

def run_benchmark(num_threads, iterations_per_thread):
    """运行基准测试"""
    print(f"\n=== Running benchmark with {num_threads} threads, "
          f"{iterations_per_thread} iterations per thread ===")
    
    threads = []
    start_time = time.time()
    
    # 创建并启动线程
    for i in range(num_threads):
        t = threading.Thread(target=benchmark_logging, args=(i, iterations_per_thread))
        threads.append(t)
        t.start()
    
    # 等待所有线程完成
    for t in threads:
        t.join()
    
    end_time = time.time()
    total_time = end_time - start_time
    total_operations = num_threads * iterations_per_thread * 4  # 4种日志级别
    
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Operations per second: {total_operations/total_time:.2f}")
    print(f"Average time per operation: {(total_time/total_operations)*1000:.3f} ms")

def main():
    # 设置日志目录
    log_dir = os.path.join(os.path.dirname(__file__), "benchmark_logs")
    ThreadSafeLoggerFactory.set_log_dir(log_dir)
    
    try:
        # 运行不同规模的基准测试
        scenarios = [
            (1, 1000),    # 单线程，1000次迭代
            (5, 1000),    # 5个线程，每个1000次迭代
            (10, 1000),   # 10个线程，每个1000次迭代
            (20, 500),    # 20个线程，每个500次迭代
        ]
        
        for num_threads, iterations in scenarios:
            run_benchmark(num_threads, iterations)
            
    finally:
        # 清理资源
        ThreadSafeLoggerFactory.reset()

if __name__ == "__main__":
    main()
