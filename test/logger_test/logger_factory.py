import os
import logging
from logging.handlers import TimedRotatingFileHandler
from threading import RLock
from datetime import datetime


class ThreadSafeLoggerFactory:
    logger_dict = {}
    handler_dict = {}
    lock = RLock()
    LOG_DIR = None
    LOG_FORMAT = "[%(levelname)s] [%(asctime)s] [%(name)s] [%(funcName)s:%(lineno)d]: %(message)s"
    
    @staticmethod
    def set_log_dir(directory):
        """设置日志目录"""
        with ThreadSafeLoggerFactory.lock:
            ThreadSafeLoggerFactory.LOG_DIR = directory
            os.makedirs(directory, exist_ok=True)
    
    @staticmethod
    def get_handler(name, level=logging.DEBUG):
        """获取日志处理器"""
        if not ThreadSafeLoggerFactory.LOG_DIR:
            # 如果没有设置日志目录，返回控制台处理器
            handler = logging.StreamHandler()
            formatter = logging.Formatter(ThreadSafeLoggerFactory.LOG_FORMAT)
            handler.setFormatter(formatter)
            return handler
            
        handler_key = f"{name}_{level}_{ThreadSafeLoggerFactory.LOG_DIR}"
        
        if handler_key not in ThreadSafeLoggerFactory.handler_dict:
            with ThreadSafeLoggerFactory.lock:
                if handler_key not in ThreadSafeLoggerFactory.handler_dict:
                    # 创建日志文件路径
                    log_file = os.path.join(
                        ThreadSafeLoggerFactory.LOG_DIR,
                        f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
                    )
                    
                    # 创建文件处理器
                    handler = TimedRotatingFileHandler(
                        log_file,
                        when='D',
                        interval=1,
                        backupCount=7,
                        delay=True,
                        encoding='utf-8'
                    )
                    
                    # 设置日志格式
                    formatter = logging.Formatter(ThreadSafeLoggerFactory.LOG_FORMAT)
                    handler.setFormatter(formatter)
                    handler.setLevel(level)
                    
                    ThreadSafeLoggerFactory.handler_dict[handler_key] = handler
                    
        return ThreadSafeLoggerFactory.handler_dict[handler_key]
    
    @staticmethod
    def get_logger(name):
        """获取日志记录器"""
        # 快速路径：检查是否已存在（无锁）
        if name in ThreadSafeLoggerFactory.logger_dict:
            return ThreadSafeLoggerFactory.logger_dict[name]
        
        # 慢路径：需要创建新的logger
        with ThreadSafeLoggerFactory.lock:
            # 双重检查
            if name not in ThreadSafeLoggerFactory.logger_dict:
                logger = logging.getLogger(name)
                logger.setLevel(logging.DEBUG)
                logger.propagate = False  # 避免日志重复
                
                # 添加处理器：一个DEBUG级别，一个ERROR级别
                debug_handler = ThreadSafeLoggerFactory.get_handler(name, logging.DEBUG)
                error_handler = ThreadSafeLoggerFactory.get_handler(f"{name}_error", logging.ERROR)
                
                logger.addHandler(debug_handler)
                logger.addHandler(error_handler)
                
                ThreadSafeLoggerFactory.logger_dict[name] = logger
                
            return ThreadSafeLoggerFactory.logger_dict[name]
    
    @staticmethod
    def reset():
        """重置所有日志记录器和处理器"""
        with ThreadSafeLoggerFactory.lock:
            # 清理所有处理器
            for logger in ThreadSafeLoggerFactory.logger_dict.values():
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)
                    handler.close()
            
            # 清理字典
            ThreadSafeLoggerFactory.logger_dict.clear()
            ThreadSafeLoggerFactory.handler_dict.clear()
