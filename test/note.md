# Logger

## Logger
- Logger是一个日志系统的入口，通过它来记录日志。
- Logger决定哪些日志需要被记录。
```python
logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG)
logger.debug("debug message")
logger.info("info message")

```

## Handler
- Handler决定了日志要输出到哪里及如何输出
- 一个Logger可以有多个Handler，每个Handler可以设置不同的日志级别，从而实现细粒度的日志控制。
- 常见的Handler有：
    - StreamHandler：将日志输出到控制台或文件。
    - FileHandler：将日志输出到文件。
    - RotatingFileHandler：将日志输出到文件，并支持日志文件的滚动。
    - TimedRotatingFileHandler：将日志输出到文件，并根据时间戳自动滚动日志文件。
    - HTTPHandler：将日志输出到HTTP服务器。
    - SMTPHandler：将日志输出到邮件服务器。

## 日志优先级别
```txt
logging.DEBUG = 10      # 调试信息
logging.INFO = 20       # 一般信息
logging.WARNING = 30    # 警告信息
logging.ERROR = 40      # 错误信息
logging.CRITICAL = 50   # 严重错误
```

## Logger和Handler的关系
- Logger通过addHandler方法添加一个或多个Handler。
- Logger在记录日志时，会根据Handler的日志级别和Filter来决定是否输出该日志。
- 每个Handler可以设置不同的日志级别，从而实现细粒度的日志控制。

### Logger增加多个Handler
```python
# 创建两个不同级别的 handler
debug_handler = ThreadSafeLoggerFactory.get_handler(name, logging.DEBUG)   # 级别为 DEBUG
error_handler = ThreadSafeLoggerFactory.get_handler(f"{name}_error", logging.ERROR)  # 级别为 ERROR

logger.addHandler(debug_handler)
logger.addHandler(error_handler)

# 使用示例
logger = ThreadSafeLoggerFactory.get_logger("myapp")

# 1. DEBUG 级别的日志
logger.debug("这是调试信息")  
# 只会被 debug_handler 处理（因为 error_handler 的级别是 ERROR）
# 只会写入 myapp_20240321.log

# 2. INFO 级别的日志
logger.info("这是普通信息")   
# 只会被 debug_handler 处理
# 只会写入 myapp_20240321.log

# 3. ERROR 级别的日志
logger.error("这是错误信息")  
# 会被两个 handler 都处理！
# 会同时写入：
# - myapp_20240321.log
# - myapp_error_20240321.log
```

### Handler选择规则
- Handler只处理大于等于自己级别的日志
- 例如：设置ERROR级别的Handler。
    - 会处理ERROR、CRITICAL级别的日志。
    - 不会处理INFO、DEBUG级别的日志。
