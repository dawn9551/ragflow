# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# RAGFlow 开发指南

RAGFlow是一个基于深度文档理解的开源RAG引擎，采用Python后端 + React前端的架构。

## 常用开发命令

### 环境设置和依赖管理
```bash
# 安装Python依赖（基础版本）
uv sync --python 3.10

# 安装完整依赖（包含embedding模型）
uv sync --python 3.10 --extra full

# 激活虚拟环境
source .venv/bin/activate

# 安装前端依赖
cd web && npm install
```

### 测试命令
```bash
# 运行不同优先级的测试
pytest --level=p1  # 高优先级测试（快速）
pytest --level=p2  # 中等优先级测试
pytest --level=p3  # 低优先级测试（完整）

# 运行特定模块测试
pytest test/testcases/test_http_api/
pytest sdk/python/test/

# 沙箱安全测试
cd sandbox && make test
```

### 代码质量检查
```bash
# 运行所有pre-commit检查
pre-commit run --all-files

# Python代码格式化和检查
ruff check --fix    # 修复可自动修复的问题
ruff format         # 格式化代码

# 前端代码检查
cd web && npm run lint
```

### 服务启动

#### Docker方式（推荐用于生产）
```bash
# 启动完整服务栈（CPU版本）
cd docker && docker compose -f docker-compose.yml up -d

# 启动GPU版本
cd docker && docker compose -f docker-compose-gpu.yml up -d

# 启动基础服务（用于开发）
docker compose -f docker/docker-compose-base.yml up -d
```

#### 源码方式（用于开发）
```bash
# 启动后端服务
export PYTHONPATH=$(pwd)
bash docker/launch_backend_service.sh

# 或直接启动API服务器
python api/ragflow_server.py

# 启动前端开发服务器
cd web && npm run dev
```

### 构建命令
```bash
# 构建前端
cd web && npm run build

# 构建Docker镜像（精简版）
docker build --build-arg LIGHTEN=1 -f Dockerfile -t ragflow:slim .

# 构建Docker镜像（完整版）
docker build -f Dockerfile -t ragflow:full .
```

## 核心架构

### 系统组件层次
- **api/** - Flask REST API服务层，包含所有业务逻辑模块
- **rag/** - RAG核心引擎，包含LLM接口、文档处理、检索系统
- **agent/** - 可视化Agent工作流引擎，基于组件化设计
- **deepdoc/** - 深度文档理解模块，包含OCR、布局分析、表格识别
- **web/** - React前端应用，基于UmiJS + Ant Design
- **graphrag/** - 知识图谱构建和查询
- **mcp/** - Model Context Protocol服务器

### 数据存储架构
- **关系数据库**: MySQL/PostgreSQL（元数据）
- **向量数据库**: Elasticsearch/Infinity（文档和向量）
- **对象存储**: Minio/S3/Azure（文件存储）
- **缓存**: Redis（会话和临时数据）

### 关键设计模式
- **模板化分块**: 针对不同文档类型使用专门的解析模板
- **组件化Agent**: 可拖拽的工作流组件系统
- **多模型支持**: 统一的LLM接口层，支持多种模型提供商
- **深度文档理解**: 基于视觉AI的文档布局分析和内容提取

## 开发环境配置

### 必需服务
启动开发环境前需要这些服务运行：
```bash
# 添加hosts映射
echo "127.0.0.1 es01 infinity mysql minio redis sandbox-executor-manager" >> /etc/hosts

# 启动依赖服务
docker compose -f docker/docker-compose-base.yml up -d
```

### 环境变量
- **HF_ENDPOINT**: 如无法访问HuggingFace，设置为 `https://hf-mirror.com`
- **PYTHONPATH**: 设置为项目根目录
- **NLTK_DATA**: 设置为 `./nltk_data`

## 测试策略

### 测试分层
- **p1**: 核心功能的快速测试
- **p2**: 主要业务逻辑测试
- **p3**: 边缘情况和集成测试

### 测试类型
- **HTTP API测试**: `test/testcases/test_http_api/`
- **SDK测试**: `sdk/python/test/`
- **Web API测试**: `test/testcases/test_web_api/`
- **安全测试**: `sandbox/tests/`

## 开发工作流

1. **功能开发**: 在相应的模块目录下开发功能
2. **代码检查**: 使用 `pre-commit run --all-files` 确保代码质量
3. **单元测试**: 运行相关的测试用例确保功能正常
4. **集成测试**: 在Docker环境中测试完整功能
5. **文档更新**: 更新相关的API文档和用户指南

## 常见开发任务

### 添加新的文档解析器
在 `deepdoc/parser/` 中添加新的解析器类，继承基础解析器接口

### 添加新的LLM提供商
在 `rag/llm/` 中实现新的模型接口，更新 `conf/llm_factories.json`

### 添加新的Agent组件
在 `agent/component/` 中创建新组件，继承 `ComponentBase` 类

### 添加新的工具
在 `agent/tools/` 中实现新工具，继承 `ToolBase` 类

## 性能优化

### 向量检索优化
- 使用Infinity替代Elasticsearch以提升检索性能
- 通过 `DOC_ENGINE=infinity` 在 `docker/.env` 中切换

### 内存优化
- 系统使用jemalloc进行内存管理
- 在 `launch_backend_service.sh` 中配置内存分配策略

### 并发优化
- 通过 `WS` 环境变量控制task executor数量
- 默认使用1个worker，可根据需要调整