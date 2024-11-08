#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import os
import random

from api.db.db_utils import bulk_insert_into_db
from deepdoc.parser import PdfParser
from peewee import JOIN
from api.db.db_models import DB, File2Document, File
from api.db import StatusEnum, FileType, TaskStatus
from api.db.db_models import Task, Document, Knowledgebase, Tenant
from api.db.services.common_service import CommonService
from api.db.services.document_service import DocumentService
from api.utils import current_timestamp, get_uuid
from deepdoc.parser.excel_parser import RAGFlowExcelParser
from rag.settings import SVR_QUEUE_NAME
from rag.utils.storage_factory import STORAGE_IMPL
from rag.utils.redis_conn import REDIS_CONN


class TaskService(CommonService):
    model = Task

    @classmethod
    @DB.connection_context()
    def get_tasks(cls, task_id):
        """获取指定任务ID的任务详细信息

        Args:
            task_id: 任务ID

        Returns:
            list: 包含任务详细信息的字典列表。如果任务不存在或重试次数超过3次,返回空列表

        Note:
            - 该方法会关联查询Task、Document、Knowledgebase和Tenant表获取完整任务信息
            - 每次调用会更新任务的进度信息和重试次数
            - 任务最多重试3次,超过则放弃该任务
        """
        # 定义需要查询的字段列表
        fields = [
            cls.model.id,  # 任务ID
            cls.model.doc_id,  # 文档ID
            cls.model.from_page,  # 起始页码
            cls.model.to_page,  # 结束页码
            cls.model.retry_count,  # 重试次数
            Document.kb_id,  # 知识库ID
            Document.parser_id,  # 解析器ID
            Document.parser_config,  # 解析器配置
            Document.name,  # 文档名称
            Document.type,  # 文档类型
            Document.location,  # 文档存储位置
            Document.size,  # 文档大小
            Knowledgebase.tenant_id,  # 租户ID
            Knowledgebase.language,  # 知识库语言
            Knowledgebase.embd_id,  # 嵌入模型ID
            Tenant.img2txt_id,  # 图像转文本模型ID
            Tenant.asr_id,  # 语音识别模型ID
            Tenant.llm_id,  # 大语言模型ID
            cls.model.update_time,  # 更新时间
        ]

        # 多表关联查询获取任务信息
        docs = (
            cls.model.select(*fields)
            .join(Document, on=(cls.model.doc_id == Document.id))  # 关联文档表
            .join(
                Knowledgebase, on=(Document.kb_id == Knowledgebase.id)
            )  # 关联知识库表
            .join(Tenant, on=(Knowledgebase.tenant_id == Tenant.id))  # 关联租户表
            .where(cls.model.id == task_id)
        )
        docs = list(docs.dicts())
        if not docs:
            return []

        # 更新任务进度信息
        msg = "\nTask has been received."
        prog = random.random() / 10.0  # 随机生成一个小于0.1的进度值

        # 检查重试次数是否超限
        if docs[0]["retry_count"] >= 3:
            msg = "\nERROR: Task is abandoned after 3 times attempts."
            prog = -1  # 进度设为-1表示任务失败

        # 更新任务状态
        cls.model.update(
            progress_msg=cls.model.progress_msg + msg,  # 追加进度消息
            progress=prog,
            retry_count=docs[0]["retry_count"] + 1,  # 重试次数+1
        ).where(cls.model.id == docs[0]["id"]).execute()

        # 重试次数超过3次则放弃任务
        if docs[0]["retry_count"] >= 3:
            return []

        return docs

    @classmethod
    @DB.connection_context()
    def get_ongoing_doc_name(cls):
        with DB.lock("get_task", -1):
            docs = (
                cls.model.select(
                    *[Document.id, Document.kb_id, Document.location, File.parent_id]
                )
                .join(Document, on=(cls.model.doc_id == Document.id))
                .join(
                    File2Document,
                    on=(File2Document.document_id == Document.id),
                    join_type=JOIN.LEFT_OUTER,
                )
                .join(
                    File,
                    on=(File2Document.file_id == File.id),
                    join_type=JOIN.LEFT_OUTER,
                )
                .where(
                    Document.status == StatusEnum.VALID.value,
                    Document.run == TaskStatus.RUNNING.value,
                    ~(Document.type == FileType.VIRTUAL.value),
                    cls.model.progress < 1,
                    cls.model.create_time >= current_timestamp() - 1000 * 600,
                )
            )
            docs = list(docs.dicts())
            if not docs:
                return []

            return list(
                set(
                    [
                        (
                            d["parent_id"] if d["parent_id"] else d["kb_id"],
                            d["location"],
                        )
                        for d in docs
                    ]
                )
            )

    @classmethod
    @DB.connection_context()
    def do_cancel(cls, id):
        try:
            task = cls.model.get_by_id(id)
            _, doc = DocumentService.get_by_id(task.doc_id)
            return doc.run == TaskStatus.CANCEL.value or doc.progress < 0
        except Exception as e:
            pass
        return False

    @classmethod
    @DB.connection_context()
    def update_progress(cls, id, info):
        if os.environ.get("MACOS"):
            if info["progress_msg"]:
                cls.model.update(
                    progress_msg=cls.model.progress_msg + "\n" + info["progress_msg"]
                ).where(cls.model.id == id).execute()
            if "progress" in info:
                cls.model.update(progress=info["progress"]).where(
                    cls.model.id == id
                ).execute()
            return

        with DB.lock("update_progress", -1):
            if info["progress_msg"]:
                cls.model.update(
                    progress_msg=cls.model.progress_msg + "\n" + info["progress_msg"]
                ).where(cls.model.id == id).execute()
            if "progress" in info:
                cls.model.update(progress=info["progress"]).where(
                    cls.model.id == id
                ).execute()


def queue_tasks(doc: dict, bucket: str, name: str):
    """将文档解析任务加入队列

    Args:
        doc (dict): 文档信息字典,包含id、type、parser_config等字段
        bucket (str): 存储桶名称
        name (str): 文件名称

    Returns:
        None

    Note:
        该方法会根据文档类型和解析器配置,将文档分割成多个子任务并加入队列
        对于PDF文档会按页数分割,Excel表格按行数分割
        其他类型文档作为单个任务处理
    """

    def new_task():
        return {"id": get_uuid(), "doc_id": doc["id"]}

    tsks = []

    if doc["type"] == FileType.PDF.value:
        file_bin = STORAGE_IMPL.get(bucket, name)  # 获取PDF文件内容
        do_layout = doc["parser_config"].get("layout_recognize", True)  # 是否识别布局
        pages = PdfParser.total_page_number(doc["name"], file_bin)  # 获取总页数
        page_size = doc["parser_config"].get(
            "task_page_size", 12
        )  # 设置每个任务处理的页数
        if doc["parser_id"] == "paper":
            page_size = doc["parser_config"].get("task_page_size", 22)
        if doc["parser_id"] in ["one", "knowledge_graph"] or not do_layout:
            page_size = 10**9  # 整个文档作为一个任务

        # 处理页面范围
        page_ranges = doc["parser_config"].get("pages") or [(1, 10**5)]
        for s, e in page_ranges:
            s -= 1
            s = max(0, s)
            e = min(e - 1, pages)
            # 按页数分割任务
            for p in range(s, e, page_size):
                task = new_task()
                task["from_page"] = p
                task["to_page"] = min(p + page_size, e)
                tsks.append(task)

    elif doc["parser_id"] == "table":
        file_bin = STORAGE_IMPL.get(bucket, name)
        rn = RAGFlowExcelParser.row_number(doc["name"], file_bin)  # 获取总行数
        # 每3000行创建一个任务
        for i in range(0, rn, 3000):
            task = new_task()
            task["from_page"] = i
            task["to_page"] = min(i + 3000, rn)
            tsks.append(task)
    else:
        tsks.append(new_task())  # 其他类型文件创建单个任务
    # 批量保存任务到数据库
    bulk_insert_into_db(Task, tsks, True)
    # 更新文档状态为开始解析
    DocumentService.begin2parse(doc["id"])
    # 将任务发送到Redis队列
    for t in tsks:
        assert REDIS_CONN.queue_product(
            SVR_QUEUE_NAME, message=t
        ), "Can't access Redis. Please check the Redis' status."
