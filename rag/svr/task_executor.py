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
import sys
import os

# 获取当前脚本所在目录的父目录
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parent_dir = os.path.dirname(current_dir)

# 将父目录添加到 Python 路径中
sys.path.insert(0, parent_dir)


import datetime
import json
import logging
import os
import hashlib
import copy
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO
from multiprocessing.context import TimeoutError
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from elasticsearch_dsl import Q

from api.db import LLMType, ParserType
from api.db.services.dialog_service import keyword_extraction, question_proposal
from api.db.services.document_service import DocumentService
from api.db.services.llm_service import LLMBundle
from api.db.services.task_service import TaskService
from api.db.services.file2document_service import File2DocumentService
from api.settings import retrievaler
from api.utils.file_utils import get_project_base_directory
from api.db.db_models import close_connection
from rag.app import (
    laws,
    paper,
    presentation,
    manual,
    qa,
    table,
    book,
    resume,
    picture,
    naive,
    one,
    audio,
    knowledge_graph,
    email,
)
from rag.nlp import search, rag_tokenizer
from rag.raptor import RecursiveAbstractiveProcessing4TreeOrganizedRetrieval as Raptor
from rag.settings import database_logger, SVR_QUEUE_NAME
from rag.settings import cron_logger, DOC_MAXIMUM_SIZE
from rag.utils import rmSpace, num_tokens_from_string
from rag.utils.es_conn import ELASTICSEARCH
from rag.utils.redis_conn import REDIS_CONN, Payload
from rag.utils.storage_factory import STORAGE_IMPL

BATCH_SIZE = 64

FACTORY = {
    "general": naive,
    ParserType.NAIVE.value: naive,
    ParserType.PAPER.value: paper,
    ParserType.BOOK.value: book,
    ParserType.PRESENTATION.value: presentation,
    ParserType.MANUAL.value: manual,
    ParserType.LAWS.value: laws,
    ParserType.QA.value: qa,
    ParserType.TABLE.value: table,
    ParserType.RESUME.value: resume,
    ParserType.PICTURE.value: picture,
    ParserType.ONE.value: one,
    ParserType.AUDIO.value: audio,
    ParserType.EMAIL.value: email,
    ParserType.KG.value: knowledge_graph,
}

CONSUMER_NAME = "task_consumer_" + ("0" if len(sys.argv) < 2 else sys.argv[1])
PAYLOAD: Payload | None = None


def set_progress(task_id, from_page=0, to_page=-1, prog=None, msg="Processing..."):
    global PAYLOAD
    if prog is not None and prog < 0:
        msg = "[ERROR]" + msg
    cancel = TaskService.do_cancel(task_id)
    if cancel:
        msg += " [Canceled]"
        prog = -1

    if to_page > 0:
        if msg:
            msg = f"Page({from_page + 1}~{to_page + 1}): " + msg
    d = {"progress_msg": msg}
    if prog is not None:
        d["progress"] = prog
    try:
        TaskService.update_progress(task_id, d)
    except Exception as e:
        cron_logger.error("set_progress:({}), {}".format(task_id, str(e)))

    close_connection()
    if cancel:
        if PAYLOAD:
            PAYLOAD.ack()
            PAYLOAD = None
        os._exit(0)


def collect():
    global CONSUMER_NAME, PAYLOAD
    try:
        PAYLOAD = REDIS_CONN.get_unacked_for(
            CONSUMER_NAME, SVR_QUEUE_NAME, "rag_flow_svr_task_broker"
        )
        if not PAYLOAD:
            PAYLOAD = REDIS_CONN.queue_consumer(
                SVR_QUEUE_NAME, "rag_flow_svr_task_broker", CONSUMER_NAME
            )
        if not PAYLOAD:
            time.sleep(1)
            return pd.DataFrame()
    except Exception as e:
        cron_logger.error("Get task event from queue exception:" + str(e))
        return pd.DataFrame()

    msg = PAYLOAD.get_message()
    if not msg:
        return pd.DataFrame()

    if TaskService.do_cancel(msg["id"]):
        cron_logger.info("Task {} has been canceled.".format(msg["id"]))
        return pd.DataFrame()
    tasks = TaskService.get_tasks(msg["id"])
    if not tasks:
        cron_logger.warn("{} empty task!".format(msg["id"]))
        return []

    tasks = pd.DataFrame(tasks)
    if msg.get("type", "") == "raptor":
        tasks["task_type"] = "raptor"
    return tasks


def get_storage_binary(bucket, name):
    return STORAGE_IMPL.get(bucket, name)


def build(row):
    """构建文档的分块和向量化处理

    该函数负责将原始文档进行分块、特征提取和向量化,主要步骤包括:
    1. 检查文件大小是否超限
    2. 从存储获取文件内容
    3. 使用分块器进行文档分块
    4. 为每个分块生成唯一ID和元数据
    5. 可选:生成关键词和问题

    Args:
        row (dict): 包含任务相关信息的字典,必须包含以下字段:
            - id: 任务ID
            - size: 文件大小
            - doc_id: 文档ID
            - kb_id: 知识库ID
            - parser_id: 解析器ID
            - name: 文件名
            - location: 存储位置
            - language: 语言
            - tenant_id: 租户ID
            - parser_config: 解析配置
            - from_page: 起始页码
            - to_page: 结束页码

    Returns:
        list: 包含处理后文档分块的列表,每个分块包含:
            - doc_id: 文档ID
            - kb_id: 知识库ID列表
            - _id: 分块唯一哈希ID
            - content_with_weight: 分块内容
            - create_time: 创建时间
            - create_timestamp_flt: 创建时间戳
            - important_kwd: (可选)关键词列表
            - important_tks: (可选)关键词token
    """
    # 检查文件大小是否超过限制
    if row["size"] > DOC_MAXIMUM_SIZE:
        set_progress(
            row["id"],
            prog=-1,
            msg="File size exceeds( <= %dMb )" % (int(DOC_MAXIMUM_SIZE / 1024 / 1024)),
        )
        return []

    # 设置进度回调函数
    callback = partial(set_progress, row["id"], row["from_page"], row["to_page"])
    chunker = FACTORY[row["parser_id"].lower()]

    # 从存储获取文件内容
    try:
        st = timer()
        # 从MinIO获取文件
        bucket, name = File2DocumentService.get_storage_address(doc_id=row["doc_id"])
        binary = get_storage_binary(bucket, name)
        cron_logger.info(
            "From minio({}) {}/{}".format(timer() - st, row["location"], row["name"])
        )
    except TimeoutError:
        callback(
            -1,
            "Internal server error: Fetch file from minio timeout. Could you try it again.",
        )
        cron_logger.error(
            "Minio {}/{}: Fetch file from minio timeout.".format(
                row["location"], row["name"]
            )
        )
        return
    except Exception as e:
        if re.search("(No such file|not found)", str(e)):
            callback(
                -1,
                "Can not find file <%s> from minio. Could you try it again?"
                % row["name"],
            )
        else:
            callback(-1, "Get file from minio: %s" % str(e).replace("'", ""))
        traceback.print_exc()
        return

    # 使用分块器处理文档
    try:
        cks = chunker.chunk(
            row["name"],
            binary=binary,
            from_page=row["from_page"],
            to_page=row["to_page"],
            lang=row["language"],
            callback=callback,
            kb_id=row["kb_id"],
            parser_config=row["parser_config"],
            tenant_id=row["tenant_id"],
        )
        cron_logger.info(
            "Chunking({}) {}/{}".format(timer() - st, row["location"], row["name"])
        )
    except Exception as e:
        callback(
            -1, "Internal server error while chunking: %s" % str(e).replace("'", "")
        )
        cron_logger.error(
            "Chunking {}/{}: {}".format(row["location"], row["name"], str(e))
        )
        traceback.print_exc()
        return

    # 为每个分块生成元数据
    docs = []
    doc = {"doc_id": row["doc_id"], "kb_id": [str(row["kb_id"])]}
    el = 0
    for ck in cks:
        # 复制基础文档信息并更新分块特定信息
        d = copy.deepcopy(doc)
        d.update(ck)

        # 生成分块唯一哈希ID
        md5 = hashlib.md5()
        md5.update((ck["content_with_weight"] + str(d["doc_id"])).encode("utf-8"))
        d["_id"] = md5.hexdigest()

        # 添加创建时间信息
        d["create_time"] = str(datetime.datetime.now()).replace("T", " ")[:19]
        d["create_timestamp_flt"] = datetime.datetime.now().timestamp()

        # 处理包含图片的分块
        if not d.get("image"):
            docs.append(d)
            continue

        # 将图片保存到存储
        try:
            output_buffer = BytesIO()
            if isinstance(d["image"], bytes):
                output_buffer = BytesIO(d["image"])
            else:
                d["image"].save(output_buffer, format="JPEG")

            st = timer()
            STORAGE_IMPL.put(row["kb_id"], d["_id"], output_buffer.getvalue())
            el += timer() - st
        except Exception as e:
            cron_logger.error(str(e))
            traceback.print_exc()

        d["img_id"] = "{}-{}".format(row["kb_id"], d["_id"])
        del d["image"]
        docs.append(d)
    cron_logger.info("MINIO PUT({}):{}".format(row["name"], el))

    # 如果配置了自动关键词提取
    if row["parser_config"].get("auto_keywords", 0):
        callback(msg="Start to generate keywords for every chunk ...")
        # 创建一个LLMBundle实例用于关键词提取
        # - tenant_id: 租户ID,用于权限控制和token计费
        # - LLMType.CHAT: 使用聊天类型的语言模型
        # - llm_name: 使用指定的模型名称
        # - lang: 设置语言(中文/英文)
        chat_mdl = LLMBundle(
            row["tenant_id"], LLMType.CHAT, llm_name=row["llm_id"], lang=row["language"]
        )

        # 遍历所有文档块,为每个块提取关键词
        for d in docs:
            # 使用LLM模型从文档内容中提取关键词
            # - chat_mdl: LLM模型实例
            # - d["content_with_weight"]: 文档块的内容
            # - row["parser_config"]["auto_keywords"]: 关键词提取的配置参数
            # split(",")将返回的关键词字符串按逗号分割成列表
            d["important_kwd"] = keyword_extraction(
                chat_mdl,
                d["content_with_weight"],
                row["parser_config"]["auto_keywords"],
            ).split(",")

            # 对提取出的关键词进行分词处理
            # - 先用空格连接所有关键词
            # - 然后用分词器进行分词,生成token序列
            d["important_tks"] = rag_tokenizer.tokenize(" ".join(d["important_kwd"]))

    # 如果配置了自动问题生成
    if row["parser_config"].get("auto_questions", 0):
        callback(msg="Start to generate questions for every chunk ...")
        chat_mdl = LLMBundle(
            row["tenant_id"], LLMType.CHAT, llm_name=row["llm_id"], lang=row["language"]
        )
        for d in docs:
            # 使用LLM生成问题
            qst = question_proposal(
                chat_mdl,
                d["content_with_weight"],
                row["parser_config"]["auto_questions"],
            )
            # 将生成的问题添加到内容前面
            d["content_with_weight"] = (
                f"Question: \n{qst}\n\nAnswer:\n" + d["content_with_weight"]
            )
            qst = rag_tokenizer.tokenize(qst)
            # 更新token信息
            if "content_ltks" in d:
                d["content_ltks"] += " " + qst
            if "content_sm_ltks" in d:
                d["content_sm_ltks"] += " " + rag_tokenizer.fine_grained_tokenize(qst)

    return docs


def init_kb(row):
    idxnm = search.index_name(row["tenant_id"])
    if ELASTICSEARCH.indexExist(idxnm):
        return
    return ELASTICSEARCH.createIdx(
        idxnm,
        json.load(
            open(
                os.path.join(get_project_base_directory(), "conf", "mapping.json"), "r"
            )
        ),
    )


def embedding(docs, mdl, parser_config=None, callback=None):
    if parser_config is None:
        parser_config = {}
    batch_size = 32
    tts, cnts = (
        [rmSpace(d["title_tks"]) for d in docs if d.get("title_tks")],
        [
            re.sub(
                r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>",
                " ",
                d["content_with_weight"],
            )
            for d in docs
        ],
    )
    tk_count = 0
    # 如果标题和内容数量相等,则同时对标题和内容进行向量化处理
    if len(tts) == len(cnts):
        # 初始化标题向量数组
        tts_ = np.array([])
        # 分批处理标题文本
        for i in range(0, len(tts), batch_size):
            # 使用模型对当前批次的标题进行向量化编码
            vts, c = mdl.encode(tts[i : i + batch_size])
            # 将编码结果添加到标题向量数组中
            if len(tts_) == 0:
                tts_ = vts  # 第一批次直接赋值
            else:
                tts_ = np.concatenate((tts_, vts), axis=0)  # 后续批次进行拼接
            # 累加token使用量
            tk_count += c
            # 更新处理进度(60%-70%)
            callback(prog=0.6 + 0.1 * (i + 1) / len(tts), msg="")
        tts = tts_

    # 初始化内容向量数组
    cnts_ = np.array([])
    # 分批处理内容文本
    for i in range(0, len(cnts), batch_size):
        # 使用模型对当前批次的内容进行向量化编码
        vts, c = mdl.encode(cnts[i : i + batch_size])
        # 将编码结果添加到内容向量数组中
        if len(cnts_) == 0:
            cnts_ = vts  # 第一批次直接赋值
        else:
            cnts_ = np.concatenate((cnts_, vts), axis=0)  # 后续批次进行拼接
        # 累加token使用量
        tk_count += c
        # 更新处理进度(70%-90%)
        callback(prog=0.7 + 0.2 * (i + 1) / len(cnts), msg="")
    cnts = cnts_

    # 从配置中获取标题权重(默认0.1)
    title_w = float(parser_config.get("filename_embd_weight", 0.1))
    # 如果有标题向量,则将标题向量和内容向量按权重合并
    # 否则直接使用内容向量
    vects = (title_w * tts + (1 - title_w) * cnts) if len(tts) == len(cnts) else cnts

    assert len(vects) == len(docs)
    for i, d in enumerate(docs):
        v = vects[i].tolist()
        d["q_%d_vec" % len(v)] = v
    return tk_count


def run_raptor(row, chat_mdl, embd_mdl, callback=None):
    vts, _ = embd_mdl.encode(["ok"])
    vctr_nm = "q_%d_vec" % len(vts[0])
    chunks = []
    for d in retrievaler.chunk_list(
        row["doc_id"], row["tenant_id"], fields=["content_with_weight", vctr_nm]
    ):
        chunks.append((d["content_with_weight"], np.array(d[vctr_nm])))

    raptor = Raptor(
        row["parser_config"]["raptor"].get("max_cluster", 64),
        chat_mdl,
        embd_mdl,
        row["parser_config"]["raptor"]["prompt"],
        row["parser_config"]["raptor"]["max_token"],
        row["parser_config"]["raptor"]["threshold"],
    )
    original_length = len(chunks)
    raptor(chunks, row["parser_config"]["raptor"]["random_seed"], callback)
    doc = {
        "doc_id": row["doc_id"],
        "kb_id": [str(row["kb_id"])],
        "docnm_kwd": row["name"],
        "title_tks": rag_tokenizer.tokenize(row["name"]),
    }
    res = []
    tk_count = 0
    for content, vctr in chunks[original_length:]:
        d = copy.deepcopy(doc)
        md5 = hashlib.md5()
        md5.update((content + str(d["doc_id"])).encode("utf-8"))
        d["_id"] = md5.hexdigest()
        d["create_time"] = str(datetime.datetime.now()).replace("T", " ")[:19]
        d["create_timestamp_flt"] = datetime.datetime.now().timestamp()
        d[vctr_nm] = vctr.tolist()
        d["content_with_weight"] = content
        d["content_ltks"] = rag_tokenizer.tokenize(content)
        d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])
        res.append(d)
        tk_count += num_tokens_from_string(content)
    return res, tk_count


def main():
    rows = collect()
    if len(rows) == 0:
        return

    for _, r in rows.iterrows():
        callback = partial(set_progress, r["id"], r["from_page"], r["to_page"])
        try:
            embd_mdl = LLMBundle(
                r["tenant_id"],
                LLMType.EMBEDDING,
                llm_name=r["embd_id"],
                lang=r["language"],
            )
        except Exception as e:
            callback(-1, msg=str(e))
            cron_logger.error(str(e))
            continue

        if r.get("task_type", "") == "raptor":
            try:
                chat_mdl = LLMBundle(
                    r["tenant_id"],
                    LLMType.CHAT,
                    llm_name=r["llm_id"],
                    lang=r["language"],
                )
                cks, tk_count = run_raptor(r, chat_mdl, embd_mdl, callback)
            except Exception as e:
                callback(-1, msg=str(e))
                cron_logger.error(str(e))
                continue
        else:
            st = timer()
            cks = build(r)
            cron_logger.info("Build chunks({}): {}".format(r["name"], timer() - st))
            if cks is None:
                continue
            if not cks:
                callback(1.0, "No chunk! Done!")
                continue
            # TODO: exception handler
            ## set_progress(r["did"], -1, "ERROR: ")
            callback(
                msg="Finished slicing files(%d). Start to embedding the content."
                % len(cks)
            )
            st = timer()
            try:
                tk_count = embedding(cks, embd_mdl, r["parser_config"], callback)
            except Exception as e:
                callback(-1, "Embedding error:{}".format(str(e)))
                cron_logger.error(str(e))
                tk_count = 0
            cron_logger.info(
                "Embedding elapsed({}): {:.2f}".format(r["name"], timer() - st)
            )
            callback(
                msg="Finished embedding({:.2f})! Start to build index!".format(
                    timer() - st
                )
            )

        init_kb(r)
        chunk_count = len(set([c["_id"] for c in cks]))
        st = timer()
        es_r = ""
        es_bulk_size = 4
        for b in range(0, len(cks), es_bulk_size):
            es_r = ELASTICSEARCH.bulk(
                cks[b : b + es_bulk_size], search.index_name(r["tenant_id"])
            )
            if b % 128 == 0:
                callback(prog=0.8 + 0.1 * (b + 1) / len(cks), msg="")

        cron_logger.info("Indexing elapsed({}): {:.2f}".format(r["name"], timer() - st))
        if es_r:
            callback(
                -1,
                "Insert chunk error, detail info please check ragflow-logs/api/cron_logger.log. Please also check ES status!",
            )
            ELASTICSEARCH.deleteByQuery(
                Q("match", doc_id=r["doc_id"]), idxnm=search.index_name(r["tenant_id"])
            )
            cron_logger.error(str(es_r))
        else:
            if TaskService.do_cancel(r["id"]):
                ELASTICSEARCH.deleteByQuery(
                    Q("match", doc_id=r["doc_id"]),
                    idxnm=search.index_name(r["tenant_id"]),
                )
                continue
            callback(1.0, "Done!")
            DocumentService.increment_chunk_num(
                r["doc_id"], r["kb_id"], tk_count, chunk_count, 0
            )
            cron_logger.info(
                "Chunk doc({}), token({}), chunks({}), elapsed:{:.2f}".format(
                    r["id"], tk_count, len(cks), timer() - st
                )
            )


def report_status():
    global CONSUMER_NAME
    while True:
        try:
            obj = REDIS_CONN.get("TASKEXE")
            if not obj:
                obj = {}
            else:
                obj = json.loads(obj)
            if CONSUMER_NAME not in obj:
                obj[CONSUMER_NAME] = []
            obj[CONSUMER_NAME].append(timer())
            obj[CONSUMER_NAME] = obj[CONSUMER_NAME][-60:]
            REDIS_CONN.set_obj("TASKEXE", obj, 60 * 2)
        except Exception as e:
            print("[Exception]:", str(e))
        time.sleep(30)


if __name__ == "__main__":
    peewee_logger = logging.getLogger("peewee")
    peewee_logger.propagate = False
    peewee_logger.addHandler(database_logger.handlers[0])
    peewee_logger.setLevel(database_logger.level)

    exe = ThreadPoolExecutor(max_workers=1)
    exe.submit(report_status)

    while True:
        main()
        if PAYLOAD:
            PAYLOAD.ack()
            PAYLOAD = None
