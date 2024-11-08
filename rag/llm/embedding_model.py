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
import re
from typing import Optional
import threading
import requests
from huggingface_hub import snapshot_download
from zhipuai import ZhipuAI
import os
from abc import ABC
from ollama import Client
import dashscope
from openai import OpenAI
import numpy as np
import asyncio

from api.settings import LIGHTEN
from api.utils.file_utils import get_home_cache_dir
from rag.utils import num_tokens_from_string, truncate
import google.generativeai as genai
import json


class Base(ABC):
    def __init__(self, key, model_name):
        pass

    def encode(self, texts: list, batch_size=32):
        raise NotImplementedError("Please implement encode method!")

    def encode_queries(self, text: str):
        raise NotImplementedError("Please implement encode method!")


class DefaultEmbedding(Base):
    """默认的文本嵌入模型类,使用 FlagEmbedding 模型

    该类实现了文本嵌入的基本功能,包括批量文本编码和单条查询编码。
    使用单例模式加载模型以节省内存,支持 GPU 加速。

    Attributes:
        _model: FlagModel 实例,用于生成文本嵌入
        _model_lock: 线程锁,用于确保模型单例的线程安全
    """

    _model = None
    _model_lock = threading.Lock()

    def __init__(self, key, model_name, **kwargs):
        """初始化 DefaultEmbedding 实例

        Args:
            key: API密钥(本地模型未使用)
            model_name: 模型名称
            **kwargs: 额外参数

        Note:
            如果从 HuggingFace 下载模型遇到问题,可以设置镜像:
            Linux: export HF_ENDPOINT=https://hf-mirror.com
        """
        # 非轻量级模式且模型未加载时初始化模型
        if not LIGHTEN and not DefaultEmbedding._model:
            with DefaultEmbedding._model_lock:  # 使用线程锁确保线程安全
                from FlagEmbedding import FlagModel
                import torch

                if not DefaultEmbedding._model:
                    try:
                        # 尝试直接加载本地模型
                        DefaultEmbedding._model = FlagModel(
                            os.path.join(
                                get_home_cache_dir(),
                                re.sub(r"^[a-zA-Z]+/", "", model_name),
                            ),
                            query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                            use_fp16=torch.cuda.is_available(),  # GPU可用时使用FP16加速
                        )
                    except Exception as e:
                        # 本地加载失败则从HuggingFace下载
                        model_dir = snapshot_download(
                            repo_id="BAAI/bge-large-zh-v1.5",
                            local_dir=os.path.join(
                                get_home_cache_dir(),
                                re.sub(r"^[a-zA-Z]+/", "", model_name),
                            ),
                            local_dir_use_symlinks=False,
                        )
                        DefaultEmbedding._model = FlagModel(
                            model_dir,
                            query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                            use_fp16=torch.cuda.is_available(),
                        )
        self._model = DefaultEmbedding._model

    def encode(self, texts: list, batch_size=32):
        """批量编码文本序列

        Args:
            texts: 待编码的文本列表
            batch_size: 批处理大小,默认32

        Returns:
            tuple: (编码结果数组, token总数)
                - 编码结果为numpy数组,shape为(len(texts), embedding_dim)
                - token总数为所有文本的token数之和
        """
        # 截断文本,限制最大长度为2048
        texts = [truncate(t, 2048) for t in texts]

        # 计算token总数
        token_count = 0
        for t in texts:
            token_count += num_tokens_from_string(t)

        # 分批编码文本
        res = []
        for i in range(0, len(texts), batch_size):
            res.extend(self._model.encode(texts[i : i + batch_size]).tolist())

        return np.array(res), token_count

    def encode_queries(self, text: str):
        """编码单条查询文本

        Args:
            text: 待编码的查询文本

        Returns:
            tuple: (编码结果向量, token数)
                - 编码结果为一维numpy数组
                - token数为该文本的token数
        """
        token_count = num_tokens_from_string(text)
        return self._model.encode_queries([text]).tolist()[0], token_count


class OpenAIEmbed(Base):
    def __init__(
        self,
        key,
        model_name="text-embedding-ada-002",
        base_url="https://api.openai.com/v1",
    ):
        if not base_url:
            base_url = "https://api.openai.com/v1"
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model_name = model_name

    def encode(self, texts: list, batch_size=32):
        texts = [truncate(t, 8191) for t in texts]
        res = self.client.embeddings.create(input=texts, model=self.model_name)
        return np.array([d.embedding for d in res.data]), res.usage.total_tokens

    def encode_queries(self, text):
        res = self.client.embeddings.create(
            input=[truncate(text, 8191)], model=self.model_name
        )
        return np.array(res.data[0].embedding), res.usage.total_tokens


class LocalAIEmbed(Base):
    def __init__(self, key, model_name, base_url):
        if not base_url:
            raise ValueError("Local embedding model url cannot be None")
        if base_url.split("/")[-1] != "v1":
            base_url = os.path.join(base_url, "v1")
        self.client = OpenAI(api_key="empty", base_url=base_url)
        self.model_name = model_name.split("___")[0]

    def encode(self, texts: list, batch_size=32):
        res = self.client.embeddings.create(input=texts, model=self.model_name)
        return (
            np.array([d.embedding for d in res.data]),
            1024,
        )  # local embedding for LmStudio donot count tokens

    def encode_queries(self, text):
        embds, cnt = self.encode([text])
        return np.array(embds[0]), cnt


class AzureEmbed(OpenAIEmbed):
    def __init__(self, key, model_name, **kwargs):
        from openai.lib.azure import AzureOpenAI

        api_key = json.loads(key).get("api_key", "")
        api_version = json.loads(key).get("api_version", "2024-02-01")
        self.client = AzureOpenAI(
            api_key=api_key, azure_endpoint=kwargs["base_url"], api_version=api_version
        )
        self.model_name = model_name


class BaiChuanEmbed(OpenAIEmbed):
    def __init__(
        self,
        key,
        model_name="Baichuan-Text-Embedding",
        base_url="https://api.baichuan-ai.com/v1",
    ):
        if not base_url:
            base_url = "https://api.baichuan-ai.com/v1"
        super().__init__(key, model_name, base_url)


class QWenEmbed(Base):
    def __init__(self, key, model_name="text_embedding_v2", **kwargs):
        dashscope.api_key = key
        self.model_name = model_name

    def encode(self, texts: list, batch_size=10):
        import dashscope

        batch_size = min(batch_size, 4)
        try:
            res = []
            token_count = 0
            texts = [truncate(t, 2048) for t in texts]
            for i in range(0, len(texts), batch_size):
                resp = dashscope.TextEmbedding.call(
                    model=self.model_name,
                    input=texts[i : i + batch_size],
                    text_type="document",
                )
                embds = [[] for _ in range(len(resp["output"]["embeddings"]))]
                for e in resp["output"]["embeddings"]:
                    embds[e["text_index"]] = e["embedding"]
                res.extend(embds)
                token_count += resp["usage"]["total_tokens"]
            return np.array(res), token_count
        except Exception as e:
            raise Exception(
                "Account abnormal. Please ensure it's on good standing to use QWen's "
                + self.model_name
            )
        return np.array([]), 0

    def encode_queries(self, text):
        try:
            resp = dashscope.TextEmbedding.call(
                model=self.model_name, input=text[:2048], text_type="query"
            )
            return np.array(resp["output"]["embeddings"][0]["embedding"]), resp[
                "usage"
            ]["total_tokens"]
        except Exception as e:
            raise Exception(
                "Account abnormal. Please ensure it's on good standing to use QWen's "
                + self.model_name
            )
        return np.array([]), 0


class ZhipuEmbed(Base):
    def __init__(self, key, model_name="embedding-2", **kwargs):
        self.client = ZhipuAI(api_key=key)
        self.model_name = model_name

    def encode(self, texts: list, batch_size=32):
        arr = []
        tks_num = 0
        for txt in texts:
            res = self.client.embeddings.create(input=txt, model=self.model_name)
            arr.append(res.data[0].embedding)
            tks_num += res.usage.total_tokens
        return np.array(arr), tks_num

    def encode_queries(self, text):
        res = self.client.embeddings.create(input=text, model=self.model_name)
        return np.array(res.data[0].embedding), res.usage.total_tokens


class OllamaEmbed(Base):
    def __init__(self, key, model_name, **kwargs):
        self.client = Client(host=kwargs["base_url"])
        self.model_name = model_name

    def encode(self, texts: list, batch_size=32):
        arr = []
        tks_num = 0
        for txt in texts:
            res = self.client.embeddings(prompt=txt, model=self.model_name)
            arr.append(res["embedding"])
            tks_num += 128
        return np.array(arr), tks_num

    def encode_queries(self, text):
        res = self.client.embeddings(prompt=text, model=self.model_name)
        return np.array(res["embedding"]), 128


class FastEmbed(Base):
    _model = None

    def __init__(
        self,
        key: Optional[str] = None,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        **kwargs,
    ):
        if not LIGHTEN and not FastEmbed._model:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name, cache_dir, threads, **kwargs)

    def encode(self, texts: list, batch_size=32):
        # Using the internal tokenizer to encode the texts and get the total
        # number of tokens
        encodings = self._model.model.tokenizer.encode_batch(texts)
        total_tokens = sum(len(e) for e in encodings)

        embeddings = [e.tolist() for e in self._model.embed(texts, batch_size)]

        return np.array(embeddings), total_tokens

    def encode_queries(self, text: str):
        # Using the internal tokenizer to encode the texts and get the total
        # number of tokens
        encoding = self._model.model.tokenizer.encode(text)
        embedding = next(self._model.query_embed(text)).tolist()

        return np.array(embedding), len(encoding.ids)


class XinferenceEmbed(Base):
    def __init__(self, key, model_name="", base_url=""):
        if base_url.split("/")[-1] != "v1":
            base_url = os.path.join(base_url, "v1")
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model_name = model_name

    def encode(self, texts: list, batch_size=32):
        res = self.client.embeddings.create(input=texts, model=self.model_name)
        return np.array([d.embedding for d in res.data]), res.usage.total_tokens

    def encode_queries(self, text):
        res = self.client.embeddings.create(input=[text], model=self.model_name)
        return np.array(res.data[0].embedding), res.usage.total_tokens


class YoudaoEmbed(Base):
    _client = None

    def __init__(
        self, key=None, model_name="maidalun1020/bce-embedding-base_v1", **kwargs
    ):
        if not LIGHTEN and not YoudaoEmbed._client:
            from BCEmbedding import EmbeddingModel as qanthing

            try:
                print("LOADING BCE...")
                YoudaoEmbed._client = qanthing(
                    model_name_or_path=os.path.join(
                        get_home_cache_dir(), "bce-embedding-base_v1"
                    )
                )
            except Exception as e:
                YoudaoEmbed._client = qanthing(
                    model_name_or_path=model_name.replace("maidalun1020", "InfiniFlow")
                )

    def encode(self, texts: list, batch_size=10):
        res = []
        token_count = 0
        for t in texts:
            token_count += num_tokens_from_string(t)
        for i in range(0, len(texts), batch_size):
            embds = YoudaoEmbed._client.encode(texts[i : i + batch_size])
            res.extend(embds)
        return np.array(res), token_count

    def encode_queries(self, text):
        embds = YoudaoEmbed._client.encode([text])
        return np.array(embds[0]), num_tokens_from_string(text)


class JinaEmbed(Base):
    def __init__(
        self,
        key,
        model_name="jina-embeddings-v2-base-zh",
        base_url="https://api.jina.ai/v1/embeddings",
    ):
        self.base_url = "https://api.jina.ai/v1/embeddings"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
        self.model_name = model_name

    def encode(self, texts: list, batch_size=None):
        texts = [truncate(t, 8196) for t in texts]
        data = {"model": self.model_name, "input": texts, "encoding_type": "float"}
        res = requests.post(self.base_url, headers=self.headers, json=data).json()
        return np.array([d["embedding"] for d in res["data"]]), res["usage"][
            "total_tokens"
        ]

    def encode_queries(self, text):
        embds, cnt = self.encode([text])
        return np.array(embds[0]), cnt


class InfinityEmbed(Base):
    _model = None

    def __init__(
        self,
        model_names: list[str] = ("BAAI/bge-small-en-v1.5",),
        engine_kwargs: dict = {},
        key=None,
    ):
        from infinity_emb import EngineArgs
        from infinity_emb.engine import AsyncEngineArray

        self._default_model = model_names[0]
        self.engine_array = AsyncEngineArray.from_args(
            [
                EngineArgs(model_name_or_path=model_name, **engine_kwargs)
                for model_name in model_names
            ]
        )

    async def _embed(self, sentences: list[str], model_name: str = ""):
        if not model_name:
            model_name = self._default_model
        engine = self.engine_array[model_name]
        was_already_running = engine.is_running
        if not was_already_running:
            await engine.astart()
        embeddings, usage = await engine.embed(sentences=sentences)
        if not was_already_running:
            await engine.astop()
        return embeddings, usage

    def encode(self, texts: list[str], model_name: str = "") -> tuple[np.ndarray, int]:
        # Using the internal tokenizer to encode the texts and get the total
        # number of tokens
        embeddings, usage = asyncio.run(self._embed(texts, model_name))
        return np.array(embeddings), usage

    def encode_queries(self, text: str) -> tuple[np.ndarray, int]:
        # Using the internal tokenizer to encode the texts and get the total
        # number of tokens
        return self.encode([text])


class MistralEmbed(Base):
    def __init__(self, key, model_name="mistral-embed", base_url=None):
        from mistralai.client import MistralClient

        self.client = MistralClient(api_key=key)
        self.model_name = model_name

    def encode(self, texts: list, batch_size=32):
        texts = [truncate(t, 8196) for t in texts]
        res = self.client.embeddings(input=texts, model=self.model_name)
        return np.array([d.embedding for d in res.data]), res.usage.total_tokens

    def encode_queries(self, text):
        res = self.client.embeddings(
            input=[truncate(text, 8196)], model=self.model_name
        )
        return np.array(res.data[0].embedding), res.usage.total_tokens


class BedrockEmbed(Base):
    def __init__(self, key, model_name, **kwargs):
        import boto3

        self.bedrock_ak = json.loads(key).get("bedrock_ak", "")
        self.bedrock_sk = json.loads(key).get("bedrock_sk", "")
        self.bedrock_region = json.loads(key).get("bedrock_region", "")
        self.model_name = model_name
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=self.bedrock_region,
            aws_access_key_id=self.bedrock_ak,
            aws_secret_access_key=self.bedrock_sk,
        )

    def encode(self, texts: list, batch_size=32):
        texts = [truncate(t, 8196) for t in texts]
        embeddings = []
        token_count = 0
        for text in texts:
            if self.model_name.split(".")[0] == "amazon":
                body = {"inputText": text}
            elif self.model_name.split(".")[0] == "cohere":
                body = {"texts": [text], "input_type": "search_document"}

            response = self.client.invoke_model(
                modelId=self.model_name, body=json.dumps(body)
            )
            model_response = json.loads(response["body"].read())
            embeddings.extend([model_response["embedding"]])
            token_count += num_tokens_from_string(text)

        return np.array(embeddings), token_count

    def encode_queries(self, text):
        embeddings = []
        token_count = num_tokens_from_string(text)
        if self.model_name.split(".")[0] == "amazon":
            body = {"inputText": truncate(text, 8196)}
        elif self.model_name.split(".")[0] == "cohere":
            body = {"texts": [truncate(text, 8196)], "input_type": "search_query"}

        response = self.client.invoke_model(
            modelId=self.model_name, body=json.dumps(body)
        )
        model_response = json.loads(response["body"].read())
        embeddings.extend(model_response["embedding"])

        return np.array(embeddings), token_count


class GeminiEmbed(Base):
    def __init__(self, key, model_name="models/text-embedding-004", **kwargs):
        genai.configure(api_key=key)
        self.model_name = "models/" + model_name

    def encode(self, texts: list, batch_size=32):
        texts = [truncate(t, 2048) for t in texts]
        token_count = sum(num_tokens_from_string(text) for text in texts)
        result = genai.embed_content(
            model=self.model_name,
            content=texts,
            task_type="retrieval_document",
            title="Embedding of list of strings",
        )
        return np.array(result["embedding"]), token_count

    def encode_queries(self, text):
        result = genai.embed_content(
            model=self.model_name,
            content=truncate(text, 2048),
            task_type="retrieval_document",
            title="Embedding of single string",
        )
        token_count = num_tokens_from_string(text)
        return np.array(result["embedding"]), token_count


class NvidiaEmbed(Base):
    def __init__(
        self, key, model_name, base_url="https://integrate.api.nvidia.com/v1/embeddings"
    ):
        if not base_url:
            base_url = "https://integrate.api.nvidia.com/v1/embeddings"
        self.api_key = key
        self.base_url = base_url
        self.headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "authorization": f"Bearer {self.api_key}",
        }
        self.model_name = model_name
        if model_name == "nvidia/embed-qa-4":
            self.base_url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/embeddings"
            self.model_name = "NV-Embed-QA"
        if model_name == "snowflake/arctic-embed-l":
            self.base_url = "https://ai.api.nvidia.com/v1/retrieval/snowflake/arctic-embed-l/embeddings"

    def encode(self, texts: list, batch_size=None):
        payload = {
            "input": texts,
            "input_type": "query",
            "model": self.model_name,
            "encoding_format": "float",
            "truncate": "END",
        }
        res = requests.post(self.base_url, headers=self.headers, json=payload).json()
        return (
            np.array([d["embedding"] for d in res["data"]]),
            res["usage"]["total_tokens"],
        )

    def encode_queries(self, text):
        embds, cnt = self.encode([text])
        return np.array(embds[0]), cnt


class LmStudioEmbed(LocalAIEmbed):
    def __init__(self, key, model_name, base_url):
        if not base_url:
            raise ValueError("Local llm url cannot be None")
        if base_url.split("/")[-1] != "v1":
            base_url = os.path.join(base_url, "v1")
        self.client = OpenAI(api_key="lm-studio", base_url=base_url)
        self.model_name = model_name


class OpenAI_APIEmbed(OpenAIEmbed):
    def __init__(self, key, model_name, base_url):
        if not base_url:
            raise ValueError("url cannot be None")
        if base_url.split("/")[-1] != "v1":
            base_url = os.path.join(base_url, "v1")
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model_name = model_name.split("___")[0]


class CoHereEmbed(Base):
    def __init__(self, key, model_name, base_url=None):
        from cohere import Client

        self.client = Client(api_key=key)
        self.model_name = model_name

    def encode(self, texts: list, batch_size=32):
        res = self.client.embed(
            texts=texts,
            model=self.model_name,
            input_type="search_query",
            embedding_types=["float"],
        )
        return np.array([d for d in res.embeddings.float]), int(
            res.meta.billed_units.input_tokens
        )

    def encode_queries(self, text):
        res = self.client.embed(
            texts=[text],
            model=self.model_name,
            input_type="search_query",
            embedding_types=["float"],
        )
        return np.array([d for d in res.embeddings.float]), int(
            res.meta.billed_units.input_tokens
        )


class TogetherAIEmbed(OllamaEmbed):
    def __init__(self, key, model_name, base_url="https://api.together.xyz/v1"):
        if not base_url:
            base_url = "https://api.together.xyz/v1"
        super().__init__(key, model_name, base_url)


class PerfXCloudEmbed(OpenAIEmbed):
    def __init__(self, key, model_name, base_url="https://cloud.perfxlab.cn/v1"):
        if not base_url:
            base_url = "https://cloud.perfxlab.cn/v1"
        super().__init__(key, model_name, base_url)


class UpstageEmbed(OpenAIEmbed):
    def __init__(self, key, model_name, base_url="https://api.upstage.ai/v1/solar"):
        if not base_url:
            base_url = "https://api.upstage.ai/v1/solar"
        super().__init__(key, model_name, base_url)


class SILICONFLOWEmbed(Base):
    def __init__(
        self, key, model_name, base_url="https://api.siliconflow.cn/v1/embeddings"
    ):
        if not base_url:
            base_url = "https://api.siliconflow.cn/v1/embeddings"
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {key}",
        }
        self.base_url = base_url
        self.model_name = model_name

    def encode(self, texts: list, batch_size=32):
        payload = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float",
        }
        res = requests.post(self.base_url, json=payload, headers=self.headers).json()
        return (
            np.array([d["embedding"] for d in res["data"]]),
            res["usage"]["total_tokens"],
        )

    def encode_queries(self, text):
        payload = {
            "model": self.model_name,
            "input": text,
            "encoding_format": "float",
        }
        res = requests.post(self.base_url, json=payload, headers=self.headers).json()
        return np.array(res["data"][0]["embedding"]), res["usage"]["total_tokens"]


class ReplicateEmbed(Base):
    def __init__(self, key, model_name, base_url=None):
        from replicate.client import Client

        self.model_name = model_name
        self.client = Client(api_token=key)

    def encode(self, texts: list, batch_size=32):
        res = self.client.run(self.model_name, input={"texts": json.dumps(texts)})
        return np.array(res), sum([num_tokens_from_string(text) for text in texts])

    def encode_queries(self, text):
        res = self.client.embed(self.model_name, input={"texts": [text]})
        return np.array(res), num_tokens_from_string(text)


class BaiduYiyanEmbed(Base):
    def __init__(self, key, model_name, base_url=None):
        import qianfan

        key = json.loads(key)
        ak = key.get("yiyan_ak", "")
        sk = key.get("yiyan_sk", "")
        self.client = qianfan.Embedding(ak=ak, sk=sk)
        self.model_name = model_name

    def encode(self, texts: list, batch_size=32):
        res = self.client.do(model=self.model_name, texts=texts).body
        return (
            np.array([r["embedding"] for r in res["data"]]),
            res["usage"]["total_tokens"],
        )

    def encode_queries(self, text):
        res = self.client.do(model=self.model_name, texts=[text]).body
        return (
            np.array([r["embedding"] for r in res["data"]]),
            res["usage"]["total_tokens"],
        )


class VoyageEmbed(Base):
    def __init__(self, key, model_name, base_url=None):
        import voyageai

        self.client = voyageai.Client(api_key=key)
        self.model_name = model_name

    def encode(self, texts: list, batch_size=32):
        res = self.client.embed(
            texts=texts, model=self.model_name, input_type="document"
        )
        return np.array(res.embeddings), res.total_tokens

    def encode_queries(self, text):
        res = self.client.embed
        res = self.client.embed(texts=text, model=self.model_name, input_type="query")
        return np.array(res.embeddings), res.total_tokens


class HuggingFaceEmbed(Base):
    def __init__(self, key, model_name, base_url=None):
        if not model_name:
            raise ValueError("Model name cannot be None")
        self.key = key
        self.model_name = model_name
        self.base_url = base_url or "http://127.0.0.1:8080"

    def encode(self, texts: list, batch_size=32):
        embeddings = []
        for text in texts:
            response = requests.post(
                f"{self.base_url}/embed",
                json={"inputs": text},
                headers={"Content-Type": "application/json"},
            )
            if response.status_code == 200:
                embedding = response.json()
                embeddings.append(embedding[0])
            else:
                raise Exception(f"Error: {response.status_code} - {response.text}")
        return np.array(embeddings), sum(
            [num_tokens_from_string(text) for text in texts]
        )

    def encode_queries(self, text):
        response = requests.post(
            f"{self.base_url}/embed",
            json={"inputs": text},
            headers={"Content-Type": "application/json"},
        )
        if response.status_code == 200:
            embedding = response.json()
            return np.array(embedding[0]), num_tokens_from_string(text)
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")
