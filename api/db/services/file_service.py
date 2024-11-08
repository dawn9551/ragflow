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
import os
from flask_login import current_user
from peewee import fn

from api.db import FileType, KNOWLEDGEBASE_FOLDER_NAME, FileSource, ParserType
from api.db.db_models import DB, File2Document, Knowledgebase
from api.db.db_models import File, Document
from api.db.services import duplicate_name
from api.db.services.common_service import CommonService
from api.db.services.document_service import DocumentService
from api.db.services.file2document_service import File2DocumentService
from api.utils import get_uuid
from api.utils.file_utils import filename_type, thumbnail_img
from rag.utils.storage_factory import STORAGE_IMPL


class FileService(CommonService):
    model = File

    @classmethod
    @DB.connection_context()
    def get_by_pf_id(
        cls, tenant_id, pf_id, page_number, items_per_page, orderby, desc, keywords
    ):
        if keywords:
            files = cls.model.select().where(
                (cls.model.tenant_id == tenant_id),
                (cls.model.parent_id == pf_id),
                (fn.LOWER(cls.model.name).contains(keywords.lower())),
                ~(cls.model.id == pf_id),
            )
        else:
            files = cls.model.select().where(
                (cls.model.tenant_id == tenant_id),
                (cls.model.parent_id == pf_id),
                ~(cls.model.id == pf_id),
            )
        count = files.count()
        if desc:
            files = files.order_by(cls.model.getter_by(orderby).desc())
        else:
            files = files.order_by(cls.model.getter_by(orderby).asc())

        files = files.paginate(page_number, items_per_page)

        res_files = list(files.dicts())
        for file in res_files:
            if file["type"] == FileType.FOLDER.value:
                file["size"] = cls.get_folder_size(file["id"])
                file["kbs_info"] = []
                children = list(
                    cls.model.select()
                    .where(
                        (cls.model.tenant_id == tenant_id),
                        (cls.model.parent_id == file["id"]),
                        ~(cls.model.id == file["id"]),
                    )
                    .dicts()
                )
                file["has_child_folder"] = any(
                    value["type"] == FileType.FOLDER.value for value in children
                )
                continue
            kbs_info = cls.get_kb_id_by_file_id(file["id"])
            file["kbs_info"] = kbs_info

        return res_files, count

    @classmethod
    @DB.connection_context()
    def get_kb_id_by_file_id(cls, file_id):
        kbs = (
            cls.model.select(*[Knowledgebase.id, Knowledgebase.name])
            .join(File2Document, on=(File2Document.file_id == file_id))
            .join(Document, on=(File2Document.document_id == Document.id))
            .join(Knowledgebase, on=(Knowledgebase.id == Document.kb_id))
            .where(cls.model.id == file_id)
        )
        if not kbs:
            return []
        kbs_info_list = []
        for kb in list(kbs.dicts()):
            kbs_info_list.append({"kb_id": kb["id"], "kb_name": kb["name"]})
        return kbs_info_list

    @classmethod
    @DB.connection_context()
    def get_by_pf_id_name(cls, id, name):
        file = cls.model.select().where(
            (cls.model.parent_id == id) & (cls.model.name == name)
        )
        if file.count():
            e, file = cls.get_by_id(file[0].id)
            if not e:
                raise RuntimeError("Database error (File retrieval)!")
            return file
        return None

    @classmethod
    @DB.connection_context()
    def get_id_list_by_id(cls, id, name, count, res):
        if count < len(name):
            file = cls.get_by_pf_id_name(id, name[count])
            if file:
                res.append(file.id)
                return cls.get_id_list_by_id(file.id, name, count + 1, res)
            else:
                return res
        else:
            return res

    @classmethod
    @DB.connection_context()
    def get_all_innermost_file_ids(cls, folder_id, result_ids):
        subfolders = cls.model.select().where(cls.model.parent_id == folder_id)
        if subfolders.exists():
            for subfolder in subfolders:
                cls.get_all_innermost_file_ids(subfolder.id, result_ids)
        else:
            result_ids.append(folder_id)
        return result_ids

    @classmethod
    @DB.connection_context()
    def create_folder(cls, file, parent_id, name, count):
        if count > len(name) - 2:
            return file
        else:
            file = cls.insert(
                {
                    "id": get_uuid(),
                    "parent_id": parent_id,
                    "tenant_id": current_user.id,
                    "created_by": current_user.id,
                    "name": name[count],
                    "location": "",
                    "size": 0,
                    "type": FileType.FOLDER.value,
                }
            )
            return cls.create_folder(file, file.id, name, count + 1)

    @classmethod
    @DB.connection_context()
    def is_parent_folder_exist(cls, parent_id):
        parent_files = cls.model.select().where(cls.model.id == parent_id)
        if parent_files.count():
            return True
        cls.delete_folder_by_pf_id(parent_id)
        return False

    @classmethod
    @DB.connection_context()
    def get_root_folder(cls, tenant_id):
        for file in cls.model.select().where(
            (cls.model.tenant_id == tenant_id), (cls.model.parent_id == cls.model.id)
        ):
            return file.to_dict()

        file_id = get_uuid()
        file = {
            "id": file_id,
            "parent_id": file_id,
            "tenant_id": tenant_id,
            "created_by": tenant_id,
            "name": "/",
            "type": FileType.FOLDER.value,
            "size": 0,
            "location": "",
        }
        cls.save(**file)
        return file

    @classmethod
    @DB.connection_context()
    def get_kb_folder(cls, tenant_id):
        for root in cls.model.select().where(
            (cls.model.tenant_id == tenant_id), (cls.model.parent_id == cls.model.id)
        ):
            for folder in cls.model.select().where(
                (cls.model.tenant_id == tenant_id),
                (cls.model.parent_id == root.id),
                (cls.model.name == KNOWLEDGEBASE_FOLDER_NAME),
            ):
                return folder.to_dict()
        assert False, "Can't find the KB folder. Database init error."

    @classmethod
    @DB.connection_context()
    def new_a_file_from_kb(
        cls, tenant_id, name, parent_id, ty=FileType.FOLDER.value, size=0, location=""
    ):
        """从知识库创建一个新文件

        Args:
            tenant_id (str): 租户ID
            name (str): 文件名称
            parent_id (str): 父文件夹ID
            ty (int, optional): 文件类型, 默认为文件夹
            size (int, optional): 文件大小. 默认为 0
            location (str, optional): 文件位置. 默认为 ""

        Returns:
            dict: 创建的文件信息字典,如果文件已存在则返回已有文件信息
        """
        # 检查是否已存在同名文件
        for file in cls.query(tenant_id=tenant_id, parent_id=parent_id, name=name):
            return file.to_dict()
        file = {
            "id": get_uuid(),  # 生成唯一ID
            "parent_id": parent_id,  # 父文件夹ID
            "tenant_id": tenant_id,  # 租户ID
            "created_by": tenant_id,  # 创建者ID
            "name": name,  # 文件名
            "type": ty,  # 文件类型
            "size": size,  # 文件大小
            "location": location,  # 存储位置
            "source_type": FileSource.KNOWLEDGEBASE,  # 来源类型：知识库
        }
        cls.save(**file)
        return file

    @classmethod
    @DB.connection_context()
    def init_knowledgebase_docs(cls, root_id, tenant_id):
        """
        初始化知识库文档的文件夹结构，建立一个层级化的文档管理系统。
        """
        # 检查是否已存在知识库文件夹
        for _ in cls.model.select().where(
            (cls.model.name == KNOWLEDGEBASE_FOLDER_NAME)
            & (cls.model.parent_id == root_id)
        ):
            return  # 如果存在，直接返回，避免重复创建
        # 创建知识库根文件夹
        folder = cls.new_a_file_from_kb(tenant_id, KNOWLEDGEBASE_FOLDER_NAME, root_id)
        # 遍历所有知识库
        for kb in Knowledgebase.select(*[Knowledgebase.id, Knowledgebase.name]).where(
            Knowledgebase.tenant_id == tenant_id
        ):  # 为每个知识库创建独立文件夹
            kb_folder = cls.new_a_file_from_kb(tenant_id, kb.name, folder["id"])
            # 将文档添加到对应的知识库文件夹中
            for doc in DocumentService.query(kb_id=kb.id):
                FileService.add_file_from_kb(doc.to_dict(), kb_folder["id"], tenant_id)

    @classmethod
    @DB.connection_context()
    def get_parent_folder(cls, file_id):
        file = cls.model.select().where(cls.model.id == file_id)
        if file.count():
            e, file = cls.get_by_id(file[0].parent_id)
            if not e:
                raise RuntimeError("Database error (File retrieval)!")
        else:
            raise RuntimeError("Database error (File doesn't exist)!")
        return file

    @classmethod
    @DB.connection_context()
    def get_all_parent_folders(cls, start_id):
        parent_folders = []
        current_id = start_id
        while current_id:
            e, file = cls.get_by_id(current_id)
            if file.parent_id != file.id and e:
                parent_folders.append(file)
                current_id = file.parent_id
            else:
                parent_folders.append(file)
                break
        return parent_folders

    @classmethod
    @DB.connection_context()
    def insert(cls, file):
        if not cls.save(**file):
            raise RuntimeError("Database error (File)!")
        e, file = cls.get_by_id(file["id"])
        if not e:
            raise RuntimeError("Database error (File retrieval)!")
        return file

    @classmethod
    @DB.connection_context()
    def delete(cls, file):
        return cls.delete_by_id(file.id)

    @classmethod
    @DB.connection_context()
    def delete_by_pf_id(cls, folder_id):
        return cls.model.delete().where(cls.model.parent_id == folder_id).execute()

    @classmethod
    @DB.connection_context()
    def delete_folder_by_pf_id(cls, user_id, folder_id):
        try:
            files = cls.model.select().where(
                (cls.model.tenant_id == user_id) & (cls.model.parent_id == folder_id)
            )
            for file in files:
                cls.delete_folder_by_pf_id(user_id, file.id)
            return (
                cls.model.delete()
                .where((cls.model.tenant_id == user_id) & (cls.model.id == folder_id))
                .execute(),
            )
        except Exception as e:
            print(e)
            raise RuntimeError("Database error (File retrieval)!")

    @classmethod
    @DB.connection_context()
    def get_file_count(cls, tenant_id):
        files = cls.model.select(cls.model.id).where(cls.model.tenant_id == tenant_id)
        return len(files)

    @classmethod
    @DB.connection_context()
    def get_folder_size(cls, folder_id):
        size = 0

        def dfs(parent_id):
            nonlocal size
            for f in cls.model.select(
                *[cls.model.id, cls.model.size, cls.model.type]
            ).where(cls.model.parent_id == parent_id, cls.model.id != parent_id):
                size += f.size
                if f.type == FileType.FOLDER.value:
                    dfs(f.id)

        dfs(folder_id)
        return size

    @classmethod
    @DB.connection_context()
    def add_file_from_kb(cls, doc, kb_folder_id, tenant_id):
        for _ in File2DocumentService.get_by_document_id(doc["id"]):
            return
        file = {
            "id": get_uuid(),
            "parent_id": kb_folder_id,
            "tenant_id": tenant_id,
            "created_by": tenant_id,
            "name": doc["name"],
            "type": doc["type"],
            "size": doc["size"],
            "location": doc["location"],
            "source_type": FileSource.KNOWLEDGEBASE,
        }
        cls.save(**file)
        File2DocumentService.save(
            **{"id": get_uuid(), "file_id": file["id"], "document_id": doc["id"]}
        )

    @classmethod
    @DB.connection_context()
    def move_file(cls, file_ids, folder_id):
        try:
            cls.filter_update((cls.model.id << file_ids,), {"parent_id": folder_id})
        except Exception as e:
            print(e)
            raise RuntimeError("Database error (File move)!")

    @classmethod
    @DB.connection_context()
    def upload_document(self, kb, file_objs, user_id):
        """上传文档到知识库

        Args:
            kb: 知识库对象
            file_objs: 文件对象列表
            user_id: 用户ID

        Returns:
            None

        Raises:
            RuntimeError: 当超过最大文件数限制或文件类型不支持时抛出
        """
        # 获取并初始化文件夹结构
        root_folder = self.get_root_folder(user_id)
        pf_id = root_folder["id"]
        self.init_knowledgebase_docs(pf_id, user_id)
        kb_root_folder = self.get_kb_folder(user_id)
        kb_folder = self.new_a_file_from_kb(kb.tenant_id, kb.name, kb_root_folder["id"])

        err, files = [], []  # 初始化错误和成功列表
        for file in file_objs:
            try:
                # 检查文件数量限制
                MAX_FILE_NUM_PER_USER = int(os.environ.get("MAX_FILE_NUM_PER_USER", 0))
                if (
                    MAX_FILE_NUM_PER_USER > 0
                    and DocumentService.get_doc_count(kb.tenant_id)
                    >= MAX_FILE_NUM_PER_USER
                ):
                    raise RuntimeError("Exceed the maximum file number of a free user!")
                # 处理文件名和类型
                filename = duplicate_name(
                    DocumentService.query, name=file.filename, kb_id=kb.id
                )
                filetype = filename_type(filename)
                if filetype == FileType.OTHER.value:
                    raise RuntimeError("This type of file has not been supported yet!")
                # 确保文件位置唯一
                location = filename
                while STORAGE_IMPL.obj_exist(kb.id, location):
                    location += "_"

                # 读取并存储文件内容
                blob = file.read()
                STORAGE_IMPL.put(kb.id, location, blob)

                doc_id = get_uuid()

                # 生成并存储缩略图
                img = thumbnail_img(filename, blob)
                thumbnail_location = ""
                if img is not None:
                    thumbnail_location = f"thumbnail_{doc_id}.png"
                    STORAGE_IMPL.put(kb.id, thumbnail_location, img)

                doc = {
                    "id": doc_id,
                    "kb_id": kb.id,
                    "parser_id": self.get_parser(
                        filetype, filename, kb.parser_id
                    ),  # 根据文件类型选择解析器
                    "parser_config": kb.parser_config,
                    "created_by": user_id,
                    "type": filetype,
                    "name": filename,
                    "location": location,
                    "size": len(blob),
                    "thumbnail": thumbnail_location,
                }
                DocumentService.insert(doc)  # 保存文档记录

                FileService.add_file_from_kb(doc, kb_folder["id"], kb.tenant_id)
                files.append((doc, blob))
            except Exception as e:
                err.append(file.filename + ": " + str(e))

        return err, files

    @staticmethod
    def get_parser(doc_type, filename, default):
        if doc_type == FileType.VISUAL:
            return ParserType.PICTURE.value
        if doc_type == FileType.AURAL:
            return ParserType.AUDIO.value
        if re.search(r"\.(ppt|pptx|pages)$", filename):
            return ParserType.PRESENTATION.value
        if re.search(r"\.(eml)$", filename):
            return ParserType.EMAIL.value
        return default
