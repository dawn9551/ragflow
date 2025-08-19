# Deep Research聊天模式集成实现指南

> **创建时间**: 2025-01-08  
> **目标**: 将Deep Research作为聊天时的独立选项集成到RAGFlow中  
> **实现范围**: 前端UI + 后端API + 数据流设计  

---

## 目录

1. [需求分析](#需求分析)
2. [整体架构设计](#整体架构设计)
3. [后端实现方案](#后端实现方案)
4. [前端实现方案](#前端实现方案)
5. [数据库设计调整](#数据库设计调整)
6. [配置和部署](#配置和部署)
7. [测试验证](#测试验证)

---

## 需求分析

### 功能需求
- **用户界面**: 在聊天界面添加"Deep Research"模式选择器
- **输入处理**: 接收用户问题，触发Deep Research流程
- **多源检索**: 自动检索Web + 本地知识库 + 知识图谱
- **流式输出**: 实时显示推理过程和搜索进度
- **结果展示**: 生成结构化的研究报告
- **历史记录**: 保存Deep Research会话历史

### 技术需求
- **兼容性**: 与现有聊天系统无缝集成
- **性能**: 支持并发Deep Research请求
- **可配置**: 支持参数调优和功能开关
- **扩展性**: 便于添加新的数据源和功能

---

## 整体架构设计

### 系统架构图

```
┌─────────────────┬─────────────────┬─────────────────┐
│   前端 (React)   │   后端 (Flask)   │   数据层         │
├─────────────────┼─────────────────┼─────────────────┤
│ 聊天界面增强     │ Deep Research    │ 会话存储         │
│ ├─ 模式选择器   │ API端点          │ ├─ 对话记录     │
│ ├─ 流式显示     │ ├─ /chat/deep    │ ├─ 推理步骤     │
│ ├─ 进度指示     │ ├─ 流式处理      │ └─ 检索结果     │
│ └─ 报告展示     │ └─ 错误处理      │                 │
├─────────────────┼─────────────────┼─────────────────┤
│ 配置界面        │ DeepResearcher   │ 知识库          │
│ ├─ 参数设置     │ 集成层           │ ├─ 向量数据库   │
│ ├─ 数据源配置   │ ├─ 知识库检索    │ ├─ 知识图谱     │
│ └─ API密钥管理  │ ├─ Web搜索       │ └─ 文档存储     │
│                 │ └─ 知识图谱查询  │                 │
└─────────────────┴─────────────────┴─────────────────┘
```

### 数据流设计

```
用户输入问题
    ↓
选择Deep Research模式
    ↓
前端发送请求到 /api/v1/chat/deep_research
    ↓
后端创建DeepResearcher实例
    ↓
开始多轮推理循环
    ├─ 生成推理步骤
    ├─ 提取搜索查询  
    ├─ 多源信息检索
    │   ├─ 知识库检索
    │   ├─ Web搜索 (Tavily)
    │   └─ 知识图谱查询
    ├─ 信息提取总结
    └─ 流式返回结果
    ↓
前端实时显示推理过程
    ↓
生成最终研究报告
    ↓
保存会话历史
```

---

## 后端实现方案

### 1. 创建Deep Research API端点

#### 新建文件: `api/apps/deep_research_app.py`

```python
#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
import json
import logging
from flask import Blueprint, request, Response
from flask_login import login_required, current_user
from functools import partial

from agentic_reasoning.deep_research import DeepResearcher
from api.db.services.dialog_service import DialogService, ConversationService
from api.db.services.llm_service import LLMBundle, TenantLLMService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.utils.api_utils import server_error_response, validate_request
from api.utils.file_utils import get_project_base_directory
from rag.app import naive
from rag.nlp import search
from rag.utils import rmSpace
from rag.utils.redis_conn import REDIS_CONN
from graphrag.search import kg_search


bp = Blueprint('deep_research_app', __name__, url_prefix='/api/v1/chat')


class DeepResearchChatService:
    """Deep Research聊天服务"""
    
    def __init__(self):
        self.dialog_service = DialogService()
        self.conversation_service = ConversationService()
        self.kb_service = KnowledgebaseService()
    
    def create_deep_researcher(self, dialog_id: str, question: str):
        """创建DeepResearcher实例"""
        # 获取对话配置
        dialog = self.dialog_service.get_by_id(dialog_id)
        if not dialog:
            raise ValueError("Dialog not found")
        
        # 获取LLM配置
        llm_id = dialog.llm_id or "qwen-max@Tongyi-Qianwen"  # 默认LLM
        chat_mdl = LLMBundle(dialog.tenant_id, 
                           TenantLLMService.llm_id2llm_type(llm_id),
                           llm_id)
        
        # 获取prompt配置
        prompt_config = dialog.prompt_config or {}
        
        # 知识库检索函数
        kb_retrieve = None
        if dialog.kb_ids:
            kb_retrieve = partial(
                self._kb_retrieve,
                dialog_id=dialog_id,
                kb_ids=dialog.kb_ids,
                tenant_id=dialog.tenant_id
            )
        
        # 知识图谱检索函数  
        kg_retrieve = None
        if prompt_config.get("use_kg", False):
            kg_retrieve = partial(
                self._kg_retrieve,
                tenant_id=dialog.tenant_id,
                kb_ids=dialog.kb_ids
            )
        
        return DeepResearcher(
            chat_mdl=chat_mdl,
            prompt_config=prompt_config,
            kb_retrieve=kb_retrieve,
            kg_retrieve=kg_retrieve
        )
    
    def _kb_retrieve(self, question: str, dialog_id: str, kb_ids: list, tenant_id: str):
        """知识库检索"""
        try:
            kbinfos = {"chunks": [], "doc_aggs": []}
            
            for kb_id in kb_ids:
                kb = self.kb_service.get_by_id(kb_id)
                if not kb:
                    continue
                
                # 执行检索
                retr = naive.retrieval(
                    question, 
                    kb.embd_id, 
                    kb.tenant_id, 
                    kb_id, 
                    1, 
                    kb.parser_config.get("top_n", 8),
                    kb.parser_config.get("similarity_threshold", 0.2),
                    kb.parser_config.get("vector_similarity_weight", 0.3),
                    doc_ids=kb.doc_ids if hasattr(kb, 'doc_ids') else []
                )
                
                if retr.get("chunks"):
                    kbinfos["chunks"].extend(retr["chunks"])
                if retr.get("doc_aggs"):
                    kbinfos["doc_aggs"].extend(retr["doc_aggs"])
            
            return kbinfos
            
        except Exception as e:
            logging.error(f"Knowledge base retrieval error: {e}")
            return {"chunks": [], "doc_aggs": []}
    
    def _kg_retrieve(self, question: str, tenant_id: str, kb_ids: list):
        """知识图谱检索"""
        try:
            if not kb_ids:
                return {"content_with_weight": ""}
            
            # 使用第一个知识库进行图谱检索
            kb_id = kb_ids[0]
            kg_res = kg_search(question, kb_id, tenant_id)
            
            return {"content_with_weight": kg_res}
            
        except Exception as e:
            logging.error(f"Knowledge graph retrieval error: {e}")
            return {"content_with_weight": ""}
    
    def save_conversation(self, dialog_id: str, question: str, answer: str, 
                         research_steps: list = None):
        """保存对话记录"""
        try:
            conversation = {
                "role": "user",
                "content": question,
                "id": f"user_{int(time.time() * 1000)}"
            }
            self.conversation_service.save(dialog_id, conversation)
            
            assistant_conversation = {
                "role": "assistant", 
                "content": answer,
                "id": f"assistant_{int(time.time() * 1000)}",
                "research_mode": "deep_research",
                "research_steps": research_steps or []
            }
            self.conversation_service.save(dialog_id, assistant_conversation)
            
        except Exception as e:
            logging.error(f"Save conversation error: {e}")


@bp.route('/deep_research', methods=['POST'])
@login_required
@validate_request("dialog_id", "question")
def deep_research_chat():
    """Deep Research聊天API"""
    try:
        req = request.json
        dialog_id = req["dialog_id"]
        question = req["question"]
        
        # 验证权限
        dialog = DialogService.get_by_id(dialog_id)
        if not dialog or dialog.tenant_id != current_user.id:
            return server_error_response("Dialog not found or access denied")
        
        # 创建服务实例
        service = DeepResearchChatService()
        
        def generate_response():
            """生成流式响应"""
            chunk_info = {"chunks": [], "doc_aggs": []}  # 用于收集引用信息
            research_steps = []  # 记录研究步骤
            final_answer = ""
            
            try:
                # 创建DeepResearcher
                researcher = service.create_deep_researcher(dialog_id, question)
                
                # 发送开始事件
                yield f"data: {json.dumps({'event': 'start', 'data': {'question': question}}, ensure_ascii=False)}\n\n"
                
                # 开始深度研究
                step_count = 0
                for result in researcher.thinking(chunk_info, question):
                    if isinstance(result, dict) and "answer" in result:
                        answer_content = result["answer"]
                        
                        # 提取推理步骤
                        if answer_content.startswith("<think>") and answer_content.endswith("</think>"):
                            thinking_content = answer_content[7:-8]  # 去除<think>标记
                            
                            # 发送推理步骤
                            step_data = {
                                "event": "thinking_step",
                                "data": {
                                    "step": step_count,
                                    "content": thinking_content,
                                    "timestamp": int(time.time() * 1000)
                                }
                            }
                            yield f"data: {json.dumps(step_data, ensure_ascii=False)}\n\n"
                            
                            research_steps.append({
                                "step": step_count,
                                "content": thinking_content,
                                "timestamp": int(time.time() * 1000)
                            })
                            step_count += 1
                    
                    elif isinstance(result, str):
                        # 最终答案
                        final_answer = result
                        if final_answer.startswith("<think>") and final_answer.endswith("</think>"):
                            final_answer = final_answer[7:-8]
                
                # 发送最终结果
                final_data = {
                    "event": "final_answer",
                    "data": {
                        "answer": final_answer,
                        "references": {
                            "chunks": chunk_info.get("chunks", []),
                            "doc_aggs": chunk_info.get("doc_aggs", [])
                        },
                        "research_steps": research_steps
                    }
                }
                yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"
                
                # 保存对话
                service.save_conversation(dialog_id, question, final_answer, research_steps)
                
                # 发送完成事件
                yield f"data: {json.dumps({'event': 'complete'}, ensure_ascii=False)}\n\n"
                
            except Exception as e:
                logging.error(f"Deep research error: {e}")
                error_data = {
                    "event": "error",
                    "data": {"message": str(e)}
                }
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
        
        return Response(
            generate_response(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'
            }
        )
        
    except Exception as e:
        return server_error_response(e)


@bp.route('/deep_research/config', methods=['GET', 'POST'])
@login_required  
def deep_research_config():
    """Deep Research配置管理"""
    try:
        if request.method == 'GET':
            # 获取当前配置
            dialog_id = request.args.get('dialog_id')
            if not dialog_id:
                return server_error_response("Dialog ID required")
            
            dialog = DialogService.get_by_id(dialog_id)
            if not dialog or dialog.tenant_id != current_user.id:
                return server_error_response("Dialog not found or access denied")
            
            config = dialog.prompt_config or {}
            deep_research_config = {
                "enabled": config.get("reasoning", False),
                "tavily_api_key": config.get("tavily_api_key", ""),
                "use_kg": config.get("use_kg", False),
                "max_search_rounds": config.get("max_search_rounds", 6),
                "temperature": config.get("temperature", 0.7)
            }
            
            return {"retcode": 0, "data": deep_research_config}
        
        elif request.method == 'POST':
            # 更新配置
            req = request.json
            dialog_id = req.get("dialog_id")
            config = req.get("config", {})
            
            if not dialog_id:
                return server_error_response("Dialog ID required")
            
            dialog = DialogService.get_by_id(dialog_id)
            if not dialog or dialog.tenant_id != current_user.id:
                return server_error_response("Dialog not found or access denied")
            
            # 更新prompt_config
            prompt_config = dialog.prompt_config or {}
            prompt_config.update({
                "reasoning": config.get("enabled", False),
                "tavily_api_key": config.get("tavily_api_key", ""),
                "use_kg": config.get("use_kg", False), 
                "max_search_rounds": config.get("max_search_rounds", 6),
                "temperature": config.get("temperature", 0.7)
            })
            
            DialogService.update_by_id(dialog_id, {"prompt_config": prompt_config})
            
            return {"retcode": 0, "data": {"message": "Configuration updated successfully"}}
            
    except Exception as e:
        return server_error_response(e)
```

### 2. 注册API路由

#### 修改文件: `api/ragflow_server.py`

```python
# 在现有的Blueprint注册部分添加
from api.apps.deep_research_app import bp as deep_research_bp

# 注册Deep Research Blueprint
app.register_blueprint(deep_research_bp)
```

### 3. 扩展Dialog模型

#### 修改文件: `api/db/db_models.py`

```python
# 在Dialog类中添加新字段（如果需要）
class Dialog(DataBaseModel):
    # ... 现有字段
    
    # 添加Deep Research相关配置
    deep_research_enabled = BooleanField(default=False)  # 是否启用Deep Research
    deep_research_config = TextField(default='{}')       # Deep Research配置JSON
```

---

## 前端实现方案

### 1. 聊天模式选择器组件

#### 新建文件: `web/src/components/chat-mode-selector/index.tsx`

```tsx
import React from 'react';
import { Select, Switch, Tooltip } from 'antd';
import { BrainIcon, MessageCircleIcon, SearchIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

export enum ChatMode {
  NORMAL = 'normal',
  DEEP_RESEARCH = 'deep_research'
}

interface ChatModeSelectorProps {
  mode: ChatMode;
  onChange: (mode: ChatMode) => void;
  disabled?: boolean;
}

const ChatModeSelector: React.FC<ChatModeSelectorProps> = ({
  mode,
  onChange,
  disabled = false
}) => {
  const { t } = useTranslation();

  const chatModeOptions = [
    {
      value: ChatMode.NORMAL,
      label: t('chat.mode.normal'),
      icon: <MessageCircleIcon size={16} />,
      description: t('chat.mode.normal.description')
    },
    {
      value: ChatMode.DEEP_RESEARCH,
      label: t('chat.mode.deepResearch'),
      icon: <BrainIcon size={16} />,
      description: t('chat.mode.deepResearch.description')
    }
  ];

  return (
    <div className="flex items-center space-x-4 p-2 bg-gray-50 rounded-lg">
      <span className="text-sm font-medium text-gray-700">
        {t('chat.mode.title')}:
      </span>
      
      <Select
        value={mode}
        onChange={onChange}
        disabled={disabled}
        className="min-w-[200px]"
        optionLabelProp="label"
      >
        {chatModeOptions.map((option) => (
          <Select.Option key={option.value} value={option.value} label={
            <div className="flex items-center space-x-2">
              {option.icon}
              <span>{option.label}</span>
            </div>
          }>
            <div className="flex items-start space-x-3 py-2">
              <div className="flex-shrink-0 mt-1">
                {option.icon}
              </div>
              <div>
                <div className="font-medium">{option.label}</div>
                <div className="text-sm text-gray-500">{option.description}</div>
              </div>
            </div>
          </Select.Option>
        ))}
      </Select>

      {mode === ChatMode.DEEP_RESEARCH && (
        <Tooltip title={t('chat.mode.deepResearch.tooltip')}>
          <div className="flex items-center space-x-1 text-blue-600">
            <SearchIcon size={14} />
            <span className="text-xs">{t('chat.mode.deepResearch.active')}</span>
          </div>
        </Tooltip>
      )}
    </div>
  );
};

export default ChatModeSelector;
```

### 2. Deep Research消息组件

#### 新建文件: `web/src/components/deep-research-message/index.tsx`

```tsx
import React, { useState } from 'react';
import { Card, Collapse, Timeline, Tag, Button, Divider } from 'antd';
import { BrainIcon, SearchIcon, FileTextIcon, ChevronDownIcon, ChevronUpIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import MarkdownContent from '@/components/next-markdown-content';

interface ResearchStep {
  step: number;
  content: string;
  timestamp: number;
}

interface Reference {
  chunks: any[];
  doc_aggs: any[];
}

interface DeepResearchMessageProps {
  answer: string;
  researchSteps: ResearchStep[];
  references: Reference;
  loading?: boolean;
  currentStep?: number;
}

const DeepResearchMessage: React.FC<DeepResearchMessageProps> = ({
  answer,
  researchSteps,
  references,
  loading = false,
  currentStep = -1
}) => {
  const { t } = useTranslation();
  const [showSteps, setShowSteps] = useState(false);
  const [showReferences, setShowReferences] = useState(false);

  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const renderThinkingProcess = () => {
    if (!researchSteps || researchSteps.length === 0) return null;

    return (
      <Card
        size="small"
        className="mb-4"
        title={
          <div className="flex items-center space-x-2">
            <BrainIcon size={16} className="text-blue-500" />
            <span>{t('deepResearch.thinkingProcess')}</span>
            <Tag color="blue">{researchSteps.length} {t('deepResearch.steps')}</Tag>
          </div>
        }
        extra={
          <Button
            type="text"
            size="small"
            icon={showSteps ? <ChevronUpIcon size={14} /> : <ChevronDownIcon size={14} />}
            onClick={() => setShowSteps(!showSteps)}
          >
            {showSteps ? t('common.collapse') : t('common.expand')}
          </Button>
        }
      >
        <Collapse ghost activeKey={showSteps ? ['steps'] : []}>
          <Collapse.Panel key="steps" header="" showArrow={false}>
            <Timeline>
              {researchSteps.map((step, index) => (
                <Timeline.Item
                  key={step.step}
                  color={
                    loading && index === currentStep
                      ? 'blue'
                      : index <= currentStep
                      ? 'green'
                      : 'gray'
                  }
                  dot={
                    loading && index === currentStep ? (
                      <SearchIcon size={12} className="animate-spin" />
                    ) : (
                      <span className="w-2 h-2 rounded-full bg-current" />
                    )
                  }
                >
                  <div className="pb-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <span className="font-medium text-gray-900">
                        {t('deepResearch.step')} {step.step + 1}
                      </span>
                      <span className="text-xs text-gray-500">
                        {formatTimestamp(step.timestamp)}
                      </span>
                    </div>
                    <div className="text-sm text-gray-700 whitespace-pre-wrap">
                      {step.content}
                    </div>
                  </div>
                </Timeline.Item>
              ))}
            </Timeline>
          </Collapse.Panel>
        </Collapse>
      </Card>
    );
  };

  const renderReferences = () => {
    if (!references || (!references.chunks?.length && !references.doc_aggs?.length)) {
      return null;
    }

    return (
      <Card
        size="small"
        className="mt-4"
        title={
          <div className="flex items-center space-x-2">
            <FileTextIcon size={16} className="text-green-500" />
            <span>{t('deepResearch.references')}</span>
            <Tag color="green">
              {(references.chunks?.length || 0) + (references.doc_aggs?.length || 0)} {t('deepResearch.sources')}
            </Tag>
          </div>
        }
        extra={
          <Button
            type="text"
            size="small"
            icon={showReferences ? <ChevronUpIcon size={14} /> : <ChevronDownIcon size={14} />}
            onClick={() => setShowReferences(!showReferences)}
          >
            {showReferences ? t('common.collapse') : t('common.expand')}
          </Button>
        }
      >
        <Collapse ghost activeKey={showReferences ? ['references'] : []}>
          <Collapse.Panel key="references" header="" showArrow={false}>
            <div className="space-y-3">
              {references.chunks?.map((chunk, index) => (
                <div key={index} className="p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center space-x-2 mb-2">
                    <Tag size="small" color="blue">KB</Tag>
                    <span className="font-medium text-sm">{chunk.doc_name}</span>
                  </div>
                  <div className="text-sm text-gray-700 line-clamp-3">
                    {chunk.content_with_weight}
                  </div>
                </div>
              ))}
              
              {references.doc_aggs?.map((doc, index) => (
                <div key={index} className="p-3 bg-blue-50 rounded-lg">
                  <div className="flex items-center space-x-2 mb-2">
                    <Tag size="small" color="green">WEB</Tag>
                    <span className="font-medium text-sm">{doc.doc_name}</span>
                  </div>
                  <div className="text-sm text-gray-700">
                    {doc.doc_name}
                  </div>
                </div>
              ))}
            </div>
          </Collapse.Panel>
        </Collapse>
      </Card>
    );
  };

  return (
    <div className="deep-research-message">
      {renderThinkingProcess()}
      
      <div className="research-answer">
        <Card 
          title={
            <div className="flex items-center space-x-2">
              <BrainIcon size={16} className="text-purple-500" />
              <span>{t('deepResearch.researchReport')}</span>
            </div>
          }
          className="mb-4"
        >
          <MarkdownContent content={answer} />
        </Card>
      </div>

      {renderReferences()}
    </div>
  );
};

export default DeepResearchMessage;
```

### 3. 聊天界面集成

#### 修改文件: `web/src/pages/chat/index.tsx`

```tsx
import React, { useState, useEffect } from 'react';
import { useParams } from 'umi';
import ChatModeSelector, { ChatMode } from '@/components/chat-mode-selector';
import DeepResearchMessage from '@/components/deep-research-message';
import { useSendMessage } from '@/hooks/use-send-message';

const ChatPage: React.FC = () => {
  const { id: dialogId } = useParams();
  const [chatMode, setChatMode] = useState<ChatMode>(ChatMode.NORMAL);
  const [isDeepResearching, setIsDeepResearching] = useState(false);
  const [currentResearchStep, setCurrentResearchStep] = useState(-1);
  const [researchSteps, setResearchSteps] = useState([]);
  
  const { sendMessage, sendingLoading } = useSendMessage();

  // Deep Research专用发送消息函数
  const sendDeepResearchMessage = async (message: string) => {
    if (!dialogId) return;

    setIsDeepResearching(true);
    setCurrentResearchStep(0);
    setResearchSteps([]);

    try {
      const response = await fetch('/api/v1/chat/deep_research', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          dialog_id: dialogId,
          question: message
        })
      });

      if (!response.ok) {
        throw new Error('Deep research request failed');
      }

      const reader = response.body?.getReader();
      if (!reader) return;

      let finalAnswer = '';
      let references = { chunks: [], doc_aggs: [] };
      const steps = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = new TextDecoder().decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              switch (data.event) {
                case 'start':
                  console.log('Deep research started:', data.data);
                  break;
                  
                case 'thinking_step':
                  steps.push(data.data);
                  setResearchSteps([...steps]);
                  setCurrentResearchStep(data.data.step);
                  break;
                  
                case 'final_answer':
                  finalAnswer = data.data.answer;
                  references = data.data.references;
                  setResearchSteps(data.data.research_steps);
                  break;
                  
                case 'error':
                  console.error('Deep research error:', data.data.message);
                  break;
                  
                case 'complete':
                  setIsDeepResearching(false);
                  setCurrentResearchStep(-1);
                  break;
              }
            } catch (e) {
              console.error('Failed to parse SSE data:', e);
            }
          }
        }
      }

      // 添加到消息列表
      if (finalAnswer) {
        // 这里需要根据你的状态管理方式来更新消息列表
        // addMessage({
        //   role: 'assistant',
        //   content: finalAnswer,
        //   research_mode: 'deep_research',
        //   research_steps: steps,
        //   references: references
        // });
      }

    } catch (error) {
      console.error('Deep research error:', error);
      setIsDeepResearching(false);
    }
  };

  // 处理消息发送
  const handleSendMessage = async (message: string) => {
    if (chatMode === ChatMode.DEEP_RESEARCH) {
      await sendDeepResearchMessage(message);
    } else {
      await sendMessage(message);
    }
  };

  // 渲染消息组件
  const renderMessage = (message: any) => {
    if (message.research_mode === 'deep_research') {
      return (
        <DeepResearchMessage
          answer={message.content}
          researchSteps={message.research_steps || []}
          references={message.references || { chunks: [], doc_aggs: [] }}
          loading={isDeepResearching}
          currentStep={currentResearchStep}
        />
      );
    }
    
    // 普通消息渲染逻辑
    return <div>{message.content}</div>;
  };

  return (
    <div className="chat-container">
      {/* 聊天模式选择器 */}
      <div className="chat-header p-4 border-b">
        <ChatModeSelector
          mode={chatMode}
          onChange={setChatMode}
          disabled={sendingLoading || isDeepResearching}
        />
      </div>

      {/* 消息列表 */}
      <div className="messages-container flex-1 overflow-y-auto p-4">
        {/* 渲染消息列表 */}
        {/* messages.map(renderMessage) */}
      </div>

      {/* 输入区域 */}
      <div className="chat-input-container p-4 border-t">
        {/* 
          根据chatMode显示不同的输入提示
          Deep Research模式下可以显示特殊的占位符文本
        */}
        <MessageInput
          onSend={handleSendMessage}
          disabled={sendingLoading || isDeepResearching}
          placeholder={
            chatMode === ChatMode.DEEP_RESEARCH
              ? t('chat.input.deepResearch.placeholder')
              : t('chat.input.normal.placeholder')
          }
        />
      </div>
    </div>
  );
};

export default ChatPage;
```

### 4. 多语言支持

#### 修改文件: `web/src/locales/zh.ts`

```typescript
export default {
  // ... 现有翻译
  
  chat: {
    mode: {
      title: '聊天模式',
      normal: '普通聊天',
      deepResearch: '深度研究',
      'normal.description': '快速对话，基于知识库直接回答',
      'deepResearch.description': '深度研究模式，多源检索，详细分析',
      'deepResearch.tooltip': '当前处于深度研究模式',
      'deepResearch.active': '研究中'
    },
    input: {
      'normal.placeholder': '请输入您的问题...',
      'deepResearch.placeholder': '请输入需要深度研究的问题，我将为您进行全面分析...'
    }
  },
  
  deepResearch: {
    thinkingProcess: '思考过程',
    steps: '步骤',
    step: '步骤',
    researchReport: '研究报告',
    references: '参考资料',
    sources: '来源',
    analyzing: '正在分析...',
    searching: '正在搜索...',
    synthesizing: '正在综合信息...'
  }
};
```

#### 修改文件: `web/src/locales/en.ts`

```typescript
export default {
  // ... existing translations
  
  chat: {
    mode: {
      title: 'Chat Mode',
      normal: 'Normal Chat',
      deepResearch: 'Deep Research',
      'normal.description': 'Quick conversation based on knowledge base',
      'deepResearch.description': 'Deep research mode with multi-source retrieval and detailed analysis',
      'deepResearch.tooltip': 'Currently in deep research mode',
      'deepResearch.active': 'Researching'
    },
    input: {
      'normal.placeholder': 'Enter your question...',
      'deepResearch.placeholder': 'Enter a question for deep research, I will provide comprehensive analysis...'
    }
  },
  
  deepResearch: {
    thinkingProcess: 'Thinking Process',
    steps: 'Steps',
    step: 'Step',
    researchReport: 'Research Report',
    references: 'References',
    sources: 'Sources',
    analyzing: 'Analyzing...',
    searching: 'Searching...',
    synthesizing: 'Synthesizing information...'
  }
};
```

---

## 数据库设计调整

### 1. 扩展Dialog表

```sql
-- 添加Deep Research相关字段
ALTER TABLE dialog ADD COLUMN deep_research_enabled BOOLEAN DEFAULT FALSE;
ALTER TABLE dialog ADD COLUMN deep_research_config TEXT DEFAULT '{}';

-- 创建索引
CREATE INDEX idx_dialog_deep_research ON dialog(deep_research_enabled);
```

### 2. 扩展Conversation表

```sql  
-- 添加研究模式标识
ALTER TABLE conversation ADD COLUMN research_mode VARCHAR(50) DEFAULT NULL;
ALTER TABLE conversation ADD COLUMN research_steps TEXT DEFAULT NULL;
ALTER TABLE conversation ADD COLUMN research_references TEXT DEFAULT NULL;

-- 创建索引
CREATE INDEX idx_conversation_research_mode ON conversation(research_mode);
```

### 3. 创建Deep Research配置表 (可选)

```sql
-- 创建Deep Research全局配置表
CREATE TABLE deep_research_config (
    id VARCHAR(128) PRIMARY KEY,
    tenant_id VARCHAR(128) NOT NULL,
    config_name VARCHAR(128) NOT NULL,
    config_value TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (tenant_id) REFERENCES tenant(id) ON DELETE CASCADE
);

CREATE INDEX idx_deep_research_config_tenant ON deep_research_config(tenant_id);
```

---

## 配置和部署

### 1. 环境变量配置

#### 修改文件: `docker/.env`

```bash
# Deep Research相关配置
DEEP_RESEARCH_ENABLED=true
DEEP_RESEARCH_MAX_ROUNDS=6
DEEP_RESEARCH_TEMPERATURE=0.7

# Tavily API配置 (可选)
TAVILY_API_KEY=your_tavily_api_key_here
```

### 2. 服务配置

#### 修改文件: `docker/service_conf.yaml.template`

```yaml
# Deep Research配置
deep_research:
  enabled: ${DEEP_RESEARCH_ENABLED:-true}
  max_rounds: ${DEEP_RESEARCH_MAX_ROUNDS:-6}
  temperature: ${DEEP_RESEARCH_TEMPERATURE:-0.7}
  
  # 默认数据源配置
  data_sources:
    knowledge_base:
      enabled: true
      weight: 0.4
    web_search:
      enabled: true
      weight: 0.4
      api_key: ${TAVILY_API_KEY:-""}
    knowledge_graph:
      enabled: true
      weight: 0.2
```

### 3. Docker配置更新

#### 修改文件: `docker/docker-compose.yml`

```yaml
version: '3.8'
services:
  ragflow:
    # ... 现有配置
    environment:
      # ... 现有环境变量
      - DEEP_RESEARCH_ENABLED=${DEEP_RESEARCH_ENABLED:-true}
      - DEEP_RESEARCH_MAX_ROUNDS=${DEEP_RESEARCH_MAX_ROUNDS:-6}
      - TAVILY_API_KEY=${TAVILY_API_KEY:-""}
```

---

## 测试验证

### 1. 单元测试

#### 新建文件: `test/testcases/test_deep_research/test_deep_research_api.py`

```python
import pytest
import json
from test.libs.auth import login_test_user
from test.testcases.test_http_api.common import request_json


class TestDeepResearchAPI:
    
    def test_deep_research_chat(self):
        """测试Deep Research聊天功能"""
        # 登录
        login_test_user()
        
        # 创建测试对话
        dialog_data = {
            "name": "Deep Research Test Dialog",
            "kb_ids": []
        }
        dialog_response = request_json('/api/v1/dialog', 'POST', dialog_data)
        assert dialog_response['retcode'] == 0
        dialog_id = dialog_response['data']['id']
        
        # 发送Deep Research请求
        research_data = {
            "dialog_id": dialog_id,
            "question": "What are the latest developments in artificial intelligence?"
        }
        
        response = request_json('/api/v1/chat/deep_research', 'POST', research_data)
        # 由于是流式响应，这里需要特殊处理
        assert response.status_code == 200
        assert response.headers.get('content-type') == 'text/event-stream; charset=utf-8'
    
    def test_deep_research_config(self):
        """测试Deep Research配置管理"""
        # 登录
        login_test_user()
        
        # 创建测试对话
        dialog_data = {
            "name": "Config Test Dialog",
            "kb_ids": []
        }
        dialog_response = request_json('/api/v1/dialog', 'POST', dialog_data)
        dialog_id = dialog_response['data']['id']
        
        # 获取配置
        config_response = request_json(f'/api/v1/chat/deep_research/config?dialog_id={dialog_id}', 'GET')
        assert config_response['retcode'] == 0
        
        # 更新配置
        new_config = {
            "dialog_id": dialog_id,
            "config": {
                "enabled": True,
                "tavily_api_key": "test_key",
                "use_kg": True,
                "max_search_rounds": 8
            }
        }
        
        update_response = request_json('/api/v1/chat/deep_research/config', 'POST', new_config)
        assert update_response['retcode'] == 0
```

### 2. 前端组件测试

#### 新建文件: `web/src/components/chat-mode-selector/__tests__/index.test.tsx`

```tsx
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import ChatModeSelector, { ChatMode } from '../index';

describe('ChatModeSelector', () => {
  const mockOnChange = jest.fn();

  beforeEach(() => {
    mockOnChange.mockClear();
  });

  test('renders correctly with normal mode', () => {
    render(
      <ChatModeSelector
        mode={ChatMode.NORMAL}
        onChange={mockOnChange}
      />
    );

    expect(screen.getByText('Normal Chat')).toBeInTheDocument();
  });

  test('renders correctly with deep research mode', () => {
    render(
      <ChatModeSelector
        mode={ChatMode.DEEP_RESEARCH}
        onChange={mockOnChange}
      />
    );

    expect(screen.getByText('Deep Research')).toBeInTheDocument();
    expect(screen.getByText('Researching')).toBeInTheDocument();
  });

  test('calls onChange when mode is changed', () => {
    render(
      <ChatModeSelector
        mode={ChatMode.NORMAL}
        onChange={mockOnChange}
      />
    );

    // 模拟选择Deep Research模式
    fireEvent.click(screen.getByText('Normal Chat'));
    fireEvent.click(screen.getByText('Deep Research'));

    expect(mockOnChange).toHaveBeenCalledWith(ChatMode.DEEP_RESEARCH);
  });

  test('is disabled when disabled prop is true', () => {
    render(
      <ChatModeSelector
        mode={ChatMode.NORMAL}
        onChange={mockOnChange}
        disabled={true}
      />
    );

    const selector = screen.getByRole('combobox');
    expect(selector).toBeDisabled();
  });
});
```

### 3. 集成测试脚本

#### 新建文件: `test/deep_research_integration_test.py`

```python
#!/usr/bin/env python3
"""Deep Research集成测试脚本"""

import requests
import json
import time
from urllib.parse import urljoin


class DeepResearchIntegrationTest:
    
    def __init__(self, base_url='http://localhost'):
        self.base_url = base_url
        self.session = requests.Session()
        self.dialog_id = None
    
    def login(self, email='test@ragflow.io', password='test123'):
        """登录测试用户"""
        login_data = {
            'email': email,
            'password': password
        }
        response = self.session.post(
            urljoin(self.base_url, '/api/v1/user/login'),
            json=login_data
        )
        assert response.status_code == 200
        result = response.json()
        assert result['retcode'] == 0
        print("✓ 登录成功")
        return result
    
    def create_dialog(self):
        """创建测试对话"""
        dialog_data = {
            'name': 'Deep Research Integration Test',
            'kb_ids': []
        }
        response = self.session.post(
            urljoin(self.base_url, '/api/v1/dialog'),
            json=dialog_data
        )
        assert response.status_code == 200
        result = response.json()
        assert result['retcode'] == 0
        self.dialog_id = result['data']['id']
        print(f"✓ 创建对话成功: {self.dialog_id}")
        return result
    
    def test_deep_research_config(self):
        """测试Deep Research配置"""
        # 获取当前配置
        response = self.session.get(
            urljoin(self.base_url, f'/api/v1/chat/deep_research/config?dialog_id={self.dialog_id}')
        )
        assert response.status_code == 200
        result = response.json()
        assert result['retcode'] == 0
        print("✓ 获取配置成功")
        
        # 更新配置
        config_data = {
            'dialog_id': self.dialog_id,
            'config': {
                'enabled': True,
                'tavily_api_key': 'test_key',
                'use_kg': False,
                'max_search_rounds': 6,
                'temperature': 0.7
            }
        }
        response = self.session.post(
            urljoin(self.base_url, '/api/v1/chat/deep_research/config'),
            json=config_data
        )
        assert response.status_code == 200
        result = response.json()
        assert result['retcode'] == 0
        print("✓ 更新配置成功")
    
    def test_deep_research_chat(self):
        """测试Deep Research聊天"""
        research_data = {
            'dialog_id': self.dialog_id,
            'question': 'What are the key advantages of RAG systems in AI applications?'
        }
        
        response = self.session.post(
            urljoin(self.base_url, '/api/v1/chat/deep_research'),
            json=research_data,
            stream=True
        )
        
        assert response.status_code == 200
        assert 'text/event-stream' in response.headers.get('content-type', '')
        
        events = []
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith('data: '):
                try:
                    event_data = json.loads(line[6:])
                    events.append(event_data)
                    print(f"📡 收到事件: {event_data.get('event', 'unknown')}")
                    
                    if event_data.get('event') == 'complete':
                        break
                except json.JSONDecodeError:
                    continue
        
        # 验证事件序列
        event_types = [event.get('event') for event in events]
        assert 'start' in event_types
        assert 'thinking_step' in event_types or 'final_answer' in event_types
        assert 'complete' in event_types
        
        print("✓ Deep Research聊天测试成功")
        return events
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始Deep Research集成测试")
        
        try:
            self.login()
            self.create_dialog()
            self.test_deep_research_config()
            self.test_deep_research_chat()
            
            print("🎉 所有测试通过!")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            raise


if __name__ == '__main__':
    test = DeepResearchIntegrationTest()
    test.run_all_tests()
```

---

## 实施步骤建议

### 第一阶段：后端基础实现 (预计2-3天)
1. ✅ 创建 `api/apps/deep_research_app.py`
2. ✅ 实现 `DeepResearchChatService` 类
3. ✅ 添加API路由注册
4. ✅ 数据库Schema调整
5. ✅ 基础配置管理

### 第二阶段：前端组件开发 (预计2-3天) 
1. ✅ 开发 `ChatModeSelector` 组件
2. ✅ 开发 `DeepResearchMessage` 组件
3. ✅ 集成到聊天界面
4. ✅ 多语言支持
5. ✅ 样式和交互优化

### 第三阶段：集成测试和优化 (预计1-2天)
1. ✅ 单元测试编写
2. ✅ 集成测试验证
3. ✅ 性能优化
4. ✅ 错误处理完善
5. ✅ 文档更新

### 第四阶段：部署和监控 (预计1天)
1. ✅ 生产环境配置
2. ✅ 监控和日志
3. ✅ 用户培训
4. ✅ 反馈收集

---

## 注意事项和风险

### 技术风险
1. **流式响应处理**: 确保前端正确处理SSE事件流
2. **并发控制**: Deep Research可能消耗较多资源，需要考虑并发限制
3. **超时管理**: 设置合理的超时时间避免长时间等待
4. **错误恢复**: 完善的错误处理和用户友好的错误提示

### 用户体验风险
1. **等待时间**: Deep Research可能需要较长时间，需要清晰的进度指示
2. **模式切换**: 确保用户理解不同聊天模式的区别
3. **结果展示**: 复杂的研究结果需要清晰的结构化展示

### 运维风险
1. **资源消耗**: Deep Research模式可能消耗更多API调用和计算资源
2. **API限制**: 第三方服务(如Tavily)的API限制需要考虑
3. **存储增长**: 研究步骤和结果可能增加数据库存储需求

---

## 扩展计划

### 短期扩展 (1-2周)
- 添加研究报告导出功能 (PDF/Word)
- 支持自定义研究模板
- 增加研究历史管理
- 添加研究质量评分

### 中期扩展 (1-2月)
- 支持多语言研究
- 集成更多数据源 (学术数据库、专业数据源)
- 添加协作研究功能
- 实现研究报告分享

### 长期扩展 (3-6月)
- AI驱动的研究规划
- 自动化研究工作流
- 研究结果验证系统
- 集成外部分析工具

---

这个实现指南提供了将Deep Research集成为聊天选项的完整技术方案。通过这种设计，用户可以在普通聊天和深度研究之间灵活切换，获得不同深度的AI assistance。整个实现保持了与现有系统的兼容性，同时提供了丰富的扩展可能性。