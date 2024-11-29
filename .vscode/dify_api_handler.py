import litellm
from litellm import CustomLLM
import time
import re
import base64
from uuid import uuid4
import json
import requests
import os
from enum import Enum
from fastapi.responses import StreamingResponse

from typing import Iterator, AsyncIterator
from litellm.types.utils import GenericStreamingChunk, ModelResponse


from pydantic import BaseModel
from typing import List, Optional
import itertools
import httpx


class ChatRequest(BaseModel):
    model: str
    messages: List[dict]
    stream: Optional[bool] = False
    user: Optional[str] = None
    


class DifyAPIError(Exception):
    """自定义异常类用于处理 Dify API 错误"""
    pass


class ResponseMode(Enum):
    STREAMING = "streaming"  # 流式模式（推荐）。基于 SSE（Server-Sent Events）实现类似打字机输出方式的流式返回。
    BLOCKING = "blocking"    # 阻塞模式，等待执行完毕后返回结果。（请求若流程较长可能会被中断）。由于 Cloudflare 限制，请求会在 100 秒超时无返回后中断。
    

def dify_api_error_handler(response):
    """处理API错误响应"""
    if response.status_code != 200:
        try:
            error_info = response.json()
            error_code = error_info.get('code', 'unknown_error')
            error_message = error_info.get('message', '未知错误')
        except ValueError:
            error_code = 'unknown_error'
            error_message = response.text or '未知错误'
        raise DifyAPIError(f"发送聊天消息失败: {error_code} - {error_message}")


def create_headers(api_key: str, include_content_type: bool = False) -> dict:
    """
    创建请求头
    
    参数:
        api_key (str): API密钥
        include_content_type (bool): 是否包含Content-Type头
    """
    headers = {
        'Authorization': f'Bearer {api_key}'
    }
    if include_content_type:
        headers['Content-Type'] = 'application/json'
    return headers


def upload_files(api_base_url: str, headers: dict, files: list[str], user: str) -> list[str]:
    """
    上传文件到 Dify API 并返回文件 ID 列表

    参数:
        api_base_url (str): API 基础 URL
        headers (dict): 包含认证信息的请求头
        files (list[str]): 要上传的文件路径列表
        user (str): 用户标识

    返回:
        list[str]: 上传成功的文件 ID 列表

    抛出:
        DifyAPIError: 当文件不存在、格式不支持或上传失败时
    """
    uploaded_file_ids = []
    
    if not files:
        return uploaded_file_ids
        
    upload_url = f'{api_base_url}/files/upload'
    for file_path in files:
        if not os.path.isfile(file_path):
            raise DifyAPIError(f"文件不存在: {file_path}")

        file_name, file_ext = os.path.splitext(os.path.basename(file_path))
        file_ext = file_ext.lower().lstrip('.')
        if file_ext not in ['png', 'jpg', 'jpeg', 'webp', 'gif']:
            raise DifyAPIError(f"不支持的文件类型: {file_ext} (文件: {file_path})")

        with open(file_path, 'rb') as f:
            files_payload = {
                'file': (os.path.basename(file_path), f, f'image/{file_ext}')
            }
            data = {
                'user': user
            }
            response = requests.post(upload_url, headers=headers, files=files_payload, data=data)

        if response.status_code != 200 and response.status_code != 201:
            try:
                error_info = response.json()
                error_message = error_info.get('message', '未知错误')
            except ValueError:
                error_message = response.text or '未知错误'
            raise DifyAPIError(f"文件上传失败 ({file_path}): {error_message}")

        upload_response = response.json()
        uploaded_file_ids.append(upload_response['id'])
    
    return uploaded_file_ids


def prepare_chat_request(api_key: str, query: str, response_mode: ResponseMode, user: str, 
                        files: list[str] = None, conversation_id: str = None,
                        inputs: dict = None, auto_generate_name: bool = True, 
                        api_base_url: str = 'http://10.144.129.132/v1') -> tuple:
    """
    准备聊天请求所需的headers和payload
    """
    # 上传文件时使用的headers（不包含Content-Type）
    upload_headers = create_headers(api_key)
    
    # 处理文件上传
    uploaded_file_ids = upload_files(api_base_url, upload_headers, files or [], user) if files else []

    # 聊天请求使用的headers（包含Content-Type）
    chat_headers = create_headers(api_key, include_content_type=True)

    # 构建请求payload
    payload = {
        'query': query,
        'response_mode': response_mode.value,
        'user': user,
        'auto_generate_name': auto_generate_name,
        'inputs': inputs if inputs else {},
        'conversation_id': conversation_id if conversation_id else None,
        'files': [{
            'type': 'image',
            'transfer_method': 'local_file',
            'upload_file_id': file_id
        } for file_id in uploaded_file_ids] if uploaded_file_ids else []
    }
    
    chat_messages_url = f'{api_base_url}/chat-messages'
    return chat_headers, payload, chat_messages_url


def handle_blocking_response(headers: dict, payload: dict, chat_messages_url: str) -> dict:
    """
    处理阻塞模式的响应
    """
    response = requests.post(chat_messages_url, headers=headers, json=payload)
    dify_api_error_handler(response)
    answer = response.json()['answer']
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-3.5-turbo-0301",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": len(answer.split()),
            "total_tokens": 9 + len(answer.split())
        }
    }


# async def handle_streaming_response(headers: dict, payload: dict, chat_messages_url: str):
#     """
#     处理流式响应
#     """
#     #client = httpx.AsyncClient()
    
#     response = requests.post(chat_messages_url, headers=headers, json=payload, stream=True)
#     #response = client.stream("POST", chat_messages_url, headers=headers, json=payload)
#     dify_api_error_handler(response)

#     buffer = ""
#     for chunk in response.iter_content(chunk_size=32):
#         if chunk:
#             buffer += chunk.decode('utf-8')
#             while '\n' in buffer:
#                 line, buffer = buffer.split('\n', 1)
#                 line = line.strip()
#                 if not line:
#                     continue
#                 if line.startswith('data: '):
#                     line = line[6:]
#                 try:
#                     data = json.loads(line)
#                     if (data['event'] == 'message' and 'answer' in data) or data['event'] == 'message_end':
#                         #yield f"data: {json.dumps({'choices': [{'delta': {'content': data['answer']}}]})}\n\n"
#                         yield data
#                 except json.JSONDecodeError:
#                     continue

async def handle_streaming_response(headers: dict, payload: dict, chat_messages_url: str):
    """
    处理流式响应
    """
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", chat_messages_url, headers=headers, json=payload) as response:
            dify_api_error_handler(response)

            buffer = ""
            async for chunk in response.aiter_bytes():
                if chunk:
                    buffer += chunk.decode('utf-8')
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith('data: '):
                            line = line[6:]
                        try:
                            data = json.loads(line)
                            if (data['event'] == 'message' and 'answer' in data) or data['event'] == 'message_end':
                                yield data
                        except json.JSONDecodeError:
                            continue


def call_dify_api(api_key: str, query: str, response_mode: ResponseMode, user: str, 
                  files: list[str] = None, conversation_id: str = None,
                  inputs: dict = None, auto_generate_name: bool = True, 
                  api_base_url: str = 'http://10.144.129.132/v1') -> str:
    """
    调用 Dify API 的主函数
    
    参数:
        api_key (str): API 密钥，用于授权
        query (str): 用户的查询或消息内容
        response_mode (ResponseMode): 响应模式（STREAMING 或 BLOCKING）
        user (str): 用户标识
        files (list[str], optional): 要上传的本地文件路径列表
        conversation_id (str, optional): 会话ID
        inputs (dict, optional): 额外的输入参数
        auto_generate_name (bool, optional): 是否自动生成标题
        api_base_url (str, optional): API的基础URL

    返回:
        str: API的响应内容
    """
    headers, payload, chat_messages_url = prepare_chat_request(
        api_key, query, response_mode, user, files, 
        conversation_id, inputs, auto_generate_name, api_base_url
    )

    if response_mode == ResponseMode.BLOCKING:
        return handle_blocking_response(headers, payload, chat_messages_url)
    elif response_mode == ResponseMode.STREAMING:
        return handle_streaming_response(headers, payload, chat_messages_url)
    else:
        raise DifyAPIError(f"不支持的响应模式: {response_mode}")


async def call_dify_api_streaming(api_key: str, query: str, response_mode: ResponseMode, user: str, 
                  files: list[str] = None, conversation_id: str = None,
                  inputs: dict = None, auto_generate_name: bool = True, 
                  api_base_url: str = 'http://10.144.129.132/v1') -> AsyncIterator[GenericStreamingChunk]:
    """
    调用 Dify API 的主函数
    
    参数:
        api_key (str): API 密钥，用于授权
        query (str): 用户的查询或消息内容
        response_mode (ResponseMode): 响应模式（STREAMING 或 BLOCKING）
        user (str): 用户标识
        files (list[str], optional): 要上传的本地文件路径列表
        conversation_id (str, optional): 会话ID
        inputs (dict, optional): 额外的输入参数
        auto_generate_name (bool, optional): 是否自动生成标题
        api_base_url (str, optional): API的基础URL

    返回:
        str: API的响应内容
    """
    headers, payload, chat_messages_url = prepare_chat_request(
        api_key, query, response_mode, user, files, 
        conversation_id, inputs, auto_generate_name, api_base_url
    )
    
    chunks = handle_streaming_response(headers, payload, chat_messages_url)
    idx = 0
    async for chunk in chunks:
        finish_flag = chunk.get("event") == "message_end"
        completion_tokens = 0
        prompt_tokens = 0
        total_tokens = 0
        if finish_flag == True:
            completion_tokens = chunk.get("metadata", {}).get("usage", {}).get("completion_tokens", 0)
            prompt_tokens = chunk.get("metadata", {}).get("usage", {}).get("prompt_tokens", 0)
            total_tokens = chunk.get("metadata", {}).get("usage", {}).get("total_tokens", 0)
        
        generic_streaming_chunk: GenericStreamingChunk = {
            "finish_reason": "stop" if finish_flag == True else None,
            "index": idx,
            "is_finished": finish_flag,
            "text": chunk.get("answer", ""),
            "tool_use": None,
            "usage": {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "total_tokens": total_tokens},
        }
        
        yield generic_streaming_chunk # type: ignore
        
        idx += 1




def pre_process_messages(messages: List[dict], file_tmp_dir: str) -> tuple[str, List[str]]:
    """
    处理消息列表，提取文本内容并保存图片文件
    
    Args:
        messages (List[dict]): 消息列表
        file_tmp_dir (str): 临时文件目录路径
        
    Returns:
        tuple[str, List[str]]: 返回处���后的查询文本和文件路径列表
    """
    files = []
    dify_messages = []
    
    for message in messages:
        if message["role"] == "user":
            if isinstance(message["content"], list):
                new_content = []
                for content in message["content"]:
                    if content["type"] == "text":
                        new_content.append(content["text"])
                    elif content["type"] == "image_url":
                        image_url = content["image_url"]["url"]
                        if re.match(r"data:image/.*;base64,", image_url):
                            # 解码base64图像
                            image_data = base64.b64decode(image_url.split(",")[1])
                            # 提取扩展名
                            extension = image_url.split(";")[0].split("/")[1]
                            file_path = os.path.join(file_tmp_dir, f"{uuid4()}.{extension}")
                            
                            # 保存图像到文件
                            with open(file_path, "wb") as f:
                                f.write(image_data)
                            
                            files.append(file_path)
                # 重组消息内容
                message["content"] = " ".join(new_content)
        dify_messages.append(message)
    
    #query = " ".join([msg["content"] for msg in dify_messages if msg["role"] == "user"]).strip()
    query = str(dify_messages)
    return query, files


def post_process_messages(files: List[str]):
    for file in files:
        if os.path.exists(file):
            os.remove(file)


class DifyLLM(CustomLLM):
    
    def __init__(self):
        self.total_files = []
        super().__init__()
    
    def __clear_files(self):
        for file in self.total_files:
            if os.path.exists(file):
                os.remove(file)
        self.total_files = []
        
    def __exit__(self):
        self.__clear_files()
        super().__exit__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.__clear_files()
        await super().__aexit__(exc_type, exc_val, exc_tb)
    
    def completion(self, *args, **kwargs) -> litellm.ModelResponse:
        return litellm.completion(
            model="poll_gemini-1.5-flash-latest",
            messages=[{"role": "user", "content": "Hello world"}],
            mock_response="Hi!",
        )  # type: ignore
        
    def streaming(self, *args, **kwargs) -> Iterator[GenericStreamingChunk]:
        generic_streaming_chunk: GenericStreamingChunk = {
            "finish_reason": "stop",
            "index": 0,
            "is_finished": True,
            "text": str(int(time.time())),
            "tool_use": None,
            "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
        }
        return generic_streaming_chunk # type: ignore
        
    async def astreaming(self, *args, **kwargs) -> AsyncIterator[GenericStreamingChunk]:
        # generic_streaming_chunk: GenericStreamingChunk = {
        #     "finish_reason": "stop",
        #     "index": 0,
        #     "is_finished": True,
        #     "text": str(int(time.time())),
        #     "tool_use": None,
        #     "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0},
        # }
        # yield generic_streaming_chunk # type: ignore
        
        # 从kwargs中获取必要参数
        file_tmp_dir = kwargs.get("optional_params", {}).get("file_tmp_dir")
        api_base_url = kwargs.get("litellm_params", {}).get("api_base", "http://10.144.129.132/v1")
        api_key = kwargs.get("litellm_params", {}).get("api_key")
        model_response = kwargs.get("model_response", litellm.ModelResponse())
        messages = kwargs.get("messages", [])
        user = kwargs.get("user", "default_user")
        
        # 构建查询文本和文件路径列表
        query, files = pre_process_messages(messages, file_tmp_dir)
        
        # 设置响应模式
        response_mode = ResponseMode.STREAMING
        
        try:
            # 调用 Dify API
            response = call_dify_api_streaming(
                api_key=api_key,
                query=query,
                response_mode=response_mode,
                user=user,
                files=files
            )
            
            async for chunk in response:
                yield chunk
                
        except Exception as e:
            #post_process_messages(files)
            raise Exception(f"Dify API 调用失败: {str(e)}")
        finally:
            post_process_messages(files)
        
        

    async def acompletion(self, *args, **kwargs) -> litellm.ModelResponse:
        # 从kwargs中获取必要参数
        file_tmp_dir = kwargs.get("optional_params", {}).get("file_tmp_dir")
        api_base_url = kwargs.get("litellm_params", {}).get("api_base", "http://10.144.129.132/v1")
        api_key = kwargs.get("litellm_params", {}).get("api_key")
        model_response = kwargs.get("model_response", litellm.ModelResponse())
        messages = kwargs.get("messages", [])
        stream = False
        user = kwargs.get("user", "default_user")
        
        # 构建查询文本和文件路径列表
        query, files = pre_process_messages(messages, file_tmp_dir)
        
        # 设置响应模式
        response_mode = ResponseMode.STREAMING if stream else ResponseMode.BLOCKING
        
        try:
            # 调用 Dify API
            response = call_dify_api(
                api_key=api_key,
                query=query,
                response_mode=response_mode,
                user=user,
                files=files
            )
            
            # 将 Dify 响应转换为 ModelResponse 格式
            if isinstance(response, dict):
                model_response.choices[0].finish_reason = response.get("choices", [])[0].get("finish_reason", "stop")
                _message = litellm.Message(**response.get("choices", [])[0].get("message"))
                model_response.choices[0].message = _message
                model_response.created = response.get("created", int(time.time()))
                model_response.model = "dify_api/" + kwargs.get("model", "dify")
                setattr(
                    model_response,
                    "usage",
                    litellm.Usage(
                        prompt_tokens=response.get("usage", {}).get("prompt_tokens", 0),
                        completion_tokens=response.get("usage", {}).get("completion_tokens", 0),
                        total_tokens=response.get("usage", {}).get("total_tokens", 0),
                    ),
                )
                return model_response
            else:
                # 处理流式响应
                return response
                
        except Exception as e:
            post_process_messages(files)
            raise Exception(f"Dify API 调用失败: {str(e)}")
        finally:
            post_process_messages(files)
        


dify_api_llm = DifyLLM()