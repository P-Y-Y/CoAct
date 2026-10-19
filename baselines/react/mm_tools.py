#!/usr/bin/env python
# coding=utf-8

import base64
import json
import mimetypes
import os
import uuid
from typing import Optional, Any, List, Dict
import logging
import requests
from PIL import Image

# Attempt to import extra dependencies, handle gracefully if missing
try:
    from xml.dom import minidom
except ImportError:
    minidom = None
try:
    from openpyxl import load_workbook
except ImportError:
    load_workbook = None
try:
    from Bio import PDB
except ImportError:
    PDB = None
try:
    from pptx import Presentation
except ImportError:
    Presentation = None
import shutil
import textwrap
import threading
import time

# ReAct project specific imports
from .llm import LLM, OpenAILLM

logger = logging.getLogger(__name__)

# =============================================================================
# MM Tools Tracker Logic
# =============================================================================
_VISUAL_USAGE_BY_THREAD: Dict[int, Dict[str, Any]] = {}
_TEXT_USAGE_BY_THREAD: Dict[int, Dict[str, Any]] = {}
_AUDIO_USAGE_BY_THREAD: Dict[int, Dict[str, Any]] = {}

def _update_mm_usage(store: Dict[int, Dict[str, Any]], calls: int = 1, tokens: int = 0, time_sec: float = 0.0) -> None:
    try:
        tid = threading.get_ident()
    except Exception:
        return
    if tid not in store:
        store[tid] = {"calls": 0, "tokens": 0, "time": 0.0}
    
    store[tid]["calls"] += calls
    store[tid]["tokens"] += tokens
    store[tid]["time"] += time_sec

def get_thread_visual_usage(thread_id: Optional[int] = None) -> Dict[str, Any]:
    if thread_id is None: thread_id = threading.get_ident()
    return dict(_VISUAL_USAGE_BY_THREAD.get(thread_id, {}))

def get_thread_text_usage(thread_id: Optional[int] = None) -> Dict[str, Any]:
    if thread_id is None: thread_id = threading.get_ident()
    return dict(_TEXT_USAGE_BY_THREAD.get(thread_id, {}))

def get_thread_audio_usage(thread_id: Optional[int] = None) -> Dict[str, Any]:
    if thread_id is None: thread_id = threading.get_ident()
    return dict(_AUDIO_USAGE_BY_THREAD.get(thread_id, {}))

def reset_mm_thread_usage():
    """Clear mm usage for current thread."""
    try:
        tid = threading.get_ident()
        if tid in _VISUAL_USAGE_BY_THREAD: del _VISUAL_USAGE_BY_THREAD[tid]
        if tid in _TEXT_USAGE_BY_THREAD: del _TEXT_USAGE_BY_THREAD[tid]
        if tid in _AUDIO_USAGE_BY_THREAD: del _AUDIO_USAGE_BY_THREAD[tid]
    except Exception:
        pass

class MarkdownConverter:
    """
    A simple fallback MarkdownConverter. 
    In a real scenario, this would use a library like MarkItDown or Docling.
    """
    def convert(self, file_path: str):
        # Return an object with .title and .text_content
        result = type('Result', (object,), {'title': os.path.basename(file_path), 'text_content': ""})()
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            result.text_content = content
        except Exception as e:
            result.text_content = f"Error reading file: {e}"
        
        return result

class MessageRole:
    SYSTEM = "system"
    USER = "user"

class BaseMMTool:
    """Base class for MM tools to avoid conflict with ReAct's Tool dataclass."""
    name: str = ""
    description: str = ""
    inputs: Dict = {}
    output_type: str = "string"

    def __init__(self):
        pass

class VisualInspectorTool(BaseMMTool):
    name = "inspect_file_as_image"
    description = """
You cannot load files directly: use this tool to process image files and answer related questions.
This tool supports the following image formats: [".jpg", ".jpeg", ".png", ".gif", ".bmp"]. For other file types, use the appropriate inspection tool."""

    inputs = {
        "file_path": {
            "description": "The path to the file you want to read as an image. Must be a '.something' file, like '.jpg','.png','.gif'. If it is text, use the text_inspector tool instead! If it is audio, use the audio_inspector tool instead! DO NOT use this tool for an HTML webpage: use the web_search tool instead!",
            "type": "string",
        },
        "question": {
            "description": "[Optional]: Your question about the image content. Provide as much context as possible. Do not pass this parameter if you just want to get a description of the image.",
            "type": "string",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, model: LLM, text_limit: int = 4000):
        super().__init__()
        self.model = model
        self.text_limit = text_limit
        # Prioritize VISION_ named variables to avoid conflict with local vLLM OPENAI_API_KEY
        self.gpt_key = os.getenv("VISION_API_KEY", os.getenv("OPENAI_API_KEY"))
        
        # Determine Base URL
        env_vision_base = os.getenv("VISION_BASE_URL")
        env_openai_base = os.getenv("OPENAI_BASE_URL")
        
        # Logic: If using OpenRouter Key (sk-or-...) but no URL is explicitly set, force OpenRouter URL
        if self.gpt_key and self.gpt_key.startswith("sk-or-v1"):
             self.gpt_url = env_vision_base or "https://openrouter.ai/api/v1"
        else:
             self.gpt_url = env_vision_base or env_openai_base or "https://api.openai.com/v1"

        self.vision_model = os.getenv("VISION_MODEL", "gpt-4o")

    def _validate_file_type(self, file_path: str):
        if not any(file_path.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]):
            raise ValueError("Unsupported file type. Use the appropriate tool for text/audio files.")

    def _encode_image(self, image_path: str) -> str:
        if image_path.startswith("http"):
            user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
            request_kwargs = {
                "headers": {"User-Agent": user_agent},
                "stream": True,
            }

            response = requests.get(image_path, **request_kwargs)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")

            extension = mimetypes.guess_extension(content_type)
            if extension is None:
                extension = ".download"

            fname = str(uuid.uuid4()) + extension
            download_dir = os.path.abspath("downloads")
            os.makedirs(download_dir, exist_ok=True)
            download_path = os.path.join(download_dir, fname)

            with open(download_path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=512):
                    fh.write(chunk)

            image_path = download_path

        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


    def forward(self, file_path: str, question: Optional[str] = None) -> str:
        self._validate_file_type(file_path)
        logger.info("[Tool Check] Starting VisualInspectorTool.forward...")
        logger.info(f"    - File Path: {file_path}")
        logger.info(f"    - Question: {question}")
        logger.info(f"    - Using Model: {self.vision_model}")
        logger.info(f"    - Using API URL: {self.gpt_url}")
        
        if not question:
            question = "Please write a detailed caption for this image."
        try:
            mime_type, _ = mimetypes.guess_type(file_path)
            base64_image = self._encode_image(file_path)
            payload = {
                "model": self.vision_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                        ],
                    }
                ],
                "max_tokens": 2000,
                "top_p": 0.1,
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.gpt_key}"
            }
            
            t0 = time.time()
            response = requests.post(
                f"{self.gpt_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            t1 = time.time()
            
            resp_json = response.json()
            description = resp_json["choices"][0]["message"]["content"]
            
            # Record Usage
            usage = resp_json.get("usage", {})
            total_tokens = usage.get("total_tokens", 0)
            _update_mm_usage(_VISUAL_USAGE_BY_THREAD, calls=1, tokens=total_tokens, time_sec=(t1 - t0))
            
            logger.info(f"[Tool Check] VisualInspectorTool SUCCESS. Tokens: {total_tokens}, Time: {t1-t0:.3f}s")
        except Exception as gpt_error:
            # Even if failed, if we spent time, we might want to record call=1, time=xxx? 
            # But usually we only care about successful API calls or at least attempted ones.
            # Here we just log error.
            logger.error(f"[Tool Check] VisualInspectorTool FAILED: {str(gpt_error)}")
            return f"Visual processing failed: {str(gpt_error)}"

        if not question.startswith("Please write a detailed caption"):
            return description
        return f"You did not provide a particular question, so here is a detailed description of the image: {description}"
    
    def wrapped_call(self, action_input: Any) -> str:
        """Wrapper for ReAct tool call style"""
        if isinstance(action_input, dict):
            file_path = action_input.get("file_path")
            question = action_input.get("question")
            if not file_path:
                return "Error: file_path is required"
            return self.forward(file_path, question)
        else:
            # Assume strict string input is just file path
            return self.forward(str(action_input))


class TextInspectorTool(BaseMMTool):
    name = "inspect_file_as_text"
    description = """
You cannot load files yourself: instead call this tool to read a file as markdown text and ask questions about it.
This tool handles the following file extensions: [".html", ".pdb", ".xlsx", ".xls", ".pdf", ".docx", ".ppt", ".pptx"], and all other types of text files. IT DOES NOT HANDLE IMAGES."""

    inputs = {
        "file_path": {
            "description": "The path to the file you want to read as text. Must be a '.something' file, like '.pdf'. If it is an image, use the visualizer tool instead! If it is an audio, use the audio tool instead! DO NOT use this tool for an HTML webpage: use the web_search tool instead!",
            "type": "string",
        },
        "question": {
            "description": "[Optional]: Your question, as a natural language sentence. Provide as much context as possible. Do not pass this parameter if you just want to directly return the content of the file.",
            "type": "string",
            "nullable": True,
        },
    }
    output_type = "string"
    md_converter = MarkdownConverter()

    def __init__(self, model: LLM, text_limit: int = 10000):
        super().__init__()
        self.model = model
        self.text_limit = text_limit

    def _call_model(self, messages):
        """Adapted to work with ReAct OpenAILLM"""
        if isinstance(self.model, OpenAILLM):
            try:
                # messages here is a list of dicts with 'role' and 'content'.
                # The user's 'content' is sometimes a list of dicts (for multimodal) or strings.
                # OpenAI ChatCompletion expects string content usually, unless using vision.
                # But here we are text only.
                # However, the user code constructs content as a list: `{"type": "text", "text": ...}`
                
                # Adapting message format:
                cleaned_messages = []
                for msg in messages:
                    content_obj = msg.get("content")
                    text_content = ""
                    if isinstance(content_obj, list):
                        for part in content_obj:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text_content += part.get("text", "")
                    elif isinstance(content_obj, str):
                        text_content = content_obj
                    
                    cleaned_messages.append({"role": msg["role"], "content": text_content})
                
                t0 = time.time()
                resp = self.model.client.chat.completions.create(
                    model=self.model.model,
                    messages=cleaned_messages,
                    temperature=0.2
                    # extra_body logic handled in LLM but we bypass generate() so we might lose it.
                    # Ignoring extra_body for now as it's complex to replicate without access to LLM internals
                )
                t1 = time.time()
                
                # Record Usage
                if resp.usage:
                    p_t = resp.usage.prompt_tokens
                    c_t = resp.usage.completion_tokens
                    total_t = resp.usage.total_tokens
                    # 1. Update global LLM stats (since we bypassed self.model.generate)
                    self.model.record_usage(p_t, c_t, total_t)
                    try:
                        self.model.total_call_time += (t1 - t0)
                        # Also track as tool time since we are in a tool
                        if hasattr(self.model, "total_tool_call_time"):
                            self.model.total_tool_call_time += (t1 - t0)
                    except AttributeError:
                        pass
                    
                    # 2. Update Tool stats
                    # [Modified] Don't track LLM time in text_usage anymore, user wants it separate.
                    # We still track tokens though.
                    _update_mm_usage(_TEXT_USAGE_BY_THREAD, calls=1, tokens=total_t, time_sec=0)
                else:
                    try:
                        self.model.call_count += 1
                        self.model.total_call_time += (t1 - t0)
                        if hasattr(self.model, "total_tool_call_time"):
                            self.model.total_tool_call_time += (t1 - t0)
                    except AttributeError:
                        pass
                    _update_mm_usage(_TEXT_USAGE_BY_THREAD, calls=1, tokens=0, time_sec=0)

                return resp.choices[0].message.content
            except Exception as e:
                logger.error(f"TextInspectorTool model call failed: {e}")
                return f"Error calling model: {e}"
        else:
             return "Model type not supported for TextInspectorTool (requires OpenAILLM)"

    def jsonld_to_markdown(self, data):
        markdown = ""
        if isinstance(data, dict):
            for key, value in data.items():
                markdown += f"**{key}**: {self.jsonld_to_markdown(value)}\n"
        elif isinstance(data, list):
            for item in data:
                markdown += f"- {self.jsonld_to_markdown(item)}\n"
        else:
            markdown += str(data)
        return markdown
    
    def parse_pdb_file(self, file_path):
        if PDB is None: return "Bio.PDB not installed"
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("protein", file_path)

        atoms = list(structure.get_atoms())
        if len(atoms) < 2:
            return "Error: PDB file contains fewer than two atoms."

        atom1, atom2 = atoms[0], atoms[1]
        distance = atom1 - atom2

        return f"First atom: {atom1.get_name()} ({atom1.coord})\n" \
            f"Second atom: {atom2.get_name()} ({atom2.coord})\n" \
            f"Distance: {distance:.3f} Angstroms ({distance * 100:.0f} pm)"

    def extract_excel_data(self, file_path, max_rows=30):
        if load_workbook is None: return "openpyxl not installed"
        try:
            workbook = load_workbook(file_path)
            result = []

            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]

                for row in sheet.iter_rows():
                    row_data = []
                    for cell in row:
                        cell_value = cell.value if cell.value is not None else ""

                        fill = cell.fill
                        if hasattr(fill, "fgColor") and fill.fgColor.type == "rgb":
                            cell_color = fill.fgColor.rgb
                            if len(cell_color) == 8:
                                cell_color = cell_color[2:]
                        else:
                            cell_color = "FFFFFF"
                        row_data.append({
                            "value": cell_value,
                            "color": cell_color
                        })
                    result.append(row_data)
                json_data = json.dumps(result, ensure_ascii=False, indent=4)
                data_with_colors = json.loads(json_data)
                text = []
                num_rows = len(data_with_colors)
                num_cols = len(data_with_colors[0]) if data_with_colors else 0
                text.append(f"This is a {num_rows} rows and {num_cols} columns table. The content is shown below:")
                for i, row in enumerate(data_with_colors):
                    row_text = ""
                    for j, cell in enumerate(row):
                        if cell["value"] != "":
                            value = cell["value"]
                        else:
                            value = "None"
                        color = cell["color"]
                        if color == "FFFFFF" or color == "000000":
                            row_text += f"{value} "
                        else:
                            row_text += f"{value}({color}) "
                    text.append(row_text)
            return "\n".join(text)
        except Exception as e:
            return f"Error: {str(e)}"

    def forward_initial_exam_mode(self, file_path, question):
        result = self.md_converter.convert(file_path)

        if file_path.endswith(".xml"):
            try:
                if minidom:
                    dom = minidom.parse(file_path)
                    result_text = " ".join(
                        [node.firstChild.nodeValue for node in dom.getElementsByTagName("*") if node.firstChild and node.firstChild.nodeType == node.TEXT_NODE]
                    )
                    result.text_content = result_text
            except Exception as e:
                raise Exception(f"Error parsing XML file: {str(e)}")

        if file_path.endswith(".csv"):
            try:
                with open(file_path, 'r') as fr:
                    contents = fr.readlines()
                result.text_content = "".join(contents)
            except Exception as e:
                raise Exception(f"Error parsing CSV file: {str(e)}")

        if file_path.endswith(".pdb"):
            try:
                pdb_info = self.parse_pdb_file(file_path)
                if not question:
                    return f"Extracted PDB Data:\n{pdb_info}"

                else:
                    result.text_content = pdb_info
            except Exception as e:
                raise Exception(f"Error parsing PDB file: {e}")

        if file_path.endswith((".ppt",  ".pptx")):
            if Presentation is None: return "python-pptx not installed"
            content = ""
            try:
                ppt = Presentation(file_path)
                for slide_number, slide in enumerate(ppt.slides, start=1):
                    content += f"=== Slide {slide_number} ===\n"
                    slide_texts = []
                    for shape in slide.shapes:
                        if hasattr(shape, "text") and shape.text.strip():
                            slide_texts.append(shape.text.strip())
                    content += '\n'.join(slide_texts) + '\n\n'
                result.text_content = content.strip()
            except Exception as e:
                return f"Error parsing PPT file: {e}"

        if file_path.endswith((".xls", ".xlsx")):
            try:
                full_content = self.extract_excel_data(file_path)
                if not question:
                    return full_content
                else:
                    result.text_content = full_content
            except Exception as e:
                raise Exception(f"Error processing Excel file: {e}")

        if file_path[-4:] in [".png", ".jpg"]:
            raise Exception("Cannot use inspect_file_as_text tool with images: use visualizer instead!")

        if ".zip" in file_path:
            return result.text_content
            
        if not question:
            return result.text_content

        if len(result.text_content) < 4000:
            return "Document content: " + result.text_content

        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": "Here is a file:\n### "
                        + str(result.title)
                        + "\n\n"
                        + result.text_content[: self.text_limit],
                    }
                ],
            },
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": "Now please write a short, 5 sentence caption for this document, that could help someone asking this question: "
                        + question
                        + "\n\nDon't answer the question yourself! Just provide useful notes on the document",
                    }
                ],
            },
        ]
        return self._call_model(messages)

    def forward(self, file_path, question: Optional[str] = None) -> str:
        logger.info("[Tool Check] Starting TextInspectorTool.forward...")
        logger.info(f"    - File Path: {file_path}")
        logger.info(f"    - Question: {question}")

        if ".jsonld" in file_path:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                result_text = self.jsonld_to_markdown(data)
                result = type('Result', (object,), {'title': file_path, 'text_content': result_text})()
            except Exception as e:
                logger.error(f"[Tool Check] TextInspectorTool JSON-LD Parsing FAILED: {str(e)}")
                raise Exception(f"Error parsing JSON-LD file: {str(e)}")
        else:
            result = self.md_converter.convert(file_path)

        if file_path[-4:] in [".png", ".jpg"]:
            logger.error("[Tool Check] TextInspectorTool FAILED: Cannot use with images")
            raise Exception("Cannot use inspect_file_as_text tool with images: use visualizer instead!")

        if ".zip" in file_path:
            logger.info("[Tool Check] TextInspectorTool SUCCESS (Zip Content).")
            return result.text_content

        if file_path.endswith((".ppt",  ".pptx")):
            if Presentation is None: return "python-pptx not installed"
            content = ""
            try:
                ppt = Presentation(file_path)
                for slide_number, slide in enumerate(ppt.slides, start=1):
                    content += f"=== Slide {slide_number} ===\n"
                    slide_texts = []
                    for shape in slide.shapes:
                        if hasattr(shape, "text") and shape.text.strip():
                            slide_texts.append(shape.text.strip())
                    content += '\n'.join(slide_texts) + '\n\n'
                result.text_content = content.strip()
            except Exception as e:
                logger.error(f"[Tool Check] TextInspectorTool PPT Parsing FAILED: {str(e)}")
                return f"Error parsing PPT file: {e}"

        if file_path.endswith((".xls", ".xlsx")):
            try:
                full_content = self.extract_excel_data(file_path)
                if not question:
                    logger.info("[Tool Check] TextInspectorTool SUCCESS (Excel Content).")
                    return full_content
                else:
                    result.text_content = full_content
            except Exception as e:
                logger.error(f"[Tool Check] TextInspectorTool Excel Parsing FAILED: {str(e)}")
                raise Exception(f"Error processing Excel file: {e}")

        if file_path.endswith(".pdb"):
            try:
                pdb_info = self.parse_pdb_file(file_path)
                if not question:
                    logger.info("[Tool Check] TextInspectorTool SUCCESS (PDB Content).")
                    return f"Extracted PDB Data:\n{pdb_info}"

                else:
                    result.text_content = pdb_info
            except Exception as e:
                logger.error(f"[Tool Check] TextInspectorTool PDB Parsing FAILED: {str(e)}")
                raise Exception(f"Error parsing PDB file: {e}")
            
        if not question:
            logger.info("[Tool Check] TextInspectorTool SUCCESS (Direct Content).")
            return result.text_content

        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": "You will have to write a short caption for this file, then answer this question:"
                        + question,
                    }
                ],
            },
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": "Here is the complete file:\n### "
                        + str(result.title)
                        + "\n\n"
                        + result.text_content[: self.text_limit],
                    }
                ],
            },
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": "Now answer the question below. Use these three headings: '1. Short answer', '2. Extremely detailed answer', '3. Additional Context on the document and question asked'."
                        + question,
                    }
                ],
            },
        ]
        response = self._call_model(messages)
        logger.info("[Tool Check] TextInspectorTool SUCCESS (Model Response).")
        return response
    
    def wrapped_call(self, action_input: Any) -> str:
        if isinstance(action_input, dict):
            file_path = action_input.get("file_path")
            question = action_input.get("question")
            if not file_path:
                return "Error: file_path is required"
            return self.forward(file_path, question)
        else:
             return self.forward(str(action_input))


class AudioInspectorTool(BaseMMTool):
    name = "inspect_file_as_audio"
    description = """
You cannot load files directly: use this tool to process audio files and answer related questions.
This tool supports the following audio formats: [".mp3", ".m4a", ".wav"]. For other file types, use the appropriate inspection tool."""

    inputs = {
        "file_path": {
            "description": "The path to the file you want to read as audio. Must be a '.something' file, like '.mp3','.m4a','.wav'. If it is an text, use the text_inspector tool instead! If it is an image, use the visualizer tool instead! DO NOT use this tool for an HTML webpage: use the web_search tool instead!",
            "type": "string",
        },
        "question": {
            "description": "[Optional]: Your question about the audio content. Provide as much context as possible. Do not pass this parameter if you just want to directly return the content of the file.",
            "type": "string",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, model: LLM, text_limit: int = 10000):
        super().__init__()
        self.model = model
        self.text_limit = text_limit
        self.audio_model = os.getenv("MTU_MODEL", "whisper-1")

    def _call_model(self, messages):
        # Same as TextInspectorTool adapter
        if isinstance(self.model, OpenAILLM):
            try:
                cleaned_messages = []
                for msg in messages:
                    content_obj = msg.get("content")
                    text_content = ""
                    if isinstance(content_obj, list):
                        for part in content_obj:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text_content += part.get("text", "")
                    elif isinstance(content_obj, str):
                        text_content = content_obj
                    
                    cleaned_messages.append({"role": msg["role"], "content": text_content})
                
                t0 = time.time()
                resp = self.model.client.chat.completions.create(
                    model=self.model.model,
                    messages=cleaned_messages,
                    temperature=0.2
                )
                t1 = time.time()

                # Record Usage
                if resp.usage:
                    p_t = resp.usage.prompt_tokens
                    c_t = resp.usage.completion_tokens
                    total_t = resp.usage.total_tokens
                    # 1. Update global LLM stats
                    self.model.record_usage(p_t, c_t, total_t)
                    try:
                        self.model.call_count += 1
                        self.model.total_call_time += (t1 - t0)
                        if hasattr(self.model, "total_tool_call_time"):
                             self.model.total_tool_call_time += (t1 - t0)
                    except AttributeError:
                        pass
                    
                    # 2. Update Tool stats
                    # [Modified] NO LONGER track LLM tokens in audio usage.
                    # We only want to track Whisper transcription length as tokens.
                    pass
                else:
                    try:
                        self.model.call_count += 1
                        self.model.total_call_time += (t1 - t0)
                        if hasattr(self.model, "total_tool_call_time"):
                             self.model.total_tool_call_time += (t1 - t0)
                    except AttributeError:
                        pass
                    pass

                return resp.choices[0].message.content
            except Exception as e:
                logger.error(f"AudioInspectorTool model call failed: {e}")
                return f"Error calling model: {e}"
        else:
             return "Model type not supported for AudioInspectorTool (requires OpenAILLM)"

    def _validate_file_type(self, file_path: str):
        """Validate if the file type is a supported audio format"""
        if not any(file_path.endswith(ext) for ext in [".mp3", ".m4a", ".wav"]):
            raise ValueError("Unsupported file type. Use the appropriate tool for text/image files.")

    def transcribe_audio(self, file_path: str) -> str:
        """Transcribe audio using Faster-Whisper (local)"""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            try:
                # Mock if not available
                logger.warning("faster-whisper not installed. Returning mock transcription.")
                return "Audio transcription is not available because faster-whisper is not installed."
            except ImportError:
                return "faster-whisper module missing"

        # Map 'whisper-1' (OpenAI) to a suitable local model size, or use what's provided
        local_model_size = self.audio_model
        if local_model_size == "whisper-1":
            local_model_size = "medium" # downgraded from large-v3 for safety
        
        try:
            logger.info(f"[AudioInspectorTool] Loading faster-whisper model: {local_model_size}")
            device = "cpu" 
            compute_type = "int8"
            
            model = WhisperModel(local_model_size, device=device, compute_type=compute_type, cpu_threads=4)

            segments, info = model.transcribe(file_path, beam_size=5)
            
            transcription = ""
            for segment in segments:
                transcription += segment.text
            return transcription
        except Exception as e:
            raise RuntimeError(f"Speech recognition failed: {str(e)}") from e

    def forward(self, file_path: str, question: Optional[str] = None) -> str:
        try:
            self._validate_file_type(file_path)
            logger.info("[Tool Check] Starting AudioInspectorTool.forward...")
            logger.info(f"    - File Path: {file_path}")
            logger.info(f"    - Question: {question}")
            
            t_w_start = time.time()
            transcript = self.transcribe_audio(file_path)
            t_w_end = time.time()
            
            # Record Whisper Time & Call Count & Tokens (Transcript length)
            # [Modified] tokens=len(transcript) to represent pure Whisper output
            transcript_tokens = len(transcript.split()) # Approximate tokens by words
            _update_mm_usage(_AUDIO_USAGE_BY_THREAD, calls=1, tokens=transcript_tokens, time_sec=(t_w_end - t_w_start))
        except Exception as e:
            logger.error(f"[Tool Check] AudioInspectorTool FAILED (Validation/Transcribe): {str(e)}")
            return f"Audio processing error: {str(e)}"
        
        if not question:
            logger.info("[Tool Check] AudioInspectorTool SUCCESS (Transcription Only).")
            return f"Audio transcription:\n{transcript[:self.text_limit]}"

        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [{
                    "type": "text",
                    "text": f"Audio transcription:\n{transcript[:self.text_limit]}"
                }]
            },
            {
                "role": MessageRole.USER,
                "content": [{
                    "type": "text",
                    "text": f"Answer the following question based on the audio content using the format:\n"
                            "1. Brief answer\n2. Detailed analysis\n3. Relevant context\nQuestion: {question}"
                }]
            }
        ]
        
        response = self._call_model(messages)
        logger.info("[Tool Check] AudioInspectorTool SUCCESS (Model Response).")
        return response

    def wrapped_call(self, action_input: Any) -> str:
        if isinstance(action_input, dict):
            file_path = action_input.get("file_path")
            question = action_input.get("question")
            if not file_path:
                return "Error: file_path is required"
            return self.forward(file_path, question)
        else:
             return self.forward(str(action_input))