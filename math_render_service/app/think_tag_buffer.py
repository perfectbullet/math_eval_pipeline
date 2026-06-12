"""Think Tag Buffer 用于过滤流式文本中的 think 标签。"""
import re
from typing import Optional


class ThinkTagBuffer:
    """缓冲区用于过滤流式文本中的  标签。

    状态机:
    - outside: 正常输出，token 直接返回
    - detect_open: 检测到 '<'，正在确认是否是 'think>'
    - inside: 在 think 标签内，丢弃内容
    - detect_close: 检测到 '</'，正在确认是否是 'think>'
    """

    # 状态常量
    STATE_OUTSIDE = "outside"
    STATE_DETECT_OPEN = "detect_open"
    STATE_INSIDE = "inside"
    STATE_DETECT_CLOSE = "detect_close"

    def __init__(self):
        """初始化 ThinkTagBuffer。"""
        self.state = self.STATE_OUTSIDE
        self.detection_buffer = ""  # 用于累积检测标签
        self.pending_output = ""    # 待输出内容

    def add(self, token: str) -> Optional[str]:
        """添加一个 token 并返回过滤后的内容。

        Args:
            token: 当前流式 token

        Returns:
            过滤后的内容，如果内容被过滤则返回 None
        """
        if not token:
            return None
        
        if self.state == self.STATE_OUTSIDE:
            return self._handle_outside(token)
        elif self.state == self.STATE_DETECT_OPEN:
            return self._handle_detect_open(token)
        elif self.state == self.STATE_INSIDE:
            return self._handle_inside(token)
        elif self.state == self.STATE_DETECT_CLOSE:
            return self._handle_detect_close(token)

        return None

    def _handle_outside(self, token: str) -> Optional[str]:
        """处理 outside 状态。"""
        # 检查是否以 '<' 开头（可能是 think 标签开始）
        if token.startswith('<'):
            self.detection_buffer = token
            self.state = self.STATE_DETECT_OPEN
            return None
        else:
            # 正常内容，直接输出
            return token

    def _handle_detect_open(self, token: str) -> Optional[str]:
        """正在检测是否是 think> 标签。"""
        self.detection_buffer += token

        # 去除空白字符后再检测
        buffer_clean = self.detection_buffer.strip()

        # 检查累积内容是否包含 'think>'
        if 'think>' in buffer_clean:
            # 确认是 think 标签，进入 inside 状态
            self.state = self.STATE_INSIDE
            self.detection_buffer = ""
            return None

        # 检查是否包含其他标签的结束符号 '>'（但不是 think>）
        if '>' in buffer_clean:
            # 找到 '>' 但不是 think>，判定不是 think 标签
            result = self.detection_buffer
            self.detection_buffer = ""
            self.state = self.STATE_OUTSIDE
            return result

        # 长度限制检查
        if len(self.detection_buffer) > 20:
            # 超过合理长度，判定不是 think 标签
            result = self.detection_buffer
            self.detection_buffer = ""
            self.state = self.STATE_OUTSIDE
            return result

        return None

    def _handle_inside(self, token: str) -> Optional[str]:
        """在 think 标签内，丢弃所有内容。"""
        # 检查是否以 '</' 开头（可能是 think 标签结束）
        if token.startswith('</'):
            self.detection_buffer = token
            self.state = self.STATE_DETECT_CLOSE
        # 丢弃所有内容
        return None

    def _handle_detect_close(self, token: str) -> Optional[str]:
        """正在检测是否是 think> 结束标签。"""
        self.detection_buffer += token

        # 去除空白字符后再检测
        buffer_clean = self.detection_buffer.strip()

        # 检查累积内容是否包含 'think>'
        if 'think>' in buffer_clean:
            # 确认是 think 结束标签，回到 outside 状态
            self.state = self.STATE_OUTSIDE
            self.detection_buffer = ""
            return None
        elif len(self.detection_buffer) > 20:
            # 超过合理长度，判定不是 think 结束标签
            self.detection_buffer = ""
            self.state = self.STATE_INSIDE
            return None

        return None

    def flush(self) -> Optional[str]:
        """刷新任何剩余的缓冲区内容。"""
        result_parts = []

        # 添加待输出内容
        if self.pending_output:
            result_parts.append(self.pending_output)
            self.pending_output = ""

        # 如果在检测状态且有累积内容，也要输出
        if self.detection_buffer and self.state != self.STATE_INSIDE:
            result_parts.append(self.detection_buffer)
            self.detection_buffer = ""

        # 重置状态
        if self.state != self.STATE_INSIDE:
            self.state = self.STATE_OUTSIDE

        result = ''.join(result_parts)
        return result if result else None

    def is_in_think_tag(self) -> bool:
        """检查当前是否在 think 标签内。"""
        return self.state == self.STATE_INSIDE
