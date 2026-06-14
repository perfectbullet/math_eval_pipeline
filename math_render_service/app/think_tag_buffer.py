"""ThinkTagBuffer 用于过滤流式文本中的 <think>...</think> 内容。"""

from typing import Optional


class ThinkTagBuffer:
    """流式过滤 <think>...</think> 标签内容。

    设计目标：
    1. 支持 token 任意切分，例如 '<thi' + 'nk>'；
    2. 支持一个 token 中同时包含普通文本、think 标签和结束后的普通文本；
    3. 支持大小写标签，例如 <Think>、</THINK>；
    4. 支持 opening tag 带属性，例如 <think type="reasoning">；
    5. 保持原来的 add / flush / is_in_think_tag 接口不变。
    """

    STATE_OUTSIDE = "outside"
    STATE_DETECT_OPEN = "detect_open"
    STATE_INSIDE = "inside"
    STATE_DETECT_CLOSE = "detect_close"

    _OPEN_HEAD = "<think"
    _CLOSE_HEAD = "</think"
    _MAX_TAG_LEN = 256

    def __init__(
        self,
        *,
        collect_think: bool = False,
        strip_think_content: bool = False,
    ):
        self.state = self.STATE_OUTSIDE
        self.detection_buffer = ""

        # 保留这个字段，避免外部代码依赖旧属性时报错
        self.pending_output = ""

        # 可选：收集 think 内容，默认不收集，只过滤
        self.collect_think = collect_think
        self.strip_think_content = strip_think_content
        self.think_blocks: list[str] = []
        self._current_think_parts: list[str] = []

    def add(self, token: str) -> Optional[str]:
        """添加一个流式 token，返回过滤后的可见文本。

        Args:
            token: 当前流式 token / chunk

        Returns:
            过滤后的可见文本；如果当前 token 全部被过滤，则返回 None。
        """
        if not token:
            return None

        output_parts: list[str] = []

        for ch in token:
            if self.state == self.STATE_OUTSIDE:
                self._handle_outside_char(ch, output_parts)

            elif self.state == self.STATE_DETECT_OPEN:
                self._handle_detect_open_char(ch, output_parts)

            elif self.state == self.STATE_INSIDE:
                self._handle_inside_char(ch)

            elif self.state == self.STATE_DETECT_CLOSE:
                self._handle_detect_close_char(ch)

            else:
                # 理论兜底：未知状态时回到 outside，避免死锁
                self.state = self.STATE_OUTSIDE
                self.detection_buffer = ""
                output_parts.append(ch)

        result = "".join(output_parts)
        return result if result else None

    def _handle_outside_char(self, ch: str, output_parts: list[str]) -> None:
        """处理 think 外部的普通字符。"""
        if ch == "<":
            self.detection_buffer = ch
            self.state = self.STATE_DETECT_OPEN
        else:
            output_parts.append(ch)

    def _handle_detect_open_char(self, ch: str, output_parts: list[str]) -> None:
        """检测 opening tag: <think> 或 <think ...>。"""
        self.detection_buffer += ch

        if self._is_complete_open_tag(self.detection_buffer):
            self._enter_think()
            return

        if self._is_possible_open_tag_prefix(self.detection_buffer):
            return

        # 不是 think 标签，比如 <div>、<thinking>，原样输出
        output_parts.append(self.detection_buffer)
        self.detection_buffer = ""
        self.state = self.STATE_OUTSIDE

    def _handle_inside_char(self, ch: str) -> None:
        """处理 think 内部字符。"""
        if ch == "<":
            self.detection_buffer = ch
            self.state = self.STATE_DETECT_CLOSE
            return

        self._append_think_content(ch)

    def _handle_detect_close_char(self, ch: str) -> None:
        """检测 closing tag: </think> 或 </think >。"""
        self.detection_buffer += ch

        if self._is_complete_close_tag(self.detection_buffer):
            self._exit_think()
            return

        if self._is_possible_close_tag_prefix(self.detection_buffer):
            return

        # 在 think 内部遇到的普通 '<xxx'，不是结束标签，继续丢弃
        self._append_think_content(self.detection_buffer)
        self.detection_buffer = ""
        self.state = self.STATE_INSIDE

    @classmethod
    def _is_possible_open_tag_prefix(cls, text: str) -> bool:
        """判断 text 是否仍可能成为 <think...> opening tag。"""
        if len(text) > cls._MAX_TAG_LEN:
            return False

        lower = text.lower()

        # '<', '<t', '<th', '<thi', '<thin', '<think'
        if len(lower) <= len(cls._OPEN_HEAD):
            return cls._OPEN_HEAD.startswith(lower)

        if not lower.startswith(cls._OPEN_HEAD):
            return False

        # <thinking> 不是 think 标签
        next_ch = lower[len(cls._OPEN_HEAD)]
        if next_ch not in (">", " ", "\t", "\n", "\r"):
            return False

        # 已经进入属性区，只要还没看到 '>'，就继续等待
        return ">" not in lower

    @classmethod
    def _is_complete_open_tag(cls, text: str) -> bool:
        """判断 text 是否是完整的 <think...> opening tag。"""
        lower = text.lower()

        if not lower.startswith(cls._OPEN_HEAD):
            return False

        if len(lower) <= len(cls._OPEN_HEAD):
            return False

        next_ch = lower[len(cls._OPEN_HEAD)]

        # 必须是 <think> 或 <think xxx>，不能是 <thinking>
        if next_ch not in (">", " ", "\t", "\n", "\r"):
            return False

        return lower.endswith(">")

    @classmethod
    def _is_possible_close_tag_prefix(cls, text: str) -> bool:
        """判断 text 是否仍可能成为 </think> closing tag。"""
        if len(text) > cls._MAX_TAG_LEN:
            return False

        lower = text.lower()

        # '<', '</', '</t', '</th', ...
        if len(lower) <= len(cls._CLOSE_HEAD):
            return cls._CLOSE_HEAD.startswith(lower)

        if not lower.startswith(cls._CLOSE_HEAD):
            return False

        # </thinking> 不是 closing tag
        tail = lower[len(cls._CLOSE_HEAD):]

        # 支持 </think> 和 </think   >
        return all(ch.isspace() or ch == ">" for ch in tail) and ">" not in tail

    @classmethod
    def _is_complete_close_tag(cls, text: str) -> bool:
        """判断 text 是否是完整的 </think> closing tag。"""
        lower = text.lower()

        if not lower.startswith(cls._CLOSE_HEAD):
            return False

        tail = lower[len(cls._CLOSE_HEAD):]

        # 支持 </think> 和 </think   >
        return bool(tail) and tail.endswith(">") and all(
            ch.isspace() or ch == ">" for ch in tail
        )

    def _enter_think(self) -> None:
        """进入 think 内容区。"""
        self.state = self.STATE_INSIDE
        self.detection_buffer = ""
        self._current_think_parts = []

    def _exit_think(self) -> None:
        """退出 think 内容区。"""
        if self.collect_think:
            content = "".join(self._current_think_parts)
            if self.strip_think_content:
                content = content.strip()
            self.think_blocks.append(content)

        self.state = self.STATE_OUTSIDE
        self.detection_buffer = ""
        self._current_think_parts = []

    def _append_think_content(self, text: str) -> None:
        """按需收集 think 内容；默认只是丢弃。"""
        if self.collect_think:
            self._current_think_parts.append(text)

    def flush(self) -> Optional[str]:
        """结束流时刷新剩余内容。

        注意：
        - 如果正在检测 opening tag，但最终不是完整标签，则原样输出；
        - 如果已经进入 think 内部，则剩余内容视为未闭合 think，直接丢弃；
        - flush 一般应该只在流结束时调用。
        """
        result = None

        if self.state == self.STATE_DETECT_OPEN and self.detection_buffer:
            result = self.detection_buffer

        elif self.state in (self.STATE_INSIDE, self.STATE_DETECT_CLOSE):
            if self.collect_think:
                if self.state == self.STATE_DETECT_CLOSE and self.detection_buffer:
                    self._current_think_parts.append(self.detection_buffer)

                content = "".join(self._current_think_parts)
                if self.strip_think_content:
                    content = content.strip()
                if content:
                    self.think_blocks.append(content)

            result = None

        self.state = self.STATE_OUTSIDE
        self.detection_buffer = ""
        self.pending_output = ""
        self._current_think_parts = []

        return result

    def is_in_think_tag(self) -> bool:
        """检查当前是否处于 think 内容内部。"""
        return self.state in {
            self.STATE_INSIDE,
            self.STATE_DETECT_CLOSE,
        }

    def get_think_blocks(self) -> list[str]:
        """返回已经收集到的 think 内容。"""
        return list(self.think_blocks)

    def get_think_content(self, sep: str = "\n\n") -> str:
        """将已经收集到的 think 内容拼接为字符串。"""
        return sep.join(self.think_blocks)