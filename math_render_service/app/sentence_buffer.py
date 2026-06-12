"""Sentence Buffer for streaming text with intelligent segmentation.

Supports:
- Sentence-level punctuation-based splitting (highest priority)
- Character limit forced splitting
- Time limit forced splitting
- Comma-based splitting (lower priority than sentence endings)
- LaTeX formula integrity preservation
"""
import re
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class SentenceSegment:
    """A single sentence segment ready for output."""
    content: str
    has_formula: bool
    is_final: bool = False

    def __len__(self) -> int:
        return len(self.content)


SplitReason = Literal[
    "sentence_end",
    "comma",
    "char_limit_space",
    "char_limit_punctuation",
    "char_limit_forced",
    "char_limit_before_formula",
    "no_split",
    "unclosed_formula",
    "formula_at_start",
    "waiting_for_complete_formula",
    "latex_closing",
]


class SentenceBuffer:
    r"""Sentence buffer with intelligent segmentation and dual output support.

    Features:
    1. Punctuation-based splitting (。！？.!?\n) - highest priority
    2. Character limit splitting - forced when exceeded
    3. Time limit splitting - forced when timeout
    4. Comma-based splitting (，；、,;) - lower priority
    5. LaTeX formula integrity - $$...$$, $...$, \(...\), \[...\] protected
    6. Formula detection - automatic detection in sentences
    """

    # Regex patterns
    SENTENCE_END_PATTERN = re.compile(r'([。！？.!?\n])')
    COMMA_PATTERN = re.compile(r'([，；、,;])')
    CLOSED_FORMULA_PATTERN = re.compile(
        r'\$\$[\s\S]+?\$\$|'
        r'\$[^$\n]+?\$|'
        r'\\\([\s\S]+?\\\)|'
        r'\\\[[\s\S]+?\\\]'
    )
    ESCAPED_DOLLAR_PATTERN = re.compile(r'\\\$')
    NON_LETTER_DIGIT_PATTERN = re.compile(r'[a-zA-Z0-9]')
    PUNCTUATION_ONLY_PATTERN = re.compile(r'^[\s\n\r。！？.,;:!?\-—\*\•]+$')
    PUNCTUATION_ONLY_EXTENDED_PATTERN = re.compile(r'^[\s\n\r。！？.,;:!?\-—\*\•\'\"]+$')
    SHORT_PREFIX_PATTERN = re.compile(r'^[0-9a-zA-Z]+[.：:：]$')
    SHORT_PREFIX_WITH_NEWLINE_PATTERN = re.compile(r'^[0-9a-zA-Z]+[.：:：]\s*$')

    # Bare \boxed{...} pattern (math model outputs without $ delimiters)
    _BARE_BOXED_PATTERN = re.compile(r'\\boxed\s*\{')

    # LaTeX environment delimiters: \begin{X} ... \end{X}
    _BEGIN_ENV_PATTERN = re.compile(r'\\begin\{([^}]+)\}')
    _END_ENV_PATTERN = re.compile(r'\\end\{([^}]+)\}')

    # LaTeX delimiter pairs for formula detection
    DELIMITER_PAIRS = [
        ('$$', '$$', 2),
        (r'\(', r'\)', 2),
        (r'\[', r'\]', 2),
    ]

    # Configuration constants
    SPACE_SEARCH_RANGE = 20

    # Delimiter patterns for counting
    _PAREN_OPEN_PATTERN = re.compile(r'\\\(')
    _PAREN_CLOSE_PATTERN = re.compile(r'\\\)')
    _BRACKET_OPEN_PATTERN = re.compile(r'\\\[')
    _BRACKET_CLOSE_PATTERN = re.compile(r'\\\]')

    def __init__(
        self,
        max_chars: int = 200,
        max_wait_seconds: float = 0.5,
        comma_split_threshold: int = 30,
    ):
        self.buffer = ""
        self._pending_merge = ""
        self.max_chars = max_chars
        self.max_wait_seconds = max_wait_seconds
        self.comma_split_threshold = comma_split_threshold
        self.last_flush_time = time.time()
        self.is_flushed = True

    def _count_latex_delimiters(self, text: str) -> dict:
        """Count all LaTeX formula delimiters in text."""
        text_clean = self.ESCAPED_DOLLAR_PATTERN.sub('', text)

        display_dollar = text_clean.count('$$')
        remaining = text_clean.replace('$$', '')
        inline_dollar = remaining.count('$')

        paren_open = len(self._PAREN_OPEN_PATTERN.findall(text))
        paren_close = len(self._PAREN_CLOSE_PATTERN.findall(text))
        bracket_open = len(self._BRACKET_OPEN_PATTERN.findall(text))
        bracket_close = len(self._BRACKET_CLOSE_PATTERN.findall(text))

        return {
            "display_dollar": display_dollar,
            "inline_dollar": inline_dollar,
            "paren_open": paren_open,
            "paren_close": paren_close,
            "bracket_open": bracket_open,
            "bracket_close": bracket_close,
        }

    def _is_in_latex_formula(self, text: str, position: int) -> bool:
        """Check if a position is within a LaTeX formula delimiter."""
        counts = self._count_latex_delimiters(text[:position])
        if (
            counts["display_dollar"] % 2 == 1
            or counts["inline_dollar"] % 2 == 1
            or counts["paren_open"] > counts["paren_close"]
            or counts["bracket_open"] > counts["bracket_close"]
        ):
            return True
        # Check if inside a \begin{X}...\end{X} environment
        prefix = text[:position]
        if len(self._BEGIN_ENV_PATTERN.findall(prefix)) > len(self._END_ENV_PATTERN.findall(prefix)):
            return True
        # Check if inside a bare \boxed{...}
        if self._is_inside_bare_boxed(prefix):
            return True
        return False

    def _is_inside_bare_boxed(self, text: str) -> bool:
        """Check if text ends inside a bare (not in $...$) \\boxed{...}."""
        for m in self._BARE_BOXED_PATTERN.finditer(text):
            brace_pos = m.end() - 1  # position of '{'
            depth = 1
            j = brace_pos + 1
            while j < len(text) and depth > 0:
                if text[j] == '\\' and j + 1 < len(text):
                    j += 2
                    continue
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                j += 1
            if depth > 0:
                return True
        return False

    def _get_unclosed_environment(self, text: str) -> Optional[str]:
        """Return the name of the first unclosed \\begin{X} environment, or None."""
        begins = self._BEGIN_ENV_PATTERN.findall(text)
        ends = self._END_ENV_PATTERN.findall(text)
        for env in set(begins):
            if begins.count(env) > ends.count(env):
                return env
        return None

    def _get_unclosed_delimiter_type(self, text: str) -> Optional[str]:
        """Check if text has unclosed LaTeX formula delimiters."""
        counts = self._count_latex_delimiters(text)

        if counts["display_dollar"] % 2 != 0:
            return '$$'
        if counts["bracket_open"] > counts["bracket_close"]:
            return r'\['
        if counts["paren_open"] > counts["paren_close"]:
            return r'\('
        if counts["inline_dollar"] % 2 != 0:
            return '$'
        # Check for unclosed bare \boxed{...}
        if self._is_inside_bare_boxed(text):
            return r'\boxed'
        # Check for unclosed \begin{X} environment
        if self._get_unclosed_environment(text) is not None:
            return r'\begin'
        return None

    def _find_last_standalone_dollar(self, text: str) -> int:
        """Find the last standalone $ delimiter (not part of $$)."""
        without_double = text.replace('$$', '')
        idx = without_double.rfind('$')
        if idx < 0:
            return -1

        # Map position back to original text, accounting for $$ pairs
        original_idx = 0
        remaining = idx
        while remaining > 0:
            if text[original_idx:original_idx + 2] == '$$':
                original_idx += 2
                remaining -= 2
            else:
                original_idx += 1
                remaining -= 1

        # Only return if not escaped
        if original_idx > 0 and text[original_idx - 1] != '\\':
            return original_idx
        return -1

    def _find_formula_boundary_split(self, text: str, delimiter: str) -> Tuple[int, str]:
        """Find a safe split position when dealing with unclosed formulas."""
        # Handle \begin{X} environment
        if delimiter == r'\begin':
            env_name = self._get_unclosed_environment(text)
            if env_name:
                opening = f'\\begin{{{env_name}}}'
                pos = text.rfind(opening)
                if pos > 0:
                    return pos, "char_limit_before_formula"
                if pos == 0:
                    return -1, "formula_at_start"
            return -1, "no_split"

        # Handle bare \boxed{...}
        if delimiter == r'\boxed':
            matches = list(self._BARE_BOXED_PATTERN.finditer(text))
            if matches:
                pos = matches[-1].start()
                if pos > 0:
                    return pos, "char_limit_before_formula"
                if pos == 0:
                    return -1, "formula_at_start"
            return -1, "no_split"

        pos = (
            self._find_last_standalone_dollar(text) if delimiter == '$'
            else text.rfind(delimiter)
        )

        if pos > 0:
            return pos, "char_limit_before_formula"
        if pos == 0:
            return -1, "formula_at_start"
        return -1, "no_split"

    def _find_space_near_limit(self, text: str, limit: int) -> Tuple[int, str]:
        """Find a whitespace position near a given limit."""
        search_start = max(0, limit - self.SPACE_SEARCH_RANGE)

        # First try to find whitespace
        for i in range(limit, search_start, -1):
            if i < len(text) and text[i] in ' \n\t':
                if self._is_decimal_near_position(text, i):
                    continue
                if not self._is_in_latex_formula(text, i):
                    return i, "char_limit_space"

        # Fallback: find punctuation as split point
        for punct in '。！？：；，""")}]':
            pos = text.rfind(punct, search_start, limit)
            if pos >= 0:
                split_pos = pos + 1
                if self._is_decimal_near_position(text, pos):
                    continue
                if not self._is_in_latex_formula(text, split_pos - 1):
                    return split_pos, "char_limit_punctuation"

        return -1, "no_split"

    def _is_decimal_near_position(self, text: str, pos: int) -> bool:
        """Check if position is adjacent to a decimal point."""
        # Check if character before is a decimal point followed by digit
        if pos > 0 and text[pos] == '.' and pos - 1 >= 0 and text[pos - 1].isdigit():
            return True
        # Check if character before position is a decimal point with digit before it
        if pos > 1 and text[pos - 1] == '.' and text[pos - 2].isdigit():
            return True
        return False

    def _would_split_complete_formula(self, text: str, split_pos: int) -> bool:
        """Check if splitting at a position would break a complete formula."""
        for opening, closing, _ in self.DELIMITER_PAIRS:
            last_opening = text.rfind(opening, 0, split_pos)
            if last_opening >= 0 and text.find(closing, split_pos) > split_pos:
                return True

        # Handle inline $ (not $$)
        last_dollar = text.rfind('$', 0, split_pos)
        if last_dollar >= 0 and not text.startswith('$$', last_dollar - 1):
            next_dollar = text.find('$', split_pos)
            if next_dollar > split_pos and not text.startswith('$$', next_dollar - 1):
                return True

        return False

    def _find_latex_closing_delimiter(self, text: str, unclosed_type: str) -> Tuple[int, str]:
        """Find closing delimiter for an unclosed LaTeX formula."""
        # Handle \begin{X} environment
        if unclosed_type == r'\begin':
            env_name = self._get_unclosed_environment(text)
            if env_name:
                closing = f'\\end{{{env_name}}}'
                pos = text.rfind(closing)
                if pos >= 0:
                    return pos + len(closing), "latex_closing"
            return -1, "no_closing"

        # Handle bare \boxed{...}
        if unclosed_type == r'\boxed':
            for m in self._BARE_BOXED_PATTERN.finditer(text):
                brace_pos = m.end() - 1
                depth = 1
                j = brace_pos + 1
                while j < len(text) and depth > 0:
                    if text[j] == '\\' and j + 1 < len(text):
                        j += 2
                        continue
                    if text[j] == '{':
                        depth += 1
                    elif text[j] == '}':
                        depth -= 1
                    j += 1
                if depth == 0:
                    return j, "latex_closing"
            return -1, "no_closing"

        closing_map = {
            '$': ('$', 1),
            '$$': ('$$', 2),
            r'\(': (r'\)', 2),
            r'\[': (r'\]', 2),
        }

        closing_info = closing_map.get(unclosed_type)
        if not closing_info:
            return -1, "no_closing"

        closing, skip = closing_info
        pos = text.rfind(closing)
        if pos < 0:
            return -1, "no_closing"

        if unclosed_type == '$':
            text_before = text[:pos].replace('$$', '')
            if text_before.count('$') % 2 == 1:
                return pos + 1, "latex_closing"
            return -1, "no_closing"

        if unclosed_type == '$$':
            if text[:pos].count('$$') % 2 == 1:
                return pos + 2, "latex_closing"
            return -1, "no_closing"

        if text.rfind(unclosed_type, 0, pos) >= 0:
            return pos + 2, "latex_closing"

        return -1, "no_closing"

    def _is_decimal_point(self, text: str, pos: int) -> bool:
        """Check if position is a decimal point in a number.

        Args:
            text: The text to check
            pos: Position after the potential decimal point (match.end())

        Returns:
            True if the position follows a decimal point with digit before it
        """
        # Check if surrounded by digits (e.g., "18.3")
        if pos - 2 >= 0 and pos < len(text):
            # pos is after the decimal point, so pos-2 is the digit before it
            prev_is_digit = text[pos - 2].isdigit()
            next_is_digit = text[pos].isdigit()
            if prev_is_digit and next_is_digit:
                return True
        # Check if at end with digit before (e.g., "18.")
        if pos == len(text) and pos - 2 >= 0 and text[pos - 2].isdigit():
            return True
        return False

    def _find_safe_split_position(self, text: str) -> Tuple[int, SplitReason]:
        """Find a safe position to split text, respecting LaTeX formula boundaries."""
        unclosed_delimiter = self._get_unclosed_delimiter_type(text)
        has_complete = self._has_complete_formula(text)

        extended_limit = self.max_chars
        if unclosed_delimiter and not has_complete and len(text) >= self.max_chars:
            extended_limit = self.max_chars * 2

        if '$$' in text[:20]:
            logger.debug(
                f"[SentenceBuffer] text_len={len(text)}, unclosed_delimiter={unclosed_delimiter}, "
                f"has_complete={has_complete}, extended_limit={extended_limit}"
            )

        # Try sentence end punctuation — highest priority
        for match in reversed(list(self.SENTENCE_END_PATTERN.finditer(text))):
            pos = match.end()
            matched_char = match.group(1)

            # Skip escaped parenthesis
            if matched_char == ')' and pos > 1 and text[pos - 2] == '\\':
                continue

            # Skip decimal points
            if matched_char == '.' and self._is_decimal_point(text, pos):
                continue

            if not self._is_in_latex_formula(text, pos - 1):
                if has_complete and pos < len(text) and self._would_split_complete_formula(text, pos):
                    continue
                return pos, "sentence_end"

        # Try comma splitting (lower priority)
        if not unclosed_delimiter and len(text) >= self.comma_split_threshold:
            for match in reversed(list(self.COMMA_PATTERN.finditer(text))):
                pos = match.end()
                if not self._is_in_latex_formula(text, pos - 1):
                    return pos, "comma"

        # Character limit forced splitting
        if len(text) >= extended_limit:
            if has_complete and unclosed_delimiter is None:
                potential_split = min(extended_limit, len(text))
                if self._would_split_complete_formula(text, potential_split):
                    return -1, "waiting_for_complete_formula"

            if unclosed_delimiter:
                closing_pos, reason = self._find_latex_closing_delimiter(text, unclosed_delimiter)
                if closing_pos > 0:
                    return closing_pos, reason
                pos, reason = self._find_formula_boundary_split(text, unclosed_delimiter)
                if pos > 0:
                    return pos, reason
                if reason == "formula_at_start":
                    return -1, "unclosed_formula"

            pos, reason = self._find_space_near_limit(text, extended_limit)
            if pos > 0:
                return pos, reason

            return min(extended_limit, len(text)), "char_limit_forced"

        # Buffer not full — if unclosed formula, wait for more tokens
        if unclosed_delimiter:
            # logger.debug(
            #     f"[SentenceBuffer] Not splitting due to unclosed formula: "
            #     f"delimiter={unclosed_delimiter!r}, buffer_len={len(text)}, "
            #     f"buffer_end={repr(text[-50:] if len(text) > 50 else text)}"
            # )
            return -1, "unclosed_formula"

        return -1, "no_split"

    @staticmethod
    def _has_complete_formula(text: str) -> bool:
        """Check if text contains at least one complete LaTeX formula."""
        def has_letter_or_digit(content: str) -> bool:
            return bool(SentenceBuffer.NON_LETTER_DIGIT_PATTERN.search(content))

        for opening, closing, skip in SentenceBuffer.DELIMITER_PAIRS:
            pos = text.find(opening)
            if pos >= 0:
                closing_pos = text.find(closing, pos + skip)
                if closing_pos >= 0 and has_letter_or_digit(text[pos + skip:closing_pos]):
                    return True

        if '$$' not in text:
            pos = text.find('$')
            if pos >= 0:
                next_pos = text.find('$', pos + 1)
                if next_pos >= 0 and has_letter_or_digit(text[pos + 1:next_pos]):
                    return True

        return False

    @staticmethod
    def _is_punctuation_only(text: str) -> bool:
        """Check if text contains only punctuation and whitespace characters."""
        return bool(SentenceBuffer.PUNCTUATION_ONLY_PATTERN.match(text))

    @staticmethod
    def _is_short_prefix_segment(text: str, min_length: int = 5) -> bool:
        """Check if segment is a short prefix like '1.', '2.', 'a.'."""
        if len(text) >= min_length:
            return False
        return bool(SentenceBuffer.SHORT_PREFIX_PATTERN.match(text))

    @staticmethod
    def _is_meaningful_segment(text: str, min_length: int = 8) -> bool:
        """Check if segment is meaningful and worth independent output."""
        if not text:
            return False

        # Check if has letters/Chinese characters and meets minimum length
        if len(text) >= min_length:
            has_letter = bool(re.search(r'[a-zA-Z\u4e00-\u9fff]', text))
            return has_letter

        # Check if is short prefix followed by whitespace only
        if SentenceBuffer.PUNCTUATION_ONLY_EXTENDED_PATTERN.match(text):
            return False

        # Check if is short prefix pattern like "1." followed by whitespace
        if SentenceBuffer.SHORT_PREFIX_WITH_NEWLINE_PATTERN.match(text):
            return False

        return False

    @staticmethod
    def _merge_punctuation_segments(segments: list[str]) -> list[str]:
        """Merge punctuation-only and short prefix segments into adjacent segments."""
        if not segments:
            return []

        n = len(segments)
        is_punct = [SentenceBuffer._is_punctuation_only(s) for s in segments]
        is_prefix = [SentenceBuffer._is_short_prefix_segment(s) for s in segments]

        result = []
        i = 0

        while i < n:
            if is_punct[i]:
                if result:
                    result[-1] += segments[i]
                else:
                    # Collect consecutive punctuation at start
                    j = i
                    while j < n and is_punct[j]:
                        j += 1
                    if j < n:
                        result.append(''.join(segments[i:j]) + segments[j])
                    else:
                        result.extend(segments[i:j])
                    i = j
            elif is_prefix[i]:
                i = SentenceBuffer._handle_prefix_segment(segments, result, i, is_prefix)
            else:
                result.append(segments[i])
                i += 1

        return result

    @staticmethod
    def _handle_prefix_segment(segments: list[str], result: list[str], i: int, is_prefix: list[bool]) -> int:
        """Handle a short prefix segment and return next index."""
        n = len(segments)
        if i + 1 < n:
            prefix = segments[i]
            j = i + 1
            # Include following newline if present
            if j < n and segments[j] == '\n':
                prefix += '\n'
                j += 1
            # Include consecutive prefix segments
            while j < n and is_prefix[j]:
                prefix += segments[j]
                j += 1
            if j < n:
                result.append(prefix + segments[j])
                return j + 1
            else:
                result.append(prefix)
                return j
        else:
            result.append(segments[i])
            return i + 1

    @staticmethod
    def _has_latex_formula(text: str) -> bool:
        """Detect if text contains LaTeX formulas (closed or unclosed)."""
        if SentenceBuffer.CLOSED_FORMULA_PATTERN.search(text):
            return True

        text_clean = SentenceBuffer.ESCAPED_DOLLAR_PATTERN.sub('', text)

        if text_clean.count('$$') % 2 != 0:
            return True

        remaining = text_clean.replace('$$', '')
        if remaining.count('$') % 2 != 0:
            return True

        for opening, closing in [(r'\(', r'\)'), (r'\[', r'\]')]:
            if text.count(opening) != text.count(closing):
                return True

        # Detect bare \boxed{...} without delimiters
        if SentenceBuffer._BARE_BOXED_PATTERN.search(text):
            return True

        return False

    def _flush_buffer(self) -> str:
        """Clear and return the current buffer content."""
        content = self.buffer
        self.buffer = ""
        self.last_flush_time = time.time()
        self.is_flushed = True
        return content

    def _apply_pending_merge(self, content: str) -> str:
        """Apply pending merge to content and reset."""
        if self._pending_merge:
            content = self._pending_merge + content
            self._pending_merge = ""
        return content

    def add(self, token: str) -> Optional[str]:
        """Add a token to the buffer and return a segment if ready to flush."""
        if not self.buffer:
            self.last_flush_time = time.time()

        self.buffer += token
        self.is_flushed = False

        split_pos, split_reason = self._find_safe_split_position(self.buffer)

        if split_pos > 0 and '$$' in self.buffer[:10]:
            logger.warning(
                f"[SentenceBuffer.add] SPLITTING formula! buffer_len={len(self.buffer)}, "
                f"split_pos={split_pos}, reason={split_reason}"
            )

        unclosed = self._get_unclosed_delimiter_type(self.buffer)
        if split_pos > 0 and unclosed:
            logger.debug(
                f"[SentenceBuffer] Splitting with unclosed delimiter: "
                f"buffer_len={len(self.buffer)}, split_pos={split_pos}, "
                f"reason={split_reason}, unclosed={unclosed!r}"
            )

        if split_pos > 0:
            segment = self.buffer[:split_pos]
            self.buffer = self.buffer[split_pos:]
            self.last_flush_time = time.time()
            self.is_flushed = True

            if not self._is_meaningful_segment(segment):
                self._pending_merge += segment
                return None

            return self._apply_pending_merge(segment)

        elapsed = time.time() - self.last_flush_time

        if self.buffer and elapsed > self.max_wait_seconds and not unclosed:
            content = self._apply_pending_merge(self._flush_buffer())
            return content

        if self.buffer and elapsed > self.max_wait_seconds and unclosed:
            logger.debug(
                f"[SentenceBuffer] Timeout bypassed: elapsed={elapsed:.2f}s, "
                f"unclosed_delimiter={unclosed!r}, buffer_len={len(self.buffer)}"
            )

        return None

    async def flush(self, is_final: bool = False) -> Optional[SentenceSegment]:
        """Flush remaining buffer content."""
        if not self.buffer and not self._pending_merge:
            return None

        content = self._apply_pending_merge(self._flush_buffer())
        has_formula = self._has_latex_formula(content)

        return SentenceSegment(
            content=content,
            has_formula=has_formula,
            is_final=is_final
        )

    def has_pending_content(self) -> bool:
        """Check if buffer has pending content."""
        return bool(self.buffer)

    def get_buffer_length(self) -> int:
        """Get current buffer length."""
        return len(self.buffer)


def has_latex_formula(text: str) -> bool:
    """Detect if text contains LaTeX formulas (closed or unclosed)."""
    return SentenceBuffer._has_latex_formula(text)
