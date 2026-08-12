import html
import re


BROKEN_HTML_ENTITY = re.compile(
    r"(?<![&\w])(#(?:x[0-9A-Fa-f]+|\d+)|amp|quot|lt|gt|nbsp|apos|hellip);",
    flags=re.IGNORECASE,
)


def clean_text(text: str) -> str:
    """Repair common AG News text artifacts and normalize whitespace."""
    text = BROKEN_HTML_ENTITY.sub(r"&\1;", str(text))
    text = html.unescape(text)
    text = text.replace("\\", " ")
    text = re.sub(
        r"\s+'(?=(?:s|t|re|ve|ll|d|m)\b)",
        "'",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", text).strip()
