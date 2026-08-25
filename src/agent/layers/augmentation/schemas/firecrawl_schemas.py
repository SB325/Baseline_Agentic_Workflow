from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, HttpUrl


# --- Enums ---
class FormatEnum(str, Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    RAW_HTML = "rawHtml"
    SCREENSHOT = "screenshot"
    LINKS = "links"
    SUMMARY = "summary"
    CHANGE_TRACKING = "changeTracking"
    BRANDING = "branding"
    JSON = "json"
    QUERY = "query"
    AUDIO = "audio"


class ParserEnum(str, Enum):
    PDF = "pdf"


class ActionTypeEnum(str, Enum):
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    SCROLL = "scroll"
    SCRAPE = "scrape"
    CLICK = "click"
    WRITE = "write"
    PRESS = "press"
    EXECUTE_JAVASCRIPT = "executeJavascript"
    GENERATE_PDF = "generatePDF"


# --- Nested Objects ---
class JsonOptions(BaseModel):
    prompt: Optional[str] = None
    schema_: Optional[Dict[str, Any]] = Field(default=None, alias="schema")

    class Config:
        populate_by_name = True
        extra = "forbid"


class QueryOptions(BaseModel):
    prompt: str = Field(..., max_length=10000)
    mode: Literal["directQuote", "freeform"] = "freeform"

    class Config:
        extra = "forbid"


class Viewport(BaseModel):
    width: float
    height: float

    class Config:
        extra = "forbid"


class ScreenshotOptions(BaseModel):
    fullPage: Optional[bool] = None
    quality: Optional[float] = None
    viewport: Optional[Viewport] = None

    class Config:
        extra = "forbid"


class PdfOptions(BaseModel):
    maxPages: Optional[int] = Field(default=None, ge=1, le=10000)

    class Config:
        extra = "forbid"


class Action(BaseModel):
    type: ActionTypeEnum
    selector: Optional[str] = None
    milliseconds: Optional[float] = None
    text: Optional[str] = None
    key: Optional[str] = None
    direction: Optional[Literal["up", "down"]] = None
    script: Optional[str] = None
    fullPage: Optional[bool] = None

    class Config:
        extra = "forbid"


class Location(BaseModel):
    country: Optional[str] = None
    languages: Optional[List[str]] = None

    class Config:
        extra = "forbid"


class Profile(BaseModel):
    name: str
    saveChanges: Optional[bool] = None

    class Config:
        extra = "forbid"


# --- Main Function Schema ---
class FirecrawlScrapeArgs(BaseModel):
    url: HttpUrl
    formats: Optional[List[FormatEnum]] = None
    jsonOptions: Optional[JsonOptions] = None
    queryOptions: Optional[QueryOptions] = None
    screenshotOptions: Optional[ScreenshotOptions] = None
    parsers: Optional[List[ParserEnum]] = None
    pdfOptions: Optional[PdfOptions] = None
    onlyMainContent: Optional[bool] = None
    includeTags: Optional[List[str]] = None
    excludeTags: Optional[List[str]] = None
    waitFor: Optional[float] = None
    actions: Optional[List[Action]] = None
    mobile: Optional[bool] = None
    skipTlsVerification: Optional[bool] = None
    removeBase64Images: Optional[bool] = None
    location: Optional[Location] = None
    storeInCache: Optional[bool] = None
    zeroDataRetention: Optional[bool] = None
    maxAge: Optional[float] = None
    lockdown: Optional[bool] = None
    proxy: Optional[Literal["basic", "stealth", "enhanced", "auto"]] = None
    profile: Optional[Profile] = None

    class Config:
        extra = "forbid"

# Schema for validation
TOOL_SCHEMAS: Dict[str, type[BaseModel]] = {
    "brave_web_search": FirecrawlScrapeArgs,
}