from pydantic import BaseModel, Field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

class FreshnessEnum(str, Enum):
    PAST_DAY = "pd"
    PAST_WEEK = "pw"
    PAST_MONTH = "pm"
    PAST_YEAR = "py"

# Web Search Tool call argument model
class WebSearchArguments(BaseModel):
    query: str = Field(..., description="The main search query to look up on the web.")
    count: Optional[int] = Field(default=5, description="The number of results to return.")
    country: Optional[str] = Field(default='US', description="Search query country, where the results come from. Formatted as 2 character country code.")
    search_lang: Optional[str] = Field(default='en', description="The search language preference. The 2 or more character language code for which the search results are provided.")
    ui_lang: Optional[str] = Field(default='en-US', description="The language of the UI. 2 or more character language code for which search results are provided.")
    offset: Optional[int] = Field(default=0, ge=0, le=9, description="Pagination offset (max 9, default 0).")
    safesearch: Optional[str] = Field(default='moderate', description="Filters search results for adult content. The following values are supported: 'off' - No filtering. 'moderate' - Filter out explicit content. 'strict' - Filter out explicit and suggestive content. The default value is 'moderate.")
    freshness: Optional[FreshnessEnum | str] = Field(default='py', description="Filters search results by when they were discovered. The following values are supported: 'pd' - Discovered within the last 24 hours. 'pw' - Discovered within the last 7 Days. 'pm' - Discovered within the last 31 Days. 'py' - Discovered within the last 365 Days. 'YYYY-MM-DDtoYYYY-MM-DD' - Timeframe is also supported by specifying the date range e.g. 2022-04-01to2022-07-30.")
    text_decorations: Optional[bool] = Field(default=False, description="Whether display strings (e.g. result snippets) should include decoration markers (e.g. highlighting characters).")
    spellcheck: Optional[bool] = Field(default=True, description="Whether to spellcheck the provided query.")
    result_filter: Optional[list] = Field(default=['web','query'], description="Result filter (default ['web', 'query']).")
    goggles: Optional[list | str] = Field(default=5, description="Goggles act as a custom re-ranking on top of Brave's search index. The parameter supports both a url where the Goggle is hosted or the definition of the Goggle. Multiple goggle URLs and/or definitions can beprovided in an array. For more details, refer to the Goggles repository (i.e., https://github.com/brave/goggles-quickstart).")
    units: Optional[str] = Field(default='imperial', description="The measurement units. If not provided, units are derived from search country.")
    extra_snippets: Optional[bool] = Field(default=False, description="A snippet is an excerpt from a page you get as a result of the query, and extra_snippets allow you to get up to 5 additional, alternative excerpts. Only available under Free AI, Base AI, Pro AI, Base Data, Pro Data and Custom plans.")
    summary: Optional[bool] = Field(default=False, description="This parameter enables summary key generation in web search results. This is required for summarizer to be enabled.")

# Schema for validation
TOOL_SCHEMAS: Dict[str, type[BaseModel]] = {
    "brave_web_search": WebSearchArguments,
}