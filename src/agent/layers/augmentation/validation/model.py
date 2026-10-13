from typing import List, Optional, Union
from pydantic import BaseModel, Field

# ********************** BraveSearch ***************************

class BaseSearchFields(BaseModel):
  query: Optional[str] = None
  country: Optional[str] = None
  search_lang: Optional[str] = None
  count: Optional[int] = None
  spellcheck: Optional[bool] = None


class ImageSearchFields(BaseSearchFields):
  safesearch: Optional[str] = None


class VideoSearchFields(ImageSearchFields):
  ui_lang: Optional[str] = None
  offset: Optional[int] = None
  freshness: Optional[str] = None


class NewsSearchFields(VideoSearchFields):
  extra_snippets: Optional[bool] = None
  goggles: Optional[Union[str, List[str]]] = None


class WebLocalSearchFields(NewsSearchFields):
  text_decorations: Optional[bool] = None
  result_filter: Optional[List[str]] = None
  units: Optional[str] = None
  summary: Optional[bool] = None


class PlaceSearchFields(ImageSearchFields):
  latitide: Optional[float] = None
  longitude: Optional[float] = None
  location: Optional[str] = None
  ui_lang: Optional[str] = None
  units: Optional[str] = None
  geoloc: Optional[str] = None
  api_version: Optional[str] = Field(None, alias="api-version")
  accept: Optional[str] = None
  cache_control: Optional[str] = Field(None, alias="cache-control")
  user_agent: Optional[str] = Field(None, alias="user-agent")


class LlmContextFields(BaseSearchFields):
  maximum_number_of_urls: Optional[int] = None
  maximum_number_of_tokens: Optional[int] = None
  maximum_number_of_snippets: Optional[int] = None
  context_threshold_mode: Optional[str] = None
  maximum_number_of_tokens_per_url: Optional[int] = None
  maximum_number_of_snippets_per_url: Optional[int] = None
  goggles: Optional[Union[str, List[str]]] = None
  freshness: Optional[str] = None
  enable_local: Optional[bool] = None
  enable_source_metadata: Optional[bool] = None
  x_loc_lat: Optional[float] = Field(None, alias="x-loc-lat")
  x_loc_long: Optional[float] = Field(None, alias="x-loc-long")
  x_loc_city: Optional[str] = Field(None, alias="x-loc-city")
  x_loc_state: Optional[str] = Field(None, alias="x-loc-state")
  x_loc_state_name: Optional[str] = Field(None, alias="x-loc-state-name")
  x_loc_country: Optional[str] = Field(None, alias="x-loc-country")
  x_loc_postal_code: Optional[str] = Field(None, alias="x-loc-postal-code")
  api_version: Optional[str] = Field(None, alias="api-version")
  accept: Optional[str] = None
  cache_control: Optional[str] = Field(None, alias="cache-control")
  user_agent: Optional[str] = Field(None, alias="user-agent")


class SummarizerFields(BaseModel):
  key: Optional[str] = None
  entity_info: Optional[bool] = None
  inline_references: Optional[bool] = None


# --- Functions Container Model ---


class Functions(BaseModel):
  brave_web_search: Optional[WebLocalSearchFields] = None
  brave_local_search: Optional[WebLocalSearchFields] = None
  brave_video_search: Optional[VideoSearchFields] = None
  brave_image_search: Optional[ImageSearchFields] = None
  brave_news_search: Optional[NewsSearchFields] = None
  brave_summarizer: Optional[SummarizerFields] = None
  brave_llm_context: Optional[LlmContextFields] = None
  brave_place_search: Optional[PlaceSearchFields] = None


class ConfigModel(BaseModel):
  functions: Functions

# ********************** Firecrawl ***************************

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# --- Individual Function Models ---

class FirecrawlScrape(BaseModel):
    url: str  # Required
    formats: Optional[Any] = None
    jsonOptions: Optional[Any] = None
    queryOptions: Optional[Any] = None
    screenshotOptions: Optional[Any] = None
    parsers: Optional[Any] = None
    pdfOptions: Optional[Any] = None
    onlyMainContent: Optional[Any] = None
    includeTags: Optional[Any] = None
    excludeTags: Optional[Any] = None
    waitFor: Optional[Any] = None
    actions: Optional[Any] = None
    mobile: Optional[Any] = None
    skipTlsVerification: Optional[Any] = None
    removeBase64Images: Optional[Any] = None
    location: Optional[Any] = None
    storeInCache: Optional[Any] = None
    zeroDataRetention: Optional[Any] = None
    maxAge: Optional[Any] = None
    lockdown: Optional[Any] = None
    proxy: Optional[Any] = None
    profile: Optional[Any] = None

class FirecrawlMap(BaseModel):
    url: str  # Required
    search: Optional[Any] = None
    sitemap: Optional[Any] = None
    includeSubdomains: Optional[Any] = None
    limit: Optional[Any] = None
    ignoreQueryParameters: Optional[Any] = None

class FirecrawlSearch(BaseModel):
    query: str  # Required
    limit: Optional[Any] = None
    tbs: Optional[Any] = None
    filter: Optional[Any] = None
    location: Optional[Any] = None
    includeDomains: Optional[Any] = None
    excludeDomains: Optional[Any] = None
    sources: Optional[Any] = None
    scrapeOptions: Optional[Any] = None
    enterprise: Optional[Any] = None

class FirecrawlSearchFeedback(BaseModel):
    searchId: str  # Required
    rating: Any  # Required
    valuableSources: Optional[Any] = None
    missingContent: Optional[Any] = None
    querySuggestions: Optional[Any] = None

class FirecrawlCrawl(BaseModel):
    url: str  # Required
    prompt: Optional[Any] = None
    excludePaths: Optional[Any] = None
    includePaths: Optional[Any] = None
    maxDiscoveryDepth: Optional[Any] = None
    sitemap: Optional[Any] = None
    limit: Optional[Any] = None
    allowExternalLinks: Optional[Any] = None
    allowSubdomains: Optional[Any] = None
    crawlEntireDomain: Optional[Any] = None
    delay: Optional[Any] = None
    maxConcurrency: Optional[Any] = None
    webhook: Optional[Any] = None
    webhookHeaders: Optional[Any] = None
    deduplicateSimilarURLs: Optional[Any] = None
    ignoreQueryParameters: Optional[Any] = None
    scrapeOptions: Optional[Any] = None

class FirecrawlCheckCrawlStatus(BaseModel):
    id: str  # Required

class FirecrawlParse(BaseModel):
    filePath: str  # Required
    contentType: Optional[Any] = None
    formats: Optional[Any] = None
    jsonOptions: Optional[Any] = None
    queryOptions: Optional[Any] = None
    parsers: Optional[Any] = None
    pdfOptions: Optional[Any] = None
    onlyMainContent: Optional[Any] = None
    includeTags: Optional[Any] = None
    excludeTags: Optional[Any] = None
    removeBase64Images: Optional[Any] = None
    skipTlsVerification: Optional[Any] = None
    storeInCache: Optional[Any] = None
    zeroDataRetention: Optional[Any] = None
    maxAge: Optional[Any] = None
    proxy: Optional[Any] = None

class FirecrawlExtract(BaseModel):
    urls: List[str]  # Required (typically a list for extraction)
    prompt: Optional[Any] = None
    schema: Optional[Any] = None
    allowExternalLinks: Optional[Any] = None
    enableWebSearch: Optional[Any] = None
    includeSubdomains: Optional[Any] = None

class FirecrawlAgent(BaseModel):
    prompt: str  # Required
    urls: Optional[Any] = None
    schema: Optional[Any] = None

class FirecrawlAgentStatus(BaseModel):
    id: str  # Required

class FirecrawlBrowserCreate(BaseModel):
    ttl: Optional[Any] = None
    activityTtl: Optional[Any] = None
    streamWebView: Optional[Any] = None
    profile: Optional[Any] = None

class FirecrawlBrowserExecute(BaseModel):
    sessionID: str  # Required
    code: str  # Required
    language: Optional[Any] = None

class FirecrawlBrowserDelete(BaseModel):
    sessionID: str  # Required

class FirecrawlBrowserList(BaseModel):
    status: Optional[Any] = None

class FirecrawlInteract(BaseModel):
    scrapeId: str  # Required
    prompt: Optional[Any] = None
    code: Optional[Any] = None
    language: Optional[Any] = None
    timeout: Optional[Any] = None

class FirecrawlInteractStop(BaseModel):
    scrapeId: str  # Required


# --- Root Container Model ---

class FirecrawlFunctions(BaseModel):
    firecrawl_scrape: Optional[FirecrawlScrape] = None
    firecrawl_map: Optional[FirecrawlMap] = None
    firecrawl_search: Optional[FirecrawlSearch] = None
    firecrawl_search_feedback: Optional[FirecrawlSearchFeedback] = None
    firecrawl_crawl: Optional[FirecrawlCrawl] = None
    firecrawl_check_crawl_status: Optional[FirecrawlCheckCrawlStatus] = None
    firecrawl_parse: Optional[FirecrawlParse] = None
    firecrawl_extract: Optional[FirecrawlExtract] = None
    firecrawl_agent: Optional[FirecrawlAgent] = None
    firecrawl_agent_status: Optional[FirecrawlAgentStatus] = None
    firecrawl_browser_create: Optional[FirecrawlBrowserCreate] = None
    firecrawl_browser_execute: Optional[FirecrawlBrowserExecute] = None
    firecrawl_browser_delete: Optional[FirecrawlBrowserDelete] = None
    firecrawl_browser_list: Optional[FirecrawlBrowserList] = None
    firecrawl_interact: Optional[FirecrawlInteract] = None
    firecrawl_interact_stop: Optional[FirecrawlInteractStop] = None

class RootConfig(BaseModel):
    functions: FirecrawlFunctions