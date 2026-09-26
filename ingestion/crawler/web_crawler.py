from dataclasses import dataclass
from urllib.parse import urljoin, urlparse, urldefrag
import time

import requests
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from ingestion.crawler.robots import RobotsChecker
from ingestion.crawler.sitemap import SitemapParser

@dataclass
class CrawledPage:
    url: str
    title: str
    text: str
    depth: int


class WebCrawler:
    def __init__(
        self,
        max_pages: int = 50,
        max_depth: int = 2,
        timeout: int = 10,
    ):
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.timeout = timeout

        self.visited_urls: set[str] = set()
        self.pages: list[CrawledPage] = []

        self.robots_checker = RobotsChecker()
        self.sitemap_parser = SitemapParser()

        self.request_delay = 1.0

    def normalize_url(self, url: str) -> str:
        """Normalize a URL for consistent crawling and deduplication."""

        url, _ = urldefrag(url)

        parsed = urlparse(url)

        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()

        path = parsed.path

        # Treat /about/ and /about as the same page.
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")

        return parsed._replace(
            scheme=scheme,
            netloc=netloc,
            path=path,
        ).geturl()

    def is_valid_url(self, url: str) -> bool:
        """Check whether a URL is suitable for crawling."""

        parsed = urlparse(url)

        if parsed.scheme not in {"http", "https"}:
            return False

        if not parsed.netloc:
            return False

        ignored_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".webp",
            ".svg",
            ".ico",
            ".mp4",
            ".mp3",
            ".avi",
            ".mov",
            ".zip",
            ".rar",
            ".7z",
            ".exe",
            ".css",
            ".js",
            ".xml",
            ".json",
            ".pdf",
        }

        path = parsed.path.lower()

        return not any(
            path.endswith(extension)
            for extension in ignored_extensions
        )

    def is_same_domain(
        self,
        url: str,
        base_domain: str,
    ) -> bool:
        """Check whether a URL belongs to the same domain."""

        parsed = urlparse(url)

        return parsed.netloc.lower() == base_domain.lower()


    @retry(
        retry=retry_if_exception_type(
            requests.RequestException
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(
            multiplier=1,
            min=1,
            max=8,
        ),
        reraise=True,
    )
    
    def fetch_page(self, url: str) -> str:
        """Download a web page."""

        time.sleep(self.request_delay)

        response = requests.get(
            url,
            timeout=self.timeout,
            headers={
                "User-Agent": "WebsiteRAGBot/1.0"
            },
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type",
            ""
        ).lower()

        if "text/html" not in content_type:
            raise requests.RequestException(
                f"Unsupported content type: {content_type}"
            )

        return response.text

    def extract_page(
        self,
        html: str,
        url: str,
        depth: int,
    ) -> CrawledPage:
        """Extract title and readable text from HTML."""

        soup = BeautifulSoup(html, "lxml")

        for element in soup(
            [
                "script",
                "style",
                "noscript",
                "svg",
                "nav",
                "footer",
                "header",
                "aside",
                "form",
            ]
        ):
            element.decompose()

        title = ""

        if soup.title:
            title = soup.title.get_text(
                " ",
                strip=True,
            )

        content = (
            soup.find("main")
            or soup.find("article")
            or soup.body
            or soup
        )

        text = soup.get_text(
            separator="\n",
            strip=True,
        )

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        cleaned_text = "\n".join(lines)

        return CrawledPage(
            url=url,
            title=title,
            text=text,
            depth=depth,
        )

    def extract_links(
        self,
        html: str,
        current_url: str,
    ) -> list[str]:
        """Extract links from a web page."""

        soup = BeautifulSoup(html, "lxml")

        links = []

        for anchor in soup.find_all("a", href=True):
            absolute_url = urljoin(
                current_url,
                anchor["href"],
            )

            normalized_url = self.normalize_url(absolute_url)

            if self.is_valid_url(normalized_url):
                links.append(normalized_url)

        return links

    def crawl(self, start_url: str) -> list[CrawledPage]:
        """Recursively crawl a website."""

        start_url = self.normalize_url(start_url)

        if not self.is_valid_url(start_url):
            raise ValueError(
                "start_url must be a valid HTTP/HTTPS webpage URL"
            )

        parsed_start_url = urlparse(start_url)

        if parsed_start_url.scheme not in {
            "http",
            "https",
        }:
            raise ValueError(
                "start_url must use http or https"
            )

        base_domain = parsed_start_url.netloc

        queue: list[tuple[str, int]] = [
            (start_url, 0)
        ]

        sitemap_urls = self.sitemap_parser.fetch_urls(
            start_url
        )

        for sitemap_url in sitemap_urls:
            normalized_url = self.normalize_url(
                sitemap_url
            )

            if (
                self.is_valid_url(normalized_url)
                and self.is_same_domain(
                    normalized_url,
                    base_domain,
                )
            ):
                queue.append(
                    (normalized_url, 1)
                )

        while queue and len(self.pages) < self.max_pages:
            current_url, depth = queue.pop(0)

            if current_url in self.visited_urls:
                continue

            if depth > self.max_depth:
                continue

            if not self.is_same_domain(
                current_url,
                base_domain,
            ):
                continue

            if not self.robots_checker.is_allowed(current_url):
                print(
                    f"Blocked by robots.txt: {current_url}"
                )
                continue

            self.visited_urls.add(current_url)

            try:
                html = self.fetch_page(current_url)

            except requests.RequestException as error:
                print(
                    f"Failed to fetch {current_url}: {error}"
                )
                continue

            page = self.extract_page(
                html,
                current_url,
                depth,
            )

            self.pages.append(page)

            if depth < self.max_depth:
                links = self.extract_links(
                    html,
                    current_url,
                )

                for link in links:
                    if link not in self.visited_urls:
                        if self.is_same_domain(
                            link,
                            base_domain,
                        ):
                            queue.append(
                                (link, depth + 1)
                            )

        return self.pages