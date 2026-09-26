from urllib.parse import urljoin
from xml.etree import ElementTree


class SitemapParser:
    def fetch_urls(self, base_url: str) -> list[str]:
        """Fetch page URLs from sitemap.xml."""

        sitemap_url = urljoin(
            base_url.rstrip("/") + "/",
            "sitemap.xml",
        )

        try:
            import requests

            response = requests.get(
                sitemap_url,
                timeout=10,
                headers={
                    "User-Agent": "WebsiteRAGBot/1.0"
                },
            )

            response.raise_for_status()

        except requests.RequestException:
            return []

        try:
            root = ElementTree.fromstring(response.content)

        except ElementTree.ParseError:
            return []

        urls = []

        for element in root.iter():
            if element.tag.endswith("loc"):
                if element.text:
                    urls.append(element.text.strip())

        return urls