from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser


class RobotsChecker:
    def __init__(
        self,
        user_agent: str = "WebsiteRAGBot/1.0",
    ):
        self.user_agent = user_agent
        self.parsers: dict[str, RobotFileParser] = {}

    def _get_parser(self, url: str) -> RobotFileParser:
        parsed = urlparse(url)

        base_url = f"{parsed.scheme}://{parsed.netloc}"

        if base_url not in self.parsers:
            robots_url = f"{base_url}/robots.txt"

            parser = RobotFileParser()
            parser.set_url(robots_url)

            try:
                parser.read()
            except Exception:
                # If robots.txt cannot be retrieved,
                # don't crash the crawler.
                pass

            self.parsers[base_url] = parser

        return self.parsers[base_url]

    def is_allowed(self, url: str) -> bool:
        parser = self._get_parser(url)

        return parser.can_fetch(
            self.user_agent,
            url,
        )