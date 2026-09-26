from ingestion.crawler.sitemap import SitemapParser


parser = SitemapParser()

urls = parser.fetch_urls(
    "https://example.com"
)

print(f"Sitemap URLs found: {len(urls)}")

for url in urls[:10]:
    print(url)