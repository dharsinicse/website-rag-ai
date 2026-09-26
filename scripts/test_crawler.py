from ingestion.crawler.web_crawler import WebCrawler


crawler = WebCrawler(
    max_pages=5,
    max_depth=1,
)

pages = crawler.crawl(
    "https://example.com"
)

print(f"\nCrawled {len(pages)} pages\n")

for page in pages:
    print("=" * 60)
    print(f"URL: {page.url}")
    print(f"Title: {page.title}")
    print(f"Depth: {page.depth}")
    print(f"Text length: {len(page.text)}")