from ingestion.processors.sections import SectionExtractor


extractor = SectionExtractor()

text = """
# Company Overview

Our company builds AI applications.

## Products

We provide RAG systems and AI assistants.

## Pricing

Our plans start at $10 per month.

## Contact

Contact our support team for assistance.
"""

sections = extractor.extract(text)

print(f"Sections found: {len(sections)}")
print()

for index, section in enumerate(sections, start=1):
    print("=" * 60)
    print(f"SECTION {index}")
    print(f"Heading: {section.heading}")
    print(f"Text: {section.text}")