from ingestion.processors.cleaner import DocumentCleaner


cleaner = DocumentCleaner()

raw_text = """
    This    is     a document.


    It contains       unnecessary spaces.

    It also contains
    inconsistent line breaks.
"""

cleaned_text = cleaner.clean(raw_text)

print("CLEANED TEXT")
print("=" * 60)
print(cleaned_text)
print("=" * 60)