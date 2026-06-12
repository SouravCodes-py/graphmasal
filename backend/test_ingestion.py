from ingestion.loader_factory import LoaderFactory

file_path = "data/Module4_cs part1.pptx"   # change later if testing PDF/DOCX

loader = LoaderFactory.get_loader(file_path)

text = loader.extract_text(file_path)

print("=" * 50)
print(text[:10000])
print("=" * 50)