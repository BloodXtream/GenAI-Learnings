from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=1,
    separator="",
)

data = TextLoader("document-loaders/notes.txt")
doc = data.load()

chunks = splitter.split_documents(doc)

print(chunks)
print(len(chunks))

for i in chunks:
    print(i.page_content)
    print()
    print()
# = -  _  ""  ''  