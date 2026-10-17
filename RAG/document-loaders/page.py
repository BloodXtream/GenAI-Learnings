from langchain_community.document_loaders import WebBaseLoader

url = "https://www.apple.com/in/shop/buy-mac/macbook-pro/14-inch-m5"
data = WebBaseLoader(url)
docs = data.load()
print(docs)
print(len(docs))