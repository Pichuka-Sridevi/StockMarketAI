import feedparser

feed = feedparser.parse(
    "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en"
)

print("Articles found:", len(feed.entries))

for article in feed.entries[:5]:
    print(article.title)