import requests
from transformers import pipeline

API_KEY = "96ff5ff01b2441f3979d29cc6e65ee02"

company = input("Enter Company Name: ")

url = (
    f"https://newsapi.org/v2/everything?"
    f"q={company}&"
    f"language=en&"
    f"pageSize=5&"
    f"apiKey={API_KEY}"
)

response = requests.get(url)

data = response.json()

articles = data.get("articles", [])

classifier = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

print("\nNews Sentiment Analysis\n")

for article in articles:

    title = article["title"]

    result = classifier(title)[0]

    print("\nHeadline:")
    print(title)

    print("Sentiment:", result["label"])
    print("Confidence:", round(result["score"], 4))