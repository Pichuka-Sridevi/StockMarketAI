from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

result = classifier(
    "Reliance Industries reported strong quarterly profits."
)

print(result)