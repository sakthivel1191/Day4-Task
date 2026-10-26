from transformers import AutoTokenizer,pipeline
tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased")

sentences=[
    "I like machine learning",
    "Ravi works in chennai",
    "The application is not working"
]
print("Token,Input Ids and Attention Mask")
for sentence in sentences:
    result=tokenizer(sentence)
    print("\nSentence:",sentence)
    print("Tokens:",tokenizer.convert_ids_to_tokens(result["input_ids"]))
    print("Input IDs:",result["input_ids"])
    print("Attention Mask:",result["attention_mask"])

word="antidisestablishmentarianism"
result=tokenizer.tokenize(word)

print("\nRare Word Tokenisation")
print("word:",word)
print("Sub-word pieces:",result)
sentiment=pipeline("sentiment-analysis")
sentences=[
    "I love this application",
    "The service is very good",
    "i am unhappy with the product",
    "The application is terriable",
    "The experience was okay"
]
print("\nSentiment Analysis")
for sentence in sentences:
    result=sentiment(sentence)[0]
    print("Sentence:",sentence)
    print("Label:",result["label"])
    print("Score:",round(result["score"],4))

ner=pipeline("ner",aggregation_strategy="simple")
sentence=(
    "Ravi from YohoTechnologies paid Rs. 45000 in Chennai on 12 March"
)
entities=ner(sentence)
print("\nNamed Entity Recognition")
for entity in entities:
    print(
        "Entity:",entity["word"],
        "| Label:",entity["entity_group"],
        "| Score:",round(entity["score"],4)
    )

classifier=pipeline("zero-shot-classification")
ticket="I was charged twice for my monthly subscription"
categories=[
    "Billing",
    "Technical",
    "Account Submission"
]
result=classifier(ticket,candidate_labels=categories)
print("\nZero-Shot Classification")
print("Ticket:",ticket)
for label,score in zip(result["labels"],result["scores"]):
    print(label,":",round(score,4))