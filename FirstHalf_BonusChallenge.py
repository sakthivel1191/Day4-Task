from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained("xlm-roberta-base")
sentence="வணக்கம், நான் தமிழ் பேசுகிறேன்."
result=tokenizer(sentence)
print("Tamil Sentence",sentence)
print("Tokens",tokenizer.convert_ids_to_tokens(result["input_ids"]))
print("Input IDs",result["input_ids"])
print("Attention Mask:",result["attention_mask"])