from transformers import pipeline,AutoModel,AutoTokenizer
fill_mask=pipeline("fill-mask",model="bert-base-uncased")
sentences=[
    "The capital of France is [MASK].",
    "i drink a cup of [MASK] every morning",
    "the sun rises in the [MASK]"
]
print("BERT FILL-MASK PREDICTIONS")
for sentence in sentences:
    results=fill_mask(sentence,top_k=3)
    print("\nSentence:",sentence)
    for result in results:
        print(
            "Prediction",result["token_str"],
            "| Score:",round(result["score"],4)
        )
generator=pipeline("text-generation",model="gpt2")
prompts=[
    "Artificial intelligence helps people",
    "A good customer support team",
    "MAchine learning is useful beacause"
]
print("\n GPT-2 TEXT GENERATION")
for prompt in prompts:
    result=generator(
        prompt,
        max_new_tokens=30,
        temperature=0.7,
        do_sample=True,
        pad_token_id=generator.tokenizer.eos_token_id
    )
    print("\nPrompt:",prompt)
    print("Generated text:",result[0]["generated_text"])

bert_model=AutoModel.from_pretrained("bert-base-uncased")
gpt_model=AutoModel.from_pretrained("gpt2")
bert_parameters=sum(
    p.numel() for p in bert_model.parameters()
)
gpt_parameters=sum(
    p.numel() for p in gpt_model.parameters()
)
print("\nModel Parameters")
print("BERT parameter:",bert_parameters)
print("GPT-2 parameter:",gpt_parameters)