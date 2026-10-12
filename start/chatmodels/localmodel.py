from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm=HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        do_sample=False,
    ),
)

chat_model=ChatHuggingFace(llm=llm)

result=chat_model.invoke("i'm the last surviver on earth now what should i do.....??")

print(result)

print(f"\nGeneration time: {time.time() - start:.2f} seconds")