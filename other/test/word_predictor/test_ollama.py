import ollama

def complete(text, n_tokens = 2, model = "qwen2.5:3b"):

    resp = ollama.generate(
        model = model,
        prompt = text,
        raw = True,
        options = {
            "num_predict": n_tokens,   # short completions
            "temperature": 0.3,  # more deterministic
            "stop": ["\n"],      # stop at newline for autocomplete feel
        },
    )

    return resp["response"]

print(complete("The quick brown fox"))

N_TOKENS = 8
MODEL = "qwen2.5:3b"
resp = ollama.generate(
    model = MODEL,
    prompt = "the quick brown fox",
    raw = True,
    options = {
        "num_predict": N_TOKENS,   # short completions
        "temperature": 0.3,  # more deterministic
    },
    logprobs = True,
    top_logprobs = 10,
)

for el in resp["logprobs"] : print(el)

logprobs = resp["logprobs"]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

