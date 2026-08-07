Small proof of concept (PoC) for a token predictor using Ollama and prompt_toolkit.

# Setup

To run this small example you need the following python packages :
- `prompt_toolkit`
- `ollama`
- `math` (this should be inside the default python installation)

You can easily install everything using `pip`

```
pip install prompt_toolkit ollama
```

## Quick installation (OPTIONAL)

If you want to install everything more quickly you can install `uv` first and then using `uv` to install the rest of the dependencies.

```
pip install uv
uv install prompt_toolkit ollama
```

## Ollama setup
Once you have installed the dependencies, you need to "set up" Ollama (i.e. download the model you want to use and start the ollama server). 

### Model download

Just download any model from the [Ollama model list](https://ollama.com/models) and run the following command to download it:

```
ollama pull <model_name>:<n_parameters>
```

Where `<model_name>` is the name of the model you want to use and `<n_parameters>` is the number of parameters of the model.
For example, if you want to use the 3 billion parameters version of `qwen2.5` you can download it using the following command :

```
ollama pull qwen2.5:3b
```

To run this example locally I suggest a small model (like the already mentioned `qwen2.5:3b`) because it will be faster and will use less memory.
Any model with less than 1 GB of memory should be fine. Of course you can use any model you want, but keep in mind that the bigger the model, the more memory it will use and the slower it will be.

### Starting the Ollama server
Once you have downloaded the model you want to use, you need to start the Ollama server. You can do this by simply running the following command

```
ollama serve
```

You do not need to specify the model name here. 
When you interact with the ollama server from the code you will also specify the model you want to use and the server will automatically load it for you.

Basically the ollama server is a process that run in the background that works as "model manager".

### Side note on Ollama server
If you have install the ollama desktop application the `ollama serve` command will not work because the desktop application already runs a server in the background.
In this case you can just skip the `ollama serve` command and run the code directly.

# Run the code

From a terminal you can run the code using the following command:

```
python token_predictor.py
```

Then you can start typing and the program should predict the next word for you. You can use the `TAB` or the arrow keys to select the word you want to use and then press `ENTER` to add it to your text.
Note that the first prediction COULD be a little bit slow because the model needs to be loaded in memory. After that the predictions should be fast.

## Note on the prediction strategy

For now the prediction strategy is very simple. The program will always suggest ONLY THE NEXT TOKEN (not word) based on the previous tokens.
Note that a token is not necessarily a whole word. It can be a whole word, or it can be just a part of a word. It can also be a punctuation mark or a special character (e.g. chinese symbols).
Why tokens and not words? Because LLMs are trained to predict tokens and not words (it's more efficient and flexible).

For now, since this program was created to be a simple PoC, I chose this strategy because it is the simplest to implement. 
If we continue to develop it for the BCI scenario we could implement more complex strategies (e.g. multiple token, branching paths, etc.) to make the predictions more accurate and useful.
