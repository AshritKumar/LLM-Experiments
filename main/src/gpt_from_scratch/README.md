* This containes from scratch implementation of a GPT model
* Builds up to the final trainsformer architecture iteratively.
* [gpt_model1.py](gpt_model1.py) and [gpt_model1.ipynb](gpt_model1.ipynb) explore a smimple model with an Embedding layer, GELU non linearity and final unembedding layer
* [gpt_model2.py](gpt_model2.py) and [gpt_model2.ipynb](gpt_model2.ipynb) adds - Positional embeddings and a Layer Noram
* [gpt_model3.py](gpt_model3.py) and [gpt_model3.ipynb](gpt_model3.ipynb) - Adds a single head causal self attention block
    * More on attention [attention_ex.ipynb](attention_ex.ipynb)
* [gpt_model4.py](gpt_model4.py) and [gpt_model4.ipynb](gpt_model4.ipynb) - Adds a transformer block
  * LN_Attn -> Attn -> += RS -> LN_MLP -> MLP_L1 -> GELU -> MLP_L2 -> += RS
