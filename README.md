# Assignment 1: Masked Multi-Head Attention

## Task
In this assignment, your task is to implement the Masked Multi-Head Attention mechanism as outlined in the "[Attention is All You Need](https://arxiv.org/pdf/1706.03762)" paper.
Specifically, you'll complete the forward function of the `CausalSelfAttention` class located in `assignment1.py`.
This class represents the Masked Multi-Head Attention layer within the Transformer *decoder*, so it's crucial to correctly apply the masking to prevent information leakage from future tokens.
Note that the input sequence `x` is assumed to have no padding tokens.

What you need to do is complete the code within
```python
# --- TODO: start of your code ---

# --- TODO: end of your code ---
```
so that the model can run properly.
Changing the code outside the `TODO` block is not necessary.


## Test your implementation

When you have finished implementing the forward function, you can run the code with
```python
python assignment1.py
```

If you implementation is correct, you will see "The output is correct." printed out on your terminal.
Otherwise, you will encounter assertion errors.

**Notice** that passing the test case does not necessary means that your implementation in entirely right.
Please check the code and read the paper carefully to see whether you have missed some components.


## Submission

You can directly submit the modified `assignment1.py` file to Canvas.
