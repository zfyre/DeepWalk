# DeepWalk

[Deep Walk Paper](https://arxiv.org/abs/1403.6652)

##### Introduction:
	A 2014 paper that is before 'Attention' Paper, hence the NLP methods are based off earlier methods.
	The paper uses the "skip-gram" Model as a base NLP model for predicting the Random walks.


https://en.wikipedia.org/wiki/Zipf%27s_law

##### Hierarchical SoftMax:
	1. It's Basically a way to approximate the soft max behaviour for large datasets.
	2. We can say that it a mixture of NNs and Decision Trees.

	PROCEDURE:
	* start from root and find the logistic weights of curr_node & apply sigmoid to get probabilities : logistic_probability_matrix[curr_node] -> sigmoid() -> probabilities b/w [0.0, 1.0]
	* Now sequentially multiply the probabilities we are getting.
	* Apply the final loss as the Binary Cross entropy for the node v_i.    

```python 
def forward(self, v_i, v_j):
	mask = torch.zeros(self.size_vertex)
	mask[v_j] = 1
	# Hypothesis / PHI(v_j)
	h = torch.matmul(mask,self.encoder)
	new_node = self.leaf_nodes[v_i]  # V_I's value in the tree
	path = path_root_v(new_node)
	p = torch.tensor([1.0])
	for j in range(0,len(path)-1):
		mul = -1
		if(path[j+1]==2*path[j]): # Left child
			mul = 1
		p=p*torch.sigmoid(mul*torch.matmul(self.logistic_probability_mat[path[j]], h))
	return p
```

###### Questions:
	1. Why do we do with that mul ?
	2. Why are we not aggregrating the losses at each level of the left/right movement?
	3. Wht ar ewe not taking any negative samples which finidng the loss?
##### Overall Questions:
	1. How Does the Deep-Walk Work?
	2. What is Hierarchical Softmax and how does it it solves the problem with softmax?
	3. How is Deep-Walk an online learning algorithm?
