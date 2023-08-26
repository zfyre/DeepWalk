import torch
import torch.nn as nn

""" Vanilla Model Using the O(N) Softmax """

class Vanilla(nn.Module):
    def __init__(self, size_vertex, D, device=None):
        super(Vanilla, self).__init__()
        self.encoder = nn.parameter.Parameter(torch.rand((size_vertex, D)), requires_grad = True).to(device) # phi
        self.decoder = nn.parameter.Parameter(torch.rand((D, size_vertex)), requires_grad = True).to(device) # phi_bar
        self.device = device
        self.size_vertex = size_vertex

        def skipgram(randwalk, window, learning_rate):
            for j in range(len(randwalk)):
                # print("j: ",j)
                for k in range(max(0,j-window) , min(j+window, len(randwalk))):

                    """ Calculating the Conditional Probabilities P(u_k | phi(v_j)) by first getting the weights outputed by the encoder->decoder layers"""
                    mask = torch.zeros(self.size_vertex).to(self.device)                    # ->> O(N) steps but torch so easy
                    mask[randwalk[j]] = 1                                         # ->> O(1) steps
                    Out =  self.forward(mask)

                    """ Taking SoftMax of the values of the output of model to get Probabilities"""
                    P = nn.Softmax(Out).dim                                       # ->> O(N) steps
                    # print(P)
                    P = P.to(self.device)

                    """ Calculate the loss : Log Loss (Logistic Loss) """
                    loss = -torch.log(P[randwalk[k]]) 
                    loss.backward()

                    for param in self.parameters():
                        param.data.sub_(learning_rate*param.grad)
                        param.grad.data.zero_()

        self.skipgram = skipgram

    def forward(self, mask):
        encoded_space = torch.matmul(mask, self.encoder).to(self.device)
        decoded_space = torch.matmul(encoded_space, self.decoder).to(self.device)
        return decoded_space.to(self.device)
    


""" Heirarchical Model Using the O(log(N)) Softmax """

def build(size_vertex):
    leaf = []
    def tree_construction(tl, tr, v):
        if(tl == tr):
            leaf.append(v)
            return
        
        tm = (tl+tr)>>1
        tree_construction(tl,tm,2*v)
        tree_construction(tm+1,tr,2*v+1)

    tree_construction(1,size_vertex+1,1)
    return leaf

def path_root_v(vertex):
    path = []
    while(vertex!=1):
        path.append(vertex)
        vertex //= 2
    path=path[::-1]
    return path

class HierarchialModel(nn.Module):
    def __init__(self, size_vertex, D, device=None) -> None:
        super(HierarchialModel, self).__init__()
        self.encoder = nn.parameter.Parameter(torch.rand((size_vertex, D)), requires_grad=True).to(device) # phi
        # These are the logistic parameters for each node which outputs the left as well as the right child for a particular node.
        """( # of Binary Tree nodes, D)"""
        self.logistic_probability_matrix = nn.parameter.Parameter(torch.rand(4*size_vertex,D),requires_grad=True).to(device) 
        self.size_vertex = size_vertex
        self.leaf_nodes = build(self.size_vertex)
        self.device = device

        def H_skipgram(randwalk, window, learning_rate):
            for j in range(len(randwalk)):
                for k in range(max(0,j-window) , min(j+window, len(randwalk))):

                    prob = self.forward(randwalk[k],randwalk[j])
                    loss = -torch.log(prob)
                    loss.backward()

                    for param in self.parameters():
                        param.data.sub_(learning_rate*param.grad)
                        param.grad.data.zero_()

        self.H_skipgram = H_skipgram
        """
            To Implement...

            1. Find the vector for the nodes in the path from root to v_i, i.e. [0, 1, 0, 1, 1 ...] '0' for left and vice-versa
            2. start from root and find the logistic weights of curr_node & apply sigmoid to get probabilities : logistic_probability_matrix[curr_node] -> sigmoid() -> probabilities b/w [0.0, 1.0]
            3. '1' & '2' can be done simultaneously...
            4. Now sequentially multiply the probabilities we are getting.
            5. Apply the final loss as the Binary Cross entropy for the node v_i.

            DOUBT: Should we try to accumulate the losses for all the stages? But we do'nt do such kind of thing when dealing with the NNs in general, we only try to find a loss using the final layer output.

        """
         
        """ Our Original Graph is 0 indexed and Tree is 1 Based!!"""

    # Give's The P(v_i|PHI(v_j))
    def forward(self, v_i, v_j):
        mask = torch.zeros(self.size_vertex)
        mask[v_j] = 1
        # Hypothesis / PHI(v_j)
        h = torch.matmul(mask,self.encoder)

        new_node = self.leaf_nodes[v_i]  # V_I's value in the tree
        path = path_root_v(new_node)
        p = torch.tensor([1.0])
        for j in range(0,len(path)-1):
            mult = -1
            if(path[j+1]==2*path[j]): # Left child
                mult = 1
            p = p*torch.sigmoid(mult*torch.matmul(self.logistic_probability_matrix[path[j]], h)).to(self.device)
        
        return p

            



