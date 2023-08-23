import torch
import torch.nn as nn


class Vanilla(nn.Module):
    def __init__(self, size_vertex, D, device):
        super(Vanilla, self).__init__()
        self.encoder = nn.parameter.Parameter(torch.rand((size_vertex, D)), requires_grad = True).to(device) # phi
        self.decoder = nn.parameter.Parameter(torch.rand((D, size_vertex)), requires_grad = True).to(device) # phi_bar

    def forward(self, mask, device):
        encoded_space = torch.matmul(mask, self.encoder).to(device)
        decoded_space = torch.matmul(encoded_space, self.decoder).to(device)
        return decoded_space.to(device)
    


def skipgram(model, randwalk, window, size_vertex, learning_rate, device):
    for j in range(len(randwalk)):
        # print("j: ",j)
        for k in range(max(0,j-window) , min(j+window, len(randwalk))):
            """ Calculating the Conditional Probabilities P(u_k | phi(v_j)) """
            mask = torch.zeros(size_vertex).to(device)                       # ->> O(N) steps
            mask[randwalk[j]] = 1 # To tackle with One Based Indexing     # ->> O(1) steps
            Out =  model(mask,device)
            # print("k: ",k)
            """ Taking SoftMax of the values of the output of model to get Probabilities"""
            P = nn.Softmax(Out).dim                                          # ->> O(N) steps
            # print(P)
            P = P.to(device)
            """ Calculate the loss : Log Loss (Logistic Loss) """
            loss = -torch.log(P[randwalk[k]]) ## To tackle with One Based Indexing
            loss.backward()

            for param in model.parameters():
                param.data.sub_(learning_rate*param.grad)
                param.grad.data.zero_()