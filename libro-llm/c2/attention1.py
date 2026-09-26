import torch


def dot_product_attention(inputs: torch.Tensor)->torch.Tensor:
    attention_scores = torch.empty(inputs.shape[0], inputs.shape[0])
    
    
    for i, x_i in  enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attention_scores[i, j] = torch.dot(x_i, x_j)

    return torch.softmax(attention_scores, dim=-1)

def attention_main():
    inputs = torch.Tensor(
        [
            [1.1, 2.1, 3.1],
            [1.2, 2.2, 3.2],
            [1.3, 2.3, 3.3],
            [1.4, 2.4, 3.4],
            [1.5, 2.5, 3.5],
        ]
    )
 
        
    print(dot_product_attention(inputs=inputs))

if __name__ == "__main__":
    attention_main()
