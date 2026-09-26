import torch

from shortcuts import t


def dot_product_attention(inputs: torch.Tensor) -> torch.Tensor:
    attention_scores = torch.empty(inputs.shape[0], inputs.shape[0])

    for i, x_i in enumerate(inputs):
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
            [1.6, 2.6, 3.6],
            [1.7, 2.7, 3.7],
            [1.8, 2.8, 3.8],
        ]
    )

    wq = t([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]])
    wk = t([[0.2, 0.2], [0.2, 0.2], [0.2, 0.2]])
    wv = t([[0.3, 0.3], [0.3, 0.3], [0.3, 0.3]])

    q = inputs @ wq
    k = inputs @ wk
    v = inputs @ wv

    # print(q)
    # print(k)
    # print(v)

     

    print("\n\nq")
    print(q)

    print("\n\nk.T")
    print(k.T)

    att_scores = q @ k.T

    print("\n\natt_scores")
    print(att_scores)

    att_weights = torch.softmax(att_scores, dim=-1)

    print("\n\natt_weights")
    print(att_weights)

    context =  att_weights @ v

    print("\n\ncontext")
    print(context)

if __name__ == "__main__":
    attention_main()
