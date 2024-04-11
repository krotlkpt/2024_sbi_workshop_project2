import torch
from sbi.utils import posterior_nn
from sbi.inference import SNPE
import pickle
from sbi.neural_nets.embedding_nets import CNNEmbedding

# in_size, hid_size, nlayers = 6, 6, 6


def initRNN(in_size: int, hid_size: int, nlayers: int): # -> torch.nn.Model:
    embedding_net = torch.nn.RNN(
        in_size,
        hid_size,
        nlayers    
    )
    return embedding_net


def initCNN(shape):

    embedding_net = CNNEmbedding(
        input_shape=shape,
        in_channels=1,
        out_channels_per_layer=[1,1,1,1,1,1],
        num_conv_layers=6,
        num_linear_layers=6,
        output_dim=8,
        kernel_size=3,
        pool_kernel_size=8
    )

    return embedding_net


def getPost(embedding_net, prior, theta, x):
    neural_posterior = posterior_nn(
        model="maf", embedding_net=embedding_net
    )

    inferer = SNPE(prior=prior, density_estimator=neural_posterior)

    density_estimator = inferer.append_simulations(
        theta, x
    ).train(training_batch_size=256)
    posterior = inferer.build_posterior(density_estimator)

    return posterior


# This is the same prior used to generate the
# 10**5 simulations and summary stats provided.
def prior(
    low=torch.Tensor([0.5, 1e-4, 1e-4, 1e-4, 50.0, 40.0, 1e-4, 35.0]),
    high=torch.Tensor([80.0, 15.0, 0.6, 0.6, 3000.0, 90.0, 0.15, 100.0]),
    seed=None
):

    return (
        torch.distributions.Independent(
            torch.distributions.Uniform(low=low, high=high),
            reinterpreted_batch_ndims=1,
        )
    )


if __name__ == "__main__":
    with open("data/summaries_all.pkl", "rb") as f:
        sums = pickle.load(f)

    with open("data/traces_all.pkl", "rb") as t:
        traces = pickle.load(t)
    print(traces[0].shape)

    with open("data/thetas_all.pkl", "rb") as th:
        thetas = pickle.load(th)

    # rnn = initRNN(8, 6, 6)
    cnn = initCNN((30001,1))
    prior = prior()
    print(prior)

    post = getPost(cnn, prior, thetas[:500], traces[:500])
