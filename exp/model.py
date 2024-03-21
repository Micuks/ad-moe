import torch
from torch import nn
from torch.functional import F


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class MLP(nn.Module):
    def __init__(self, x_dim, h_dims=[128, 64], rep_dim=32, bias=False):
        super().__init__()
        self.rep_dim = rep_dim
        neurons = [x_dim, *h_dims]
        layers = [
            linear_bn_leakyReLU(neurons[i - 1], neurons[i], bias=bias)
            for i in range(1, len(neurons))
        ]

        self.hidden = nn.ModuleList(layers)
        self.code = nn.Linear(h_dims[-1], rep_dim, bias=bias)

    def forward(self, x):
        x = x.view(int(x.size(0)), -1)
        for layer in self.hidden:
            x = layer(x)

        return self.code(x)


class linear_bn_leakyReLU(nn.Module):
    def __init__(self, in_features, out_features, bias=False, eps=1e-4):
        super(linear_bn_leakyReLU, self).__init__()

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bn = nn.BatchNorm1d(out_features, eps=eps, affine=bias)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.linear(x)), 0.01)


class PreNet(nn.Module):
    def __init__(self, input_size, act_fun):
        super(PreNet, self).__init__()
        self.feature = nn.Sequential(nn.Linear(input_size, 20), act_fun)
        self.reg = nn.Linear(40, 1)

    def forward(self, X_left, X_right):
        feature_left = self.feature(X_left)
        feature_right = self.feature(X_right)

        feature = torch.cat((feature_left, feature_right), dim=1)
        score = self.reg(feature)

        return score.squeeze()


class GatingNetwork(nn.Module):
    def __init__(self, input_size, num_experts=8):
        super(GatingNetwork, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_size, num_experts),
            nn.LayerNorm(num_experts),  # To avoid NaN
            nn.Softmax(dim=1),
        )

        self.gate.apply(init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        with torch.no_grad():
            pre_softmax_activations = self.gate[:-1](x)
        print(f"Pre-softmax activations: {pre_softmax_activations}")
        return self.gate(x)


class _MOEAnomalyDetection(nn.Module):
    def __init__(self, input_size, num_experts, rep_dim=32):
        super(_MOEAnomalyDetection, self).__init__()

        self.gating_network = GatingNetwork(input_size, num_experts)
        self.experts = nn.ModuleList(
            [MLP(input_size, rep_dim=rep_dim) for _ in range(num_experts)]
        )
        self.score_transform = nn.Linear(rep_dim, 1)

        self.gating_network.apply(init_weights)
        [expert.apply(init_weights) for expert in self.experts]
        self.score_transform.apply(init_weights)

    def forward(self, x):
        gate_weights = self.gating_network(x)

        scores = [self.score_transform(expert(x)) for expert in self.experts]
        scores = torch.stack(scores, dim=1).squeeze(-1)
        # print(gate_weights.shape, scores.shape)
        final_score = torch.sum(gate_weights * scores, dim=1)

        return final_score
