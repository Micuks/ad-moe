import torch
from torch import nn
from torch.functional import F


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


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


class AutoEncoderExpert(nn.Module):
    def __init__(self, input_size, rep_dim=32, hidden_dim=64) -> None:
        super(AutoEncoderExpert, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rep_dim),
            # nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(rep_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_size),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


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


class GatingNetwork(nn.Module):
    def __init__(
        self,
        input_size,
        eps=1e-4,
        num_tasks=38,
        embedding_dim=32,
        inner_dim=128,
        num_experts=8,
    ):
        super(GatingNetwork, self).__init__()
        # self.task_embedding = nn.Embedding(num_tasks, embedding_dim)
        self.gate = nn.Sequential(
            nn.Linear(input_size, inner_dim),
            # nn.Linear(input_size + embedding_dim, inner_dim),
            # nn.LayerNorm(num_experts),  # To avoid NaN
            nn.BatchNorm1d(inner_dim, eps=eps),  # To avoid NaN
            nn.ReLU(),
            nn.Linear(inner_dim, num_experts),
            nn.Softmax(dim=1),
        )

        self.gate.apply(init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # task_embed = self.task_embedding(task_id)
        # combined_input = torch.cat((x, task_embed), dim=1)
        # return self.gate(combined_input)

        return self.gate(x)


class _MOEAnomalyDetection(nn.Module):
    def __init__(self, input_size, num_experts: int, rep_dim=32, expert_type="mlp"):
        super(_MOEAnomalyDetection, self).__init__()
        self.expert_type = expert_type

        self.gating_network = GatingNetwork(input_size, num_experts)
        if expert_type == "mlp":
            self.experts = nn.ModuleList(
                [MLP(input_size, rep_dim=rep_dim) for _ in range(num_experts)]
            )
            self.score_transform = nn.Linear(rep_dim, 1)
            self.score_transform.apply(init_weights)
        elif expert_type == "autoencoder":
            self.experts = nn.ModuleList(
                [
                    AutoEncoderExpert(input_size=input_size, rep_dim=rep_dim)
                    for _ in range(num_experts)
                ]
            )
            # self.score_transform = nn.Linear(input_size, 1)

        self.gating_network.apply(init_weights)
        [expert.apply(init_weights) for expert in self.experts]

    def forward(self, x):
        gate_weights = self.gating_network(x)
        scores = [expert(x) for expert in self.experts]
        if self.expert_type == "mlp":
            scores = [self.score_transform(score) for score in scores]
            scores = torch.stack(scores, dim=1).squeeze(-1)
            # print(gate_weights.shape, scores.shape)
            final_score = torch.sum(gate_weights * scores, dim=1)
        elif self.expert_type == "autoencoder":
            scores = torch.stack(scores, dim=1)
            # print(gate_weights.shape, scores.shape)
            final_score = torch.sum(gate_weights.unsqueeze(-1) * scores, dim=1)

        return final_score
