import spacy
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k
import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import random
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
import deepspeed.comm as dist
import argparse
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# data parallel을 위해 data를 나눠주는 sampler 추가
from torch.utils.data.distributed import DistributedSampler

from torch.utils.data import DataLoader
## 소스 코드
## https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice/blob/master/code_practices/Attention_is_All_You_Need_Tutorial_(German_English).ipynb

def add_argument():
    parser = argparse.ArgumentParser(description="CIFAR")

    # For train.
    parser.add_argument(
        "-e",
        "--epochs",
        default=30,
        type=int,
        help="number of total epochs (default: 30)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=2000,
        help="output logging information at a given interval",
    )

    # For mixed precision training.
    # parser.add_argument(
    #     "--dtype",
    #     default="fp16",
    #     type=str,
    #     choices=["bf16", "fp16", "fp32"],
    #     help="Datatype used for training",
    # )

    # For ZeRO Optimization.
    parser.add_argument(
        "--stage",
        default=0,
        type=int,
        choices=[0, 1, 2, 3],
        help="Datatype used for training",
    )

    # For MoE (Mixture of Experts).
    parser.add_argument(
        "--moe",
        default=False,
        action="store_true",
        help="use deepspeed mixture of experts (moe)",
    )
    parser.add_argument(
        "--ep-world-size", default=2, type=int, help="(moe) expert parallel world size"
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        nargs="+",
        default=[
            1,
        ],
        help="number of experts list, MoE related.",
    )
    parser.add_argument(
        "--mlp-type",
        type=str,
        default="standard",
        help="Only applicable when num-experts > 1, accepts [standard, residual]",
    )
    parser.add_argument(
        "--top-k", default=1, type=int, help="(moe) gating top 1 and 2 supported"
    )
    parser.add_argument(
        "--min-capacity",
        default=0,
        type=int,
        help="(moe) minimum capacity of an expert regardless of the capacity_factor",
    )
    parser.add_argument(
        "--noisy-gate-policy",
        default=None,
        type=str,
        help="(moe) noisy gating (only supported with top-1). Valid values are None, RSample, and Jitter",
    )
    parser.add_argument(
        "--moe-param-group",
        default=False,
        action="store_true",
        help="(moe) create separate moe param groups, required when using ZeRO w. MoE",
    )

    # Include DeepSpeed configuration arguments.
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args


def create_moe_param_groups(model):
    """Create separate parameter groups for each expert."""
    parameters = {"params": [p for p in model.parameters()], "name": "parameters"}
    return split_params_into_different_moe_groups_for_optimizer(parameters)


def get_ds_config(args):
    """Get the DeepSpeed configuration dictionary."""
    ds_config = {
        "train_batch_size": 16,
        "steps_per_print": 2000,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.001,
                "betas": [0.8, 0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7,
                # hansol
                "torch_adam": True,
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 0.001,
                "warmup_num_steps": 1000,
            },
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        # "bf16": {"enabled": args.dtype == "bf16"},
        # "fp16": {
        #     "enabled": args.dtype == "fp16",
        #     "fp16_master_weights_and_grads": False,
        #     "loss_scale": 0,
        #     "loss_scale_window": 500,
        #     "hysteresis": 2,
        #     "min_loss_scale": 1,
        #     "initial_scale_power": 15,
        # },
        "wall_clock_breakdown": False,
        "zero_optimization": {
            "stage": args.stage,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "allgather_bucket_size": 50000000,
            "reduce_bucket_size": 50000000,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "cpu_offload": False,
        },
        # hansol
        "comms_logger": {
            "enabled": True,
            "verbose": False,
            "prof_all": True,
            "debug": False,
            },
        "model_parallel_size": 2,
        "expert_parallel_size": 2,
        "data_parallel_size": 2,

    }
    return ds_config


class MultiHeadAttentionLayer(nn.Module):
    #local_device
    def __init__(self, hidden_dim, n_heads, dropout_ratio):
        super().__init__()

        assert hidden_dim % n_heads == 0

        self.hidden_dim = hidden_dim # 임베딩 차원
        self.n_heads = n_heads # 헤드(head)의 개수: 서로 다른 어텐션(attention) 컨셉의 수
        self.head_dim = hidden_dim // n_heads # 각 헤드(head)에서의 임베딩 차원

        self.fc_q = nn.Linear(hidden_dim, hidden_dim) # Query 값에 적용될 FC 레이어
        self.fc_k = nn.Linear(hidden_dim, hidden_dim) # Key 값에 적용될 FC 레이어
        self.fc_v = nn.Linear(hidden_dim, hidden_dim) # Value 값에 적용될 FC 레이어

        self.fc_o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_ratio)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))#.to(local_device)

    def forward(self, query, key, value, mask = None):

        batch_size = query.shape[0]

        # query: [batch_size, query_len, hidden_dim]
        # key: [batch_size, key_len, hidden_dim]
        # value: [batch_size, value_len, hidden_dim]
        # dtype = query.dtype
        # #hansol
        # self.fc_q.weight.data = self.fc_q.weight.data.to(dtype)
        # self.fc_k.weight.data = self.fc_k.weight.data.to(dtype)
        # self.fc_v.weight.data = self.fc_v.weight.data.to(dtype)
        # self.fc_o.weight.data = self.fc_o.weight.data.to(dtype)
        self.scale = self.scale.to(query.device)
        
        Q = self.fc_q(query).to(query.device)
        K = self.fc_k(key).to(query.device)
        V = self.fc_v(value).to(query.device)

        # Q: [batch_size, query_len, hidden_dim]
        # K: [batch_size, key_len, hidden_dim]
        # V: [batch_size, value_len, hidden_dim]

        # hidden_dim → n_heads X head_dim 형태로 변형
        # n_heads(h)개의 서로 다른 어텐션(attention) 컨셉을 학습하도록 유도
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q: [batch_size, n_heads, query_len, head_dim]
        # K: [batch_size, n_heads, key_len, head_dim]
        # V: [batch_size, n_heads, value_len, head_dim]

        # Attention Energy 계산
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy: [batch_size, n_heads, query_len, key_len]

        # 마스크(mask)를 사용하는 경우
        if mask is not None:
            # 마스크(mask) 값이 0인 부분을 -1e10으로 채우기
            energy = energy.masked_fill(mask==0, -1e10)

        # 어텐션(attention) 스코어 계산: 각 단어에 대한 확률 값
        attention = torch.softmax(energy, dim=-1)

        # attention: [batch_size, n_heads, query_len, key_len]

        # 여기에서 Scaled Dot-Product Attention을 계산
        x = torch.matmul(self.dropout(attention), V)

        # x: [batch_size, n_heads, query_len, head_dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x: [batch_size, query_len, n_heads, head_dim]

        x = x.view(batch_size, -1, self.hidden_dim)

        # x: [batch_size, query_len, hidden_dim]

        x = self.fc_o(x)

        # x: [batch_size, query_len, hidden_dim]

        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hidden_dim, pf_dim, dropout_ratio, args):
        super().__init__()
        self.moe = args.moe
        if self.moe:
            # fc_1 = nn.Linear(hidden_dim, pf_dim)
            # dropout = nn.Dropout(dropout_ratio)
            # fc_3 = nn.Linear(pf_dim, hidden_dim)
            self.moe_layer_list = []
            for n_e in args.num_experts:
                # Create moe layers based on the number of experts.
                self.moe_layer_list.append(
                    deepspeed.moe.layer.MoE(
                        hidden_size=hidden_dim,
                        expert=nn.Sequential(
                            nn.Linear(hidden_dim, pf_dim),
                            nn.ReLU(),
                            nn.Dropout(dropout_ratio),
                            nn.Linear(pf_dim, hidden_dim)
                        ),
                        num_experts=n_e,
                        ep_size=args.ep_world_size,
                        use_residual=args.mlp_type == "residual",
                        k=args.top_k,
                        min_capacity=args.min_capacity,
                        noisy_gate_policy=args.noisy_gate_policy,
                        # ep_size=4
                    )
                )
            self.moe_layer_list = nn.ModuleList(self.moe_layer_list)
        else:
            self.fc_1 = nn.Linear(hidden_dim, pf_dim)
            self.fc_2 = nn.Linear(pf_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):

        # x: [batch_size, seq_len, hidden_dim]
        
        if self.moe:
            for layer in self.moe_layer_list:
                x, _, _ = layer(x)
            
            

        # 원래 아래 하나만
        else :
            x = self.dropout(torch.relu(self.fc_1(x)))

            # x: [batch_size, seq_len, pf_dim]

            x = self.fc_2(x)

        # x: [batch_size, seq_len, hidden_dim]

        return x


class EncoderLayer(nn.Module):
    #local_device
    def __init__(self, hidden_dim, n_heads, pf_dim, dropout_ratio, args):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio)#, local_device
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hidden_dim, pf_dim, dropout_ratio, args)
        self.dropout = nn.Dropout(dropout_ratio)

    # 하나의 임베딩이 복제되어 Query, Key, Value로 입력되는 방식
    def forward(self, src, src_mask):

        # src: [batch_size, src_len, hidden_dim]
        # src_mask: [batch_size, src_len]

        # self attention
        # 필요한 경우 마스크(mask) 행렬을 이용하여 어텐션(attention)할 단어를 조절 가능
        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # src: [batch_size, src_len, hidden_dim]

        # position-wise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src: [batch_size, src_len, hidden_dim]

        return src

class Encoder(nn.Module):
    #local_device
    def __init__(self, input_dim, hidden_dim, n_layers, n_heads, pf_dim, dropout_ratio, arg, max_length=100):
        super().__init__()

        #self.device = local_device

        self.tok_embedding = nn.Embedding(input_dim, hidden_dim)
        self.pos_embedding = nn.Embedding(max_length, hidden_dim)

        self.layers = nn.ModuleList([EncoderLayer(hidden_dim, n_heads, pf_dim, dropout_ratio, args) for _ in range(n_layers)])
        # droput_out 다음 local_device
        self.dropout = nn.Dropout(dropout_ratio)

        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))#.to(local_device)

    def forward(self, src, src_mask):

        # src: [batch_size, src_len]
        # src_mask: [batch_size, src_len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(src.device)

        # pos: [batch_size, src_len]

        # 소스 문장의 임베딩과 위치 임베딩을 더한 것을 사용
        # print("here",src.dtype)
        # print(src.device)
        # print(pos.device)
        # print(pos.dtype)
        src = self.dropout((self.tok_embedding(src).to(src.device) * self.scale.to(src.device)) + self.pos_embedding(pos.to(src.device)))

        # src: [batch_size, src_len, hidden_dim]

        # 모든 인코더 레이어를 차례대로 거치면서 순전파(forward) 수행
        for layer in self.layers:
            src = layer(src, src_mask)

        # src: [batch_size, src_len, hidden_dim]

        return src # 마지막 레이어의 출력을 반환

class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads, pf_dim, dropout_ratio, args): # dropout_ratio 다음 local_device
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio) #local_device
        self.encoder_attention = MultiHeadAttentionLayer(hidden_dim, n_heads, dropout_ratio) #local_device
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hidden_dim, pf_dim, dropout_ratio, args)
        self.dropout = nn.Dropout(dropout_ratio)

    # 인코더의 출력 값(enc_src)을 어텐션(attention)하는 구조
    def forward(self, trg, enc_src, trg_mask, src_mask):

        # trg: [batch_size, trg_len, hidden_dim]
        # enc_src: [batch_size, src_len, hidden_dim]
        # trg_mask: [batch_size, trg_len]
        # src_mask: [batch_size, src_len]

        # self attention
        # 자기 자신에 대하여 어텐션(attention)
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        # trg: [batch_size, trg_len, hidden_dim]

        # encoder attention
        # 디코더의 쿼리(Query)를 이용해 인코더를 어텐션(attention)
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)

        # dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        # trg: [batch_size, trg_len, hidden_dim]

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        # trg: [batch_size, trg_len, hidden_dim]
        # attention: [batch_size, n_heads, trg_len, src_len]

        return trg, attention

class Decoder(nn.Module): # dropout 다음 local_device
    def __init__(self, output_dim, hidden_dim, n_layers, n_heads, pf_dim, dropout_ratio, args, max_length=100):
        super().__init__()

        #self.device = local_device

        self.tok_embedding = nn.Embedding(output_dim, hidden_dim)
        self.pos_embedding = nn.Embedding(max_length, hidden_dim)
        #local_device
        self.layers = nn.ModuleList([DecoderLayer(hidden_dim, n_heads, pf_dim, dropout_ratio, args) for _ in range(n_layers)])

        self.fc_out = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout_ratio)

        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))#.to(local_device)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        # trg: [batch_size, trg_len]
        # enc_src: [batch_size, src_len, hidden_dim]
        # trg_mask: [batch_size, trg_len]
        # src_mask: [batch_size, src_len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(trg.device)

        # pos: [batch_size, trg_len]

        trg = self.dropout((self.tok_embedding(trg) * self.scale.to(trg.device)) + self.pos_embedding(pos))

        # trg: [batch_size, trg_len, hidden_dim]

        for layer in self.layers:
            # 소스 마스크와 타겟 마스크 모두 사용
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        # trg: [batch_size, trg_len, hidden_dim]
        # attention: [batch_size, n_heads, trg_len, src_len]

        output = self.fc_out(trg)

        # output: [batch_size, trg_len, output_dim]

        return output, attention

class Transformer(nn.Module):
    #local_device
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, args):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        #self.device = local_device

    # 소스 문장의  토큰에 대하여 마스크(mask) 값을 0으로 설정
    def make_src_mask(self, src):

        # src: [batch_size, src_len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask: [batch_size, 1, 1, src_len]

        return src_mask

    # 타겟 문장에서 각 단어는 다음 단어가 무엇인지 알 수 없도록(이전 단어만 보도록) 만들기 위해 마스크를 사용
    def make_trg_mask(self, trg):

        # trg: [batch_size, trg_len]

        """ (마스크 예시)
        1 0 0 0 0
        1 1 0 0 0
        1 1 1 0 0
        1 1 1 0 0
        1 1 1 0 0
        """
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        # trg_pad_mask: [batch_size, 1, 1, trg_len]

        trg_len = trg.shape[1]

        """ (마스크 예시)
        1 0 0 0 0
        1 1 0 0 0
        1 1 1 0 0
        1 1 1 1 0
        1 1 1 1 1
        """
        # trg_len) 다음에 ,device = self.device
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len),device = trg.device)).bool()

        # trg_sub_mask: [trg_len, trg_len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask: [batch_size, 1, trg_len, trg_len]

        return trg_mask

    def forward(self, src, trg):

        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]

        src_mask = self.make_src_mask(src).to(src.device)
        trg_mask = self.make_trg_mask(trg).to(trg.device)
        # print(src.device)
        # print(trg.device)

        # src_mask: [batch_size, 1, 1, src_len]
        # trg_mask: [batch_size, 1, trg_len, trg_len]

        enc_src = self.encoder(src, src_mask)

        # enc_src: [batch_size, src_len, hidden_dim]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        # output: [batch_size, trg_len, output_dim]
        # attention: [batch_size, n_heads, trg_len, src_len]

        return output, attention


# 독일어(Deutsch) 문장을 토큰화 하는 함수 (순서를 뒤집지 않음)
def tokenize_de(text, space_de):
    return [token.text for token in spacy_de.tokenizer(text)]

# 영어(English) 문장을 토큰화 하는 함수
def tokenize_en(text, spacy_en):
    return [token.text for token in spacy_en.tokenizer(text)]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(f'The model has {count_parameters(model):,} trainable parameters')

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


# 모델 학습(train) 함수
# target_dtype
def train(model, iterator, optimizer, criterion, clip, local_device):
    model.train() # 학습 모드
    epoch_loss = 0

    # 전체 학습 데이터를 확인하며
    # print(iterator)
    for i, batch in enumerate(iterator):
        # print(batch)
        # print("Iteration : ",i, "Local device : ", local_device)
        # print("Iteration : ",i, "Local device : ", local_device, " Data : ", batch)
        print("Iteration : ", i, "Local device : ", local_device)
        src = batch['src'].to(local_device)
        trg = batch['trg'].to(local_device)
        # print(src.dtype)
        # print(trg.dtype)
        # if target_dtype != None:
        #     src = src.to(target_dtype)
        #     trg = trg.to(target_dtype)
        # print(src.dtype)
        # print(trg.dtype)
        optimizer.zero_grad()

        # 출력 단어의 마지막 인덱스()는 제외
        # 입력을 할 때는 부터 시작하도록 처리
        output, _ = model(src, trg[:,:-1])

        # output: [배치 크기, trg_len - 1, output_dim]
        # trg: [배치 크기, trg_len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        # 출력 단어의 인덱스 0()은 제외
        trg = trg[:,1:].contiguous().view(-1)

        # output: [배치 크기 * trg_len - 1, output_dim]
        # trg: [배치 크기 * trg len - 1]

        # 모델의 출력 결과와 타겟 문장을 비교하여 손실 계산
        loss = criterion(output, trg)
        #loss.backward() # 기울기(gradient) 계산
        model.backward(loss)
        # 기울기(gradient) clipping 진행
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # 파라미터 업데이트
        #optimizer.step()
        model.step()

        # 전체 손실 값 계산
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# 모델 평가(evaluate) 함수
def evaluate(model, iterator, criterion, local_device):
    model.eval() # 평가 모드
    epoch_loss = 0

    with torch.no_grad():
        # 전체 평가 데이터를 확인하며
        for i, batch in enumerate(iterator):
            src = batch['src'].to(local_device)
            trg = batch['trg'].to(local_device)

            # 출력 단어의 마지막 인덱스()는 제외
            # 입력을 할 때는 부터 시작하도록 처리
            output, _ = model(src, trg[:,:-1])

            # output: [배치 크기, trg_len - 1, output_dim]
            # trg: [배치 크기, trg_len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            # 출력 단어의 인덱스 0()은 제외
            trg = trg[:,1:].contiguous().view(-1)

            # output: [배치 크기 * trg_len - 1, output_dim]
            # trg: [배치 크기 * trg len - 1]

            # 모델의 출력 결과와 타겟 문장을 비교하여 손실 계산
            loss = criterion(output, trg)

            # 전체 손실 값 계산
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

class TorchTextDataset(Dataset):
    def __init__(self, torchtext_dataset):
        self.dataset = torchtext_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        return {
            'src': data.src,
            'trg': data.trg
        }

def main(args):
    deepspeed.init_distributed()

    spacy_en = spacy.load("en_core_web_sm") # 영어 토큰화(tokenization)
    spacy_de = spacy.load('de_core_news_sm') # 독일어 토큰화(tokenization)

    # tokenized = spacy_en.tokenizer("I am a graduate student.")
    # 독일어(Deutsch) 문장을 토큰화 하는 함수 (순서를 뒤집지 않음)
    def tokenize_de(text):
        return [token.text for token in spacy_de.tokenizer(text)]

    # 영어(English) 문장을 토큰화 하는 함수
    def tokenize_en(text):
        return [token.text for token in spacy_en.tokenizer(text)]

    SRC = Field(tokenize=tokenize_de, init_token="", eos_token="", lower=True, batch_first=True)
    TRG = Field(tokenize=tokenize_en, init_token="", eos_token="", lower=True, batch_first=True)

    if dist.get_rank() != 0:
        # Might be downloading cifar data, let rank 0 download first.
        dist.barrier()


    train_dataset, valid_dataset, test_dataset = Multi30k.splits(exts=(".de", ".en"), fields=(SRC, TRG))

    if dist.get_rank() == 0:
            # Cifar data is downloaded, indicate other ranks can proceed.
            dist.barrier()

    SRC.build_vocab(train_dataset, min_freq=2)
    TRG.build_vocab(train_dataset, min_freq=2)

    BATCH_SIZE = 128

    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    HIDDEN_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    def collate_fn(batch):
        src_batch, trg_batch = [], []
        for sample in batch:
            src_batch.append(torch.tensor([SRC.vocab.stoi[token] for token in sample['src']]))
            trg_batch.append(torch.tensor([TRG.vocab.stoi[token] for token in sample['trg']]))

        src_batch = pad_sequence(src_batch, padding_value=SRC_PAD_IDX, batch_first=True)
        trg_batch = pad_sequence(trg_batch, padding_value=TRG_PAD_IDX, batch_first=True)

        return {'src': src_batch, 'trg': trg_batch}

    # 인코더(encoder)와 디코더(decoder) 객체 선언
    enc = Encoder(INPUT_DIM, HIDDEN_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, args)
    dec = Decoder(OUTPUT_DIM, HIDDEN_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, args)

    # TRG_PAD_IDX 다음 local_device
    model = Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, args)#.to(local_device)

    # Get the local device name (str) and local rank (int).

    # Get list of parameters that require gradients.
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    # 일반적인 데이터 로더(data loader)의 iterator와 유사하게 사용 가능
    # train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    #     (train_dataset, valid_dataset, test_dataset),
    #     batch_size=BATCH_SIZE)
        # device=local_device)
         # train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


  
    train_dataset = TorchTextDataset(train_dataset)
    valid_dataset = TorchTextDataset(valid_dataset) 
    valid_iterator = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)

    ds_config = get_ds_config(args) 
    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=parameters,
        training_data=train_dataset,
        config=ds_config,
        collate_fn=collate_fn,
    )

    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, sampler=train_sampler, pin_memory=True)
    # print("trainloader: dtype : ", trainloader)

    local_device = get_accelerator().device_name(model_engine.local_rank)
    local_rank = model_engine.local_rank
    # model = Transformer(args, local_device).to(local_device)


    # If using MoE, create separate param groups for each expert.
    if args.moe_param_group:
        parameters = create_moe_param_groups(model)


    # For float32, target_dtype will be None so no datatype conversion needed.
    # target_dtype = None
    # if model_engine.bfloat16_enabled():
    #     target_dtype = torch.bfloat16
    # elif model_engine.fp16_enabled():
    #     target_dtype = torch.half

    # Transformer 객체 선언
    #model = Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, local_device).to(device)

    model.apply(initialize_weights)

    # Adam optimizer로 학습 최적화
    LEARNING_RATE = 0.0005
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 뒷 부분의 패딩(padding)에 대해서는 값 무시
    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

    # N_EPOCHS = 10
    CLIP = 1
    best_valid_loss = float('inf')

    for epoch in range(args.epochs):
        start_time = time.time() # 시작 시간 기록
        #target_dtype
        print("start\n")
        train_sampler.set_epoch(epoch)
        print("len : ", len(trainloader))
        train_loss = train(model_engine, trainloader, optimizer, criterion, CLIP, local_device)
        valid_loss = evaluate(model_engine, valid_iterator, criterion, local_device)

        end_time = time.time() # 종료 시간 기록
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'transformer_german_to_english.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}')
        print(f'\tValidation Loss: {valid_loss:.3f} | Validation PPL: {math.exp(valid_loss):.3f}')
    
        dist.log_summary()
    print("Finished Training")

if __name__ == "__main__":
    args = add_argument()
    main(args)