import streamlit as st
import torch
import torch.nn as nn
import pickle

st.set_page_config(
    page_title="Mini GPT Content Generator",
    page_icon="🤖"
)

# Load vocabulary
with open("word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)

with open("idx2word.pkl", "rb") as f:
    idx2word = pickle.load(f)

vocab_size = len(word2idx)

# Positional Embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(
            x.size(1),
            device=x.device
        ).unsqueeze(0)

        return self.embedding(positions)

# Masked Attention
class MaskedAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            d_model,
            heads,
            batch_first=True
        )

    def forward(self, x):

        seq_len = x.size(1)

        mask = torch.triu(
            torch.ones(seq_len, seq_len),
            diagonal=1
        ).bool().to(x.device)

        output, _ = self.attn(
            x, x, x,
            attn_mask=mask
        )

        return output

# GPT Block
class GPTBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.attn = MaskedAttention(d_model, 4)

        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model)
        )

        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):

        x = self.norm1(
            x + self.attn(x)
        )

        x = self.norm2(
            x + self.ffn(x)
        )

        return x

# Mini GPT
class MiniGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        max_len=64,
        d_model=128,
        layers=4
    ):
        super().__init__()

        self.token_embed = nn.Embedding(
            vocab_size,
            d_model
        )

        self.pos_embed = PositionalEmbedding(
            max_len,
            d_model
        )

        self.blocks = nn.ModuleList([
            GPTBlock(d_model)
            for _ in range(layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(self, x):

        x = (
            self.token_embed(x)
            + self.pos_embed(x)
        )

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return self.fc(x)

# Load model
model = MiniGPT(vocab_size)

model.load_state_dict(
    torch.load(
        "mini_gpt.pth",
        map_location="cpu"
    )
)

model.eval()

# Tokenizer
def tokenize(text):

    return [
        word2idx.get(
            word,
            word2idx["<UNK>"]
        )
        for word in text.split()
    ]

# Generation
def generate_text(
    prompt,
    max_tokens=50
):

    tokens = tokenize(prompt)

    tokens = torch.tensor(
        [tokens]
    )

    with torch.no_grad():

        for _ in range(max_tokens):

            output = model(tokens)

            probs = torch.softmax(
                output[:, -1, :] / 0.8,
                dim=-1
            )

            next_token = torch.multinomial(
                probs,
                1
            )

            tokens = torch.cat(
                [tokens, next_token],
                dim=1
            )

    words = []

    for idx in tokens[0]:

        words.append(
            idx2word.get(
                idx.item(),
                "<UNK>"
            )
        )

    return " ".join(words)

# UI
st.title("🤖 Mini GPT Content Generator")

st.write(
    "Generate content using your trained Mini GPT model."
)

prompt = st.text_area(
    "Enter Prompt",
    "Artificial Intelligence"
)

if st.button("Generate"):

    result = generate_text(prompt)

    st.subheader("Generated Content")

    st.write(result)