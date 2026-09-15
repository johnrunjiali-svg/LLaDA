"""Stand-ins for the iLLaDA tokenizer and model, so mdm_vim.py runs without weights.

FakeTokenizer  word pieces carrying their leading space (like BPE's 'Ġ'), long words
               split into subwords, and iLLaDA's special ids (bos 0, eos 2, mask 5).
FakeModel      has a canned answer per prompt and behaves like a masked diffusion model:
               more confident the more neighbours are unmasked, confused by wrongly
               committed neighbours, tempted to repeat neighbouring tokens, confident
               about trailing EOS, and at unmasked positions it mostly copies its input.

Both work with the real tokenizer too:  FakeModel(AutoTokenizer.from_pretrained(...)).
"""
import re
from types import SimpleNamespace

import torch

SPECIALS = ['<|startoftext|>', '<|pad|>', '<|endoftext|>', '<|user|>', '<|assistant|>', '<|mdm_mask|>']
ANSWERS = [
    (('capital', 'france'), 'The capital of France is Paris. It is also the largest city in the country.'),
    (('diffusion',), 'A masked diffusion model starts from an all-mask answer and unmasks the tokens it is most confident about, a few at a time.'),
    (('joke',), 'Why did the transformer cross the road? To attend to the other side.'),
]
DEFAULT_ANSWER = 'I am only a fake model, so this answer is made up. Try asking about France, diffusion, or a joke.'
FILLER = ' the a of and to in is it that was for on with as by at from this be are or not'
PIECE = re.compile(r'<\|[a-z_]+\|>| ?[A-Za-z]+| ?[0-9]| ?[^\sA-Za-z0-9]|\s')


class FakeTokenizer:
    bos_token_id, eos_token_id, mask_token_id = 0, 2, 5

    def __init__(self, capacity=4096):
        self.capacity = capacity
        self.id2piece = list(SPECIALS)
        self.piece2id = {p: i for i, p in enumerate(SPECIALS)}
        for _, text in ANSWERS:
            self.encode(text)
        self.encode(DEFAULT_ANSWER + FILLER)

    def __len__(self):
        return self.capacity

    def _id(self, piece):
        if piece not in self.piece2id:
            if len(self.id2piece) == self.capacity:
                raise ValueError('fake vocab is full')
            self.piece2id[piece] = len(self.id2piece)
            self.id2piece.append(piece)
        return self.piece2id[piece]

    def encode(self, text, add_special_tokens=False):
        ids = [self.bos_token_id] if add_special_tokens else []
        for w in PIECE.findall(text):
            subwords = [w] if len(w) <= 7 else [w[:5]] + [w[j:j + 4] for j in range(5, len(w), 4)]
            ids += [self._id(s) for s in subwords]
        return ids

    def __call__(self, text):
        return {'input_ids': self.encode(text)}  # the chat template already starts with <|startoftext|>

    def decode(self, ids, **kwargs):
        return ''.join(self.id2piece[i] if i < len(self.id2piece) else f'<unused{i}>' for i in ids)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        text = '<|startoftext|>' + ''.join(f'<|{m["role"]}|>\n{m["content"]}\n' for m in messages)
        if add_generation_prompt:
            text += '<|assistant|>\n'
        return self.encode(text) if tokenize else text


class FakeModel:
    def __init__(self, tokenizer, mask_id=5):
        self.tokenizer, self.mask_id = tokenizer, mask_id
        self.vocab_size = len(tokenizer)
        self.device = torch.device('cpu')
        self.eos_id = tokenizer.eos_token_id
        self.filler = [tokenizer.encode(w, add_special_tokens=False)[0] for w in re.findall(r' \w+', FILLER)]
        # The answer starts right after the generation prompt the chat template appends.
        msg = [{'role': 'user', 'content': 'x'}]
        without = tokenizer.apply_chat_template(msg, add_generation_prompt=False, tokenize=False)
        with_gen = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        suffix = with_gen[len(without):] if with_gen.startswith(without) else ''
        self.gen_suffix = tokenizer.encode(suffix, add_special_tokens=False) if suffix else []

    def answer_start(self, ids):
        s = self.gen_suffix
        for j in range(len(ids) - len(s), -1, -1) if s else ():
            if ids[j:j + len(s)] == s:
                return j + len(s)
        return ids.index(self.mask_id) if self.mask_id in ids else len(ids)  # --no-chat: guess

    def target(self, prompt, length):
        prompt = prompt.lower()
        text = next((a for keys, a in ANSWERS if all(k in prompt for k in keys)), DEFAULT_ANSWER)
        ids = self.tokenizer.encode(text, add_special_tokens=False)[:length]
        return ids + [self.eos_id] * (length - len(ids)), len(ids)

    @torch.no_grad()
    def __call__(self, input_ids, **kwargs):
        ids = input_ids[0].tolist()
        start = self.answer_start(ids)
        x = ids[start:]
        target, n_words = self.target(self.tokenizer.decode(ids[:start]), len(x))
        g = torch.Generator().manual_seed(hash(tuple(ids)) % 2**31)  # same input, same output
        logits = torch.randn(len(ids), self.vocab_size, generator=g) - 14.0
        logits[:, len(getattr(self.tokenizer, 'id2piece', ())) or self.vocab_size:] = -30.0  # FakeTokenizer's unused ids
        noise = torch.randn(len(x), 8, generator=g).tolist()
        fill = torch.randint(len(self.filler), (len(x), 2), generator=g).tolist()

        for j in range(start):
            logits[j, ids[j]] = 10.0
        for i, tok in enumerate(x):
            row, n = logits[start + i], noise[i]
            if tok != self.mask_id:  # visible token: mostly copy it, remember what it "should" be
                row[tok] = 9.0
                row[target[i]] = max(row[target[i]].item(), 6.5)
                continue
            neigh = [k for k in range(max(0, i - 3), min(len(x), i + 4)) if k != i]
            known = [k for k in neigh if x[k] != self.mask_id]
            wrong = sum(x[k] != target[k] for k in known)
            base = 2.5 + 4.0 * len(known) / max(1, len(neigh)) - 2.0 * wrong + 0.8 * n[0]
            base += 2.5 if i > n_words else 1.0 if i == 0 else 0.0
            for k, bump in ((i - 1, 1.5), (i + 1, 1.2)):  # repeating a neighbour's token
                if 0 <= k < len(x):
                    row[target[k]] = bump + 0.5 * n[1 + (k > i)]
            for f, nz in zip(fill[i], n[3:5]):
                row[self.filler[f]] = 1.0 + 0.3 * wrong + 0.5 * nz
            row[target[i]] = base
        return SimpleNamespace(logits=logits[None])
