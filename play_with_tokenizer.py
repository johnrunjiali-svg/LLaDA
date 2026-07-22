"""Scratchpad for poking at the LLaDA tokenizer (no GPU / no model weights needed).

Run with:  uv run python play_with_tokenizer.py
"""

import torch
from transformers import AutoTokenizer

MODEL = 'GSAI-ML/LLaDA-8B-Instruct'
MASK_ID = 126336    # <|mdm_mask|>, the token LLaDA diffuses from
EOS_ID = 126081


def show_pieces(tokenizer, text):
    ids = tokenizer(text)['input_ids']
    print(f'\n{text!r}  ->  {len(ids)} tokens')
    for i, t in zip(ids, tokenizer.convert_ids_to_tokens(ids)):
        print(f'  {i:>7}  {t!r}')


def show_special_tokens(tokenizer):
    print('=' * 70)
    print(f'tokenizer   : {type(tokenizer).__name__}')
    print(f'vocab_size  : {tokenizer.vocab_size}   len(tokenizer): {len(tokenizer)}')
    print(f'mask token  : {tokenizer.convert_ids_to_tokens(MASK_ID)!r} (id {MASK_ID})')
    print(f'eos token   : {tokenizer.convert_ids_to_tokens(EOS_ID)!r} (id {EOS_ID})')
    print('specials    :', tokenizer.all_special_tokens)
    print('=' * 70)


def show_chat_template(tokenizer, question='What is diffusion in language models?'):
    m = [{'role': 'user', 'content': question}]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    print('\n--- chat template (raw string) ---')
    print(repr(prompt))
    show_pieces(tokenizer, prompt)


def show_forward_diffusion(tokenizer, text, seed=0):
    """Mirror the forward process from GUIDELINES.md: mask each token w.p. t ~ U(0,1)."""
    torch.manual_seed(seed)
    ids = torch.tensor(tokenizer(text)['input_ids']).unsqueeze(0)

    print('\n--- forward diffusion at a few noise levels ---')
    for t in (0.25, 0.5, 0.75):
        masked = torch.rand(ids.shape) < t
        noisy = torch.where(masked, MASK_ID, ids)
        # decode without skipping specials so the masks stay visible
        print(f'  t={t:.2f}  {tokenizer.decode(noisy[0], skip_special_tokens=False)}')


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    show_special_tokens(tokenizer)
    show_pieces(tokenizer, 'Hello, LLaDA!')
    show_chat_template(tokenizer)
    show_forward_diffusion(tokenizer, 'The quick brown fox jumps over the lazy dog.')


if __name__ == '__main__':
    main()
