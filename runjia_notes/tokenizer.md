# Tokenizers (BPE), via the LLaDA tokenizer

Play with it: `uv run python play_with_tokenizer.py`

## TL;DR

Nothing mysterious. Given a trained merge table, encoding is:

```
text -> [byte remap] -> [regex chunks] -> [BPE merge inside each chunk] -> ids
```

Each chunk becomes **one or more** tokens. Tokens are just integer ids.

LLaDA: `vocab_size = 126080` (BPE vocab), `len(tokenizer) = 126349` (incl. added
specials). The specials live *above* vocab_size — e.g. `<|mdm_mask|>` = **126336**.

---

## Step 0 — Byte remap (the `Ġ` thing)

Every byte is mapped to a *printable* Unicode char. `space -> 'Ġ'`, `newline -> 'Ċ'`.
So `'ĠWorld'` is literally `" World"`, space included. Nothing was added.

**Not for readability** — that's a side effect (only 68 of 256 bytes actually
move; printable ASCII maps to itself). Two real reasons, below.

### Reason 1: kill OOV / UNK

A vocab is a **fixed, finite list**, frozen before training — the embedding
matrix has one row per entry, so it can never grow. Text containing something
not in the list is **OOV** (out-of-vocabulary); there's no row, so the tokenizer
emits the **UNK** ("unknown") placeholder id.

**UNK is irreversible data loss.** BERT (has `[UNK]`) vs LLaDA (byte-level):

```
'模型'
   BERT  -> ['[UNK]','[UNK]']            decode: '[UNK] [UNK]'      <- destroyed
   LLaDA -> ['æ¨¡åŀĭ']                    decode: '模型'
'robot 🤖 here'
   BERT  -> ['robot','[UNK]','here']     decode: 'robot [UNK] here' <- destroyed
   LLaDA -> ['robot','ĠðŁ¤','ĸ','Ġhere'] decode: 'robot 🤖 here'
```

Naive fix — "add more characters" — fails: Unicode has ~150k chars, GPT-2 noted
you need **~5,000** base chars for decent coverage (~15% of a 32k vocab!) and
it's *still* incomplete (new emoji yearly; invalid UTF-8 isn't a character at all).

**The insight: use bytes, not characters.** Text is already bytes, and there are
exactly **256** byte values.

```
'A' -> 1 byte : [65]          '中' -> 3 bytes: [228,184,173]
'é' -> 2 bytes: [195,169]     '🤖' -> 4 bytes: [240,159,164,150]
```

Take all 256 as base symbols => **OOV is mathematically impossible**, in any
script, forever, including binary garbage. Unfamiliar input just degrades to
single-byte tokens (see 🤖 above) — degraded but **losslessly recoverable**.

> **256 slots, once, for zero OOV forever** — vs 5,000 slots for partial
> coverage. Hence LLaDA's `unk_token is None`: the concept doesn't apply.

Side effect: `'模型' -> 'æ¨¡åŀĭ'` isn't mojibake — 模 is bytes `[230,168,161]`,
and those byte values map to chars `æ`,`¨`,`¡`. Decodes back perfectly.

### Reason 2: the merge file would be unparseable

The merge table is **one merge per line, two symbols separated by a space** —
parsing is just `line.split(" ")` expecting 2 parts. But then a symbol that *is*
a space collides with the separator (**delimiter collision**, same as a comma
inside a CSV field):

```
WITH remap                        WITHOUT remap
  'Ġ Ġ' -> ['Ġ','Ġ']  2 ok          '   ' -> ['','','','']  4  ??
  'Ġ t' -> ['Ġ','t']  2 ok          '  t' -> ['','','t']    3  ??
  'i n' -> ['i','n']  2 ok          'i n' -> ['i','n']      2  ??  <- ambiguous!
```

That last one: is `'i n'` a merge of `i`+`n`, or a symbol containing a space?
Unknowable. And merging space+space is **rank 0**, the most frequent pair in the
corpus — so it's line 1 of the file, not an edge case.

Two ways out: (1) escaping/quoting like CSV, or (2) make the delimiter
impossible in the data. GPT-2 chose (2) — shift the offending bytes up by 256
into a printable range: `0x20 + 0x100 = 0x120 = 'Ġ'`. Then `split(" ")` is
correct by construction, forever.

**Newlines are the worse case**: a space breaks *field* boundaries, a newline
breaks *record* boundaries — one merge would span two lines and the file's
structure collapses entirely. Hence `\n -> 'Ċ'`. (GPT-2's comment: the mapping
"avoids mapping to whitespace/control characters the bpe code barfs on.")

> Caveat: modern `tokenizer.json` stores merges as JSON arrays
> (`[["Ġ","Ġ"],...]`), which handles spaces fine — so today Reason 2 is legacy,
> kept for compat. Reason 1 stands on its own regardless.

> Fun fact: `Ġ` is meaningless — U+0120, a Maltese letter. An arbitrary +256
> offset landed on it. From GPT-2's `bytes_to_unicode()`, now everywhere.

## Step 1 — Regex chunks = WALLS

This does **not** tokenize. It cuts the sentence into chunks that merges may
**never cross**.

```
"Text diffusion is hippopotamus-like."
-> ['Text', 'Ġdiffusion', 'Ġis', 'Ġhippopotamus', '-like', '.']
```

Rules that matter (cl100k-style pattern):

| clause | effect |
|---|---|
| `'(?i:[sdmt]\|ll\|ve\|re)` | contractions split: `don` + `'t` |
| `[^\r\n\p{L}\p{N}]?+\p{L}+` | optional leading space + letters -> **space attaches to the FOLLOWING word** |
| `\p{N}` | **one digit at a time**: `3.14` -> `3`,`.`,`1`,`4` |
| `\s+(?!\S)` | trailing whitespace held back: `"  stop"` -> `Ġ` + `Ġstop` |

**Why walls exist:** without them BPE would blindly fuse `"of the"`, `"World."`,
`"World,"`, `"World!"` — burning the vocab on redundant phrase/punctuation combos
and forcing the model to learn `World.` and `World,` as unrelated symbols.

Verified on LLaDA's 126,080 tokens:
```
tokens containing letter-SPACE-letter : 0
tokens like  word.word                : 0
```
Structurally impossible, not merely rare.

## Step 2 — BPE merge, inside each chunk

Chunk is exploded to single chars, then merged back up. **Lowest merge rank
first** — not longest-match, not left-to-right. Deterministic, greedy, and *not*
guaranteed minimal token count.

```
'Ġdiffusion' -> ['Ġ','d','i','f','f','u','s','i','o','n']
  rank    6: 'on'          rank  115: 'if'      rank   952: 'Ġdiff'
  rank   37: 'ion'         rank  180: 'us'      rank  7190: 'usion'
  rank   43: 'Ġd'          rank  744: 'iff'     rank 34300: 'Ġdiffusion'  -> 1 token
```

Rank = order learned during training = frequency order.

### The key rule

> **A word is ONE token iff that exact string is in the vocab.**
> Subword splitting is the *fallback* for rare words, not the default.

**32,484** LLaDA vocab entries are `Ġ` + a complete 5+ letter word (~25% of vocab).

```
'Ġdiffusion'    in vocab -> True   -> 1 token
'Ġhippopotamus' in vocab -> False  -> ['Ġhipp','opot','amus']
```

Boundaries are contextual, not a property of the word:
```
'LLaDA'   -> ['LL','a','DA']        # different leading byte
' LLaDA'  -> ['ĠL','La','DA']       # => different merge path
```

---

## How the table is trained

No gradients, no NN — just counting:

```
1. vocab = the 256 byte symbols
2. pre-tokenize corpus into chunks (the walls)
3. count adjacent pairs, WITHIN chunks only
4. merge the single most frequent pair; append it to the merge list
5. goto 3        # LLaDA: 125,824 times
```

The ordered merge list *is* the model; inference replays it. First merges are
`('Ġ','Ġ')`, `('Ġ','t')`, `('i','n')`, `('Ġ','a')`, `('h','e')`.

> Fun fact: BPE is a 1994 **compression** algorithm (Philip Gage), repurposed for
> NLP by Sennrich et al. 2015.

## Does it matter? Yes

**Exchange rate between text and compute:**
```
English: 49 chars ->  9 tokens  (5.44 chars/token)
Chinese: 13 chars -> 10 tokens  (1.30 chars/token)
Code   : 43 chars -> 13 tokens  (3.31 chars/token)
```
Chinese ~4x more tokens/char => 4x cost, 4x latency, 4x less fits in context.
A property of the training mix, not the language.

- **Digits split individually** (`12345` -> `1,2,3,4,5`) — deliberate; fusing
  multi-digit numbers measurably hurts arithmetic.
- **Glitch tokens**: `SolidGoldMagikarp` — tokens in GPT-2/3's vocab that barely
  appeared in training kept garbage embeddings -> bizarre outputs.
- **For LLaDA**: sets diffusion granularity. One mask = one token.
  `Ġun`/`mask`/`ing` = 3 independently recoverable slots; `Ġdiffusion` = 1
  all-or-nothing slot.

## Gotchas

- `convert_ids_to_tokens` = raw vocab lookup, still byte-remapped (`'ĠWorld'`).
  `decode` = runs the ByteLevel decoder (`' World'`). Different layers, both right.
  Bridge: `convert_tokens_to_string`.
- `"World"` and `" World"` are **different ids**. Masking a token masks its
  leading space too.
- **Roundtrip is not always exact.** Byte-level BPE is lossless, but the
  *normalizer* before it isn't. LLaDA uses `NFC()`:
  ```
  'cafe' + combining-acute  ->  'café'   # 5 codepoints -> 4, silently
  ```
  Returns a canonically-equivalent string, not a byte-identical one.
- Check `tok.backend_tokenizer.normalizer` before trusting roundtrips on another
  model — BERT lowercases + strips accents; Llama-2/SentencePiece injects a
  leading space.
- Watch `add_special_tokens` — may prepend BOS. (LLaDA's doesn't by default.)
