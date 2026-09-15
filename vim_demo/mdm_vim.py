"""Human-in-the-loop masked-diffusion sampling, with a text file as the UI.

A watcher owns a file. You edit it in vim and :w; the watcher applies your edits,
runs one forward pass, and re-renders the file, which vim reloads.

  server, real weights :  uv run python vim_demo/mdm_vim.py --file mdm.txt
  anywhere, no weights :  uv run python vim_demo/mdm_vim.py --fake --file mdm.txt
  editor (other pane)  :  vim -S vim_demo/mdm.vim mdm.txt

From a Python session that already has the model and tokenizer loaded:

  import sys; sys.path.insert(0, 'vim_demo')
  from mdm_vim import serve
  sess = serve(model, tokenizer, 'mdm.txt', prompt='What is the capital of France?', gen_len=16)

The table below the header has one column per answer position:
  pos / tok / id✎ / pick✎      the current token; type an id, or a candidate rank to commit
  state / conf / order         ✔ committed (step, how), confidence, low-confidence unmask order
  #1..#K                       top-K candidates: text, id, probability (▸ = current token)
Masked positions show the live distribution. Committed positions show the distribution the
model gave when you committed them ("frozen"); `view live` shows the current forward pass
instead, where the token is visible to the model (LLaDA is not trained on those outputs).
"""
import argparse
import itertools
import re
import time
import traceback
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import torch

MASK = '[M]'
SPLIT_MASK = re.compile(r'(\[M\])')
CELL_SEP = re.compile(r'[│|]')
TOP_LINE = re.compile(r'(CMD|PROMPT|ANSWER)\s*✎?\s*> ?(.*)')
STATUS_REV = re.compile(r'status\s*> rev (\d+)')
IS_INT = re.compile(r'[0-9]+').fullmatch
EIGHTHS = ' ▏▎▍▌▋▊▉'
LABEL_W = 9
VIM_SCRIPT = Path(__file__).with_name('mdm.vim')
HELP = """\
# {title}
# edit a ✎ line or cell, then :w  →  edits applied, one forward pass, this file re-rendered
#   pick ✎    1..K commits that candidate · x re-masks           id ✎      any token id (m = mask)
#   ANSWER ✎  free text, [M] = mask, \\n = newline                PROMPT ✎  new question, answer is kept
#   CMD ✎     auto [k] · finish [k] · undo · reset · len N · topk K · wrap N · view live|frozen · quit   (in vim: :Mdm auto 2)
#   ▸ current token · conf = p(top-1) of a mask, p(token) when it was committed · order = which mask low-confidence sampling unmasks first
"""


# ---------------------------------------------------------------- text helpers

def dwidth(s):
    return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in s)


def fit(s, w):
    """Truncate or pad s to exactly w terminal columns."""
    if dwidth(s) > w:
        out = ''
        for c in s:
            if dwidth(out + c) > w - 1:
                break
            out += c
        s = out + '…'
    return s + ' ' * (w - dwidth(s))


def show(s):
    """Token text for read-only cells: whitespace made visible."""
    s = s.replace(' ', '␣').replace('\n', '↵').replace('\t', '⇥')
    return ''.join(c if c.isprintable() else f'\\x{ord(c):02x}' for c in s) or '∅'


def esc(s):
    """Token text for editable lines; unesc() inverts it."""
    s = s.replace('\\', '\\\\').replace('\n', '\\n').replace('\t', '\\t')
    return re.sub(r'[\x00-\x1f\x7f]', lambda m: f'\\x{ord(m[0]):02x}', s)


def unesc(s):
    table = {'n': '\n', 't': '\t'}
    return re.sub(r'\\(x[0-9a-fA-F]{2}|.)', lambda m: chr(int(m[1][1:], 16)) if len(m[1]) == 3 else table.get(m[1], m[1]), s)


def fmt_p(p):
    return '  ?' if p is None else '1.0' if p >= 0.995 else f'{p:.2f}'[1:]


def bar(p, w):
    n = round((p or 0) * w * 8)
    return '█' * (n // 8) + (EIGHTHS[n % 8] if n % 8 else '')


# ---------------------------------------------------------------- file parsing

@dataclass
class Page:
    """What one version of the file says."""
    rev: int | None = None
    cmd: str = ''
    prompt: str | None = None
    answer: str | None = None
    ids: dict = field(default_factory=dict)    # position -> id cell text
    picks: dict = field(default_factory=dict)  # position -> pick cell text
    cands: dict = field(default_factory=dict)  # position -> candidate ids, in displayed order
    problems: list = field(default_factory=list)


def parse(text):
    f, pos = Page(), None
    for line in text.split('\n'):
        line = line.rstrip('\r')
        if m := TOP_LINE.match(line):
            if m[1] == 'CMD':
                f.cmd = m[2].strip()
            elif m[1] == 'PROMPT':
                f.prompt = unesc(m[2])
            else:
                f.answer = m[2]
            continue
        if m := STATUS_REV.match(line):
            f.rev = int(m[1])
        if line.startswith('# ') or not CELL_SEP.search(line):
            continue
        label, *cells = CELL_SEP.split(line)
        label = label.replace('✎', '').strip()
        if len(cells) == len(pos or ()) + 1 and not cells[-1].strip():
            cells.pop()
        if label == 'pos':
            nums = [c.strip() for c in cells if c.strip()]
            pos = [int(c) for c in nums] if all(IS_INT(c) for c in nums) else None
        elif not pos:
            continue
        elif len(cells) != len(pos):
            if label in ('id', 'pick'):
                f.problems.append(f'{label} row of pos {pos[0]}..{pos[-1]} has {len(cells)} cells, expected {len(pos)}: ignored')
        elif label in ('id', 'pick'):
            (f.ids if label == 'id' else f.picks).update({i: c.strip() for i, c in zip(pos, cells)})
        elif re.fullmatch(r'#\d+', label):
            for i, c in zip(pos, cells):
                if len(words := c.split()) >= 3 and IS_INT(words[-2]):
                    f.cands.setdefault(i, []).append(int(words[-2]))
    return f


# ---------------------------------------------------------------- session

@dataclass(frozen=True)
class Slot:
    tok: int
    step: int = 0
    how: str = ''
    p: float | None = None     # model's probability of tok when it was committed
    top: tuple | None = None   # (ids, probs) the model showed when it was committed


class Session:
    """One human-driven sampling run: prompt, answer slots, and the latest forward pass."""

    def __init__(self, model, tokenizer, prompt='What is the capital of France?', gen_len=16, mask_id=5,
                 topk=5, wrap=6, col_width=20, chat=True, title='masked diffusion playground'):
        self.model, self.tokenizer, self.mask_id = model, tokenizer, mask_id
        self.topk, self.wrap, self.col_width = topk, wrap, max(col_width, 16)
        self.chat, self.title = chat, title
        self.device = getattr(model, 'device', 'cpu')
        self.view, self.step, self.history, self.stopped = 'frozen', 0, [], False
        self.notes, self._pieces = ['everything masked: pick, type, or `auto`'], {}
        self.rev, self.renders = 0, {}  # recent renders, so a save is diffed against the one it was edited from
        self.set_prompt(prompt)
        self.slots = [Slot(mask_id)] * gen_len
        self.forward()

    # ---- model and tokenizer

    def set_prompt(self, prompt):
        self.prompt, self.dirty = prompt, True
        if self.chat:
            prompt = self.tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                                        add_generation_prompt=True, tokenize=False)
        self.prompt_ids = self.tokenizer(prompt)['input_ids']

    def piece(self, tok):
        if tok not in self._pieces:
            self._pieces[tok] = self.tokenizer.decode([tok], skip_special_tokens=False,
                                                      clean_up_tokenization_spaces=False)
        return self._pieces[tok]

    @torch.no_grad()
    def forward(self):
        t0 = time.perf_counter()
        x = torch.tensor([self.prompt_ids + [s.tok for s in self.slots]], device=self.device)
        logits = self.model(input_ids=x).logits[0, len(self.prompt_ids):]
        self.probs = logits.float().softmax(-1)
        top = self.probs.topk(max(self.topk, 10), dim=-1)
        self.top_ids, self.top_p = top.indices.tolist(), top.values.tolist()
        self.fwd_secs, self.dirty = time.perf_counter() - t0, False

    def refresh(self):
        if self.dirty:
            self.forward()

    # ---- state

    def masked(self):
        return [i for i, s in enumerate(self.slots) if s.tok == self.mask_id]

    def snapshot(self):
        return self.prompt, self.prompt_ids, list(self.slots), self.step

    def begin_step(self):
        self.history.append(self.snapshot())
        self.step += 1
        self.dirty = True

    def frozen(self, i):
        s = self.slots[i]
        return s.tok != self.mask_id and s.top is not None and self.view == 'frozen'

    def dist(self, i):
        """Candidates shown for position i: live for masks, as-committed for ✔ (unless view live)."""
        if self.frozen(i):
            return self.slots[i].top[0][:self.topk], self.slots[i].top[1][:self.topk]
        return self.top_ids[i][:self.topk], self.top_p[i][:self.topk]

    def conf(self, i):
        s = self.slots[i]
        if s.tok == self.mask_id:
            return self.top_p[i][0]
        return s.p if self.frozen(i) else self.probs[i, s.tok].item()

    def commit(self, i, tok, how):
        """Put tok at position i, remembering what the model showed when the choice was made."""
        self.dirty = True
        if tok == self.mask_id:
            self.slots[i] = Slot(tok)
            return f'{i}→{MASK}'
        if self.frozen(i):  # changing a committed token: judge it by the distribution it was picked from
            top = self.slots[i].top
            p = dict(zip(*top)).get(tok)
        else:
            top, p = (self.top_ids[i], self.top_p[i]), self.probs[i, tok].item()
        self.slots[i] = Slot(tok, self.step, how, p, top)
        return f'{i}←{show(self.piece(tok))} {how} p={fmt_p(p)}'

    def answer_pieces(self):
        return [MASK if s.tok == self.mask_id else esc(self.piece(s.tok)) for s in self.slots]

    def retokenize(self, new):
        """Diff an edited ANSWER line against the rendered one and tokenize only the changed span,
        so untouched positions keep their ids. Returns (head, tail, ids): slots[head:tail] -> ids."""
        pieces = self.answer_pieces()
        old = ''.join(pieces)
        shorter = min(len(old), len(new))
        pre = next((k for k, (a, b) in enumerate(zip(old, new)) if a != b), shorter)
        suf = next((k for k, (a, b) in enumerate(zip(old[::-1], new[::-1])) if a != b), shorter)
        suf = min(suf, shorter - pre)
        ends = list(itertools.accumulate(map(len, pieces)))
        starts = [e - len(pc) for e, pc in zip(ends, pieces)]
        head = sum(e <= pre for e in ends)
        tail = next((j for j in range(head, len(pieces)) if starts[j] >= len(old) - suf), len(pieces))
        a = ends[head - 1] if head else 0
        b = len(new) - (len(old) - (starts[tail] if tail < len(pieces) else len(old)))
        ids = []
        for seg in SPLIT_MASK.split(new[a:b]):
            if seg == MASK:
                ids.append(self.mask_id)
            elif seg:
                ids += self.tokenizer.encode(unesc(seg), add_special_tokens=False)
        return head, tail, ids

    # ---- applying a save

    def handle(self, text):
        """Apply what the user changed in the file, run the command, refresh the forward pass.

        Changes are measured against the render the file was edited from (its rev), not the current
        state, so a save made before the editor reloaded the previous result does not undo that result."""
        page = parse(text)
        base = parse(self.renders.get(page.rev) or self.render())
        backup = {**self.__dict__, 'slots': list(self.slots), 'history': list(self.history)}
        try:
            self.notes = self.apply(page, base)
            self.refresh()
        except Exception:
            self.__dict__.update(backup)
            raise

    def apply(self, f, base):
        notes = list(f.problems)
        cmd, *args = f.cmd.split() or ['']
        if cmd == 'undo':
            if not self.history:
                return notes + ['nothing to undo']
            self.prompt, self.prompt_ids, self.slots, self.step = self.history.pop()
            self.dirty = True
            return notes + [f'undo → back to step {self.step}']

        edits = self.cell_edits(f, base, notes)
        resized = None
        if f.answer is not None and f.answer != base.answer:
            head, tail, ids = self.retokenize(f.answer)
            if base.answer != ''.join(self.answer_pieces()):
                notes.append('ANSWER ignored: it was edited on an outdated render, redo it')
            elif len(ids) == tail - head:
                text_edits = {head + j: (t, 'text') for j, t in enumerate(ids) if t != self.slots[head + j].tok}
                if clash := sorted(i for i in text_edits if i in edits and edits[i][0] != text_edits[i][0]):
                    notes.append(f'ANSWER ignored at pos {clash}: id/pick edits win')
                edits = {**text_edits, **edits}
            elif edits:
                notes.append('ANSWER ignored: it changes the length, save it without id/pick edits')
            else:
                resized = head, tail, ids

        new_prompt = f.prompt not in (None, base.prompt, self.prompt)
        if edits or resized or new_prompt:
            self.begin_step()
            if new_prompt:
                self.set_prompt(f.prompt)
                notes.append('prompt changed')
            if resized:
                head, tail, ids = resized
                old_len = len(self.slots)
                self.slots[head:tail] = [Slot(t) if t == self.mask_id else Slot(t, self.step, 'text') for t in ids]
                notes.append(f'ANSWER retokenized, length {old_len} → {len(self.slots)}')
            notes += [self.commit(i, tok, how) for i, (tok, how) in sorted(edits.items())]
        if cmd:
            notes += self.command(cmd, args)
        return notes or ['no edits']

    def cell_edits(self, f, base, notes):
        """{position: (token id, how)} from the id and pick rows; pick wins if both are set."""
        edits, L, V = {}, len(self.slots), self.probs.shape[-1]
        for i, v in f.ids.items():
            if i >= L or not v or v == base.ids.get(i):
                continue
            tok = self.mask_id if v.lower() in ('m', '[m]') else int(v) if IS_INT(v) else -1
            if not 0 <= tok < V:
                notes.append(f'pos {i}: bad id {v!r}')
            elif tok != self.slots[i].tok:
                edits[i] = tok, 'id'
        for i, v in f.picks.items():
            if i >= L or not v or v == base.picks.get(i):
                continue
            cands = base.cands.get(i, [])  # what the user was looking at
            if v.lower() in ('x', 'm'):
                tok, how = self.mask_id, 'x'
            elif IS_INT(v) and 1 <= int(v) <= len(cands):
                tok, how = cands[int(v) - 1], f'pick#{v}'
            else:
                notes.append(f'pos {i}: bad pick {v!r}')
                continue
            if i in edits and edits[i][0] != tok:
                notes.append(f'pos {i}: pick wins over id')
            edits.pop(i, None)
            if tok != self.slots[i].tok:
                edits[i] = tok, how
        return edits

    def command(self, cmd, args):
        n = int(args[0]) if args and IS_INT(args[0]) else None
        if cmd == 'auto':
            return self.auto(n or 1)
        if cmd == 'finish':
            steps = 0
            while self.masked() and steps <= len(self.slots):
                self.auto(n or 1)
                steps += 1
            return [f'finish: {steps} auto steps, {n or 1} token(s) each']
        if cmd == 'reset':
            self.begin_step()
            self.slots = [Slot(self.mask_id)] * len(self.slots)
            return ['reset: everything masked']
        if cmd == 'len' and n:
            self.begin_step()
            self.slots = self.slots[:n] + [Slot(self.mask_id)] * (n - len(self.slots))
            return [f'answer length → {n}']
        if cmd == 'topk' and n:
            self.topk, self.dirty = n, True
            return [f'showing top {n}']
        if cmd == 'wrap' and n is not None:
            self.wrap = n
            return [f'{n or "all"} positions per block']
        if cmd == 'view' and args and args[0] in ('live', 'frozen'):
            self.view = args[0]
            return [f'committed positions show the {args[0]} distribution']
        if cmd == 'quit':
            self.stopped = True
            return ['watcher stopped; restart it to keep going']
        return [f'unknown command {" ".join([cmd, *args])!r}']

    def auto(self, k):
        """One step of LLaDA's low-confidence remasking: unmask the k masks with the most probable top-1."""
        self.refresh()
        if not (masked := self.masked()):
            return ['auto: no masks left']
        self.begin_step()
        best = sorted(masked, key=lambda i: -self.top_p[i][0])[:k]
        return [self.commit(i, self.top_ids[i][0], 'auto') for i in sorted(best)]

    # ---- rendering

    def render(self):
        W, L = self.col_width, len(self.slots)
        out = HELP.format(title=self.title).splitlines() + [
            'CMD    ✎ > ',
            f'PROMPT ✎ > {esc(self.prompt)}',
            f'ANSWER ✎ > {"".join(self.answer_pieces())}',
            f'status   > rev {self.rev + 1} · step {self.step} · {L - len(self.masked())}/{L} unmasked · prompt {len(self.prompt_ids)} tokens'
            f' · view {self.view} · forward {self.fwd_secs:.2f}s · {time.strftime("%H:%M:%S")}',
            f'last     > {" · ".join(self.notes)}',
            '',
        ]
        order = {i: r + 1 for r, i in enumerate(sorted(self.masked(), key=lambda i: -self.top_p[i][0]))}

        def row(label, cells):
            return fit(label, LABEL_W) + '│' + '│'.join(f' {fit(c, W)} ' for c in cells) + '│'

        for b in range(0, L, self.wrap or L or 1):
            cols = range(b, min(b + (self.wrap or L), L))
            rule = '─' * LABEL_W + '┼' + '┼'.join('─' * (W + 2) for _ in cols) + '┤'
            out += [
                row('pos', [str(i) for i in cols]),
                row('tok', [MASK if self.slots[i].tok == self.mask_id else show(self.piece(self.slots[i].tok)) for i in cols]),
                row('id    ✎', [str(self.slots[i].tok) for i in cols]),
                row('pick  ✎', ['' for _ in cols]),
                rule,
                row('state', [self.state(i) for i in cols]),
                row('conf', [f'{fmt_p(self.conf(i))} {bar(self.conf(i), W - 4)}' for i in cols]),
                row('order', [str(order.get(i, '')) for i in cols]),
                rule,
            ]
            dists = [self.dist(i) for i in cols]
            for k in range(self.topk):
                out.append(row(f'#{k + 1}', [self.candidate(i, k, d) for i, d in zip(cols, dists)]))
            out.append('')
        self.rev += 1
        self.renders[self.rev] = text = '\n'.join(out) + '\n'
        if len(self.renders) > 32:
            del self.renders[min(self.renders)]
        return text

    def state(self, i):
        s = self.slots[i]
        if s.tok == self.mask_id:
            return MASK
        return f'✔ s{s.step} {s.how}' + ('' if self.frozen(i) else ' live')

    def candidate(self, i, k, dist):
        ids, ps = dist
        if k >= len(ids):
            return ''
        right = f' {ids[k]} {fmt_p(ps[k])}'
        mark = '▸' if ids[k] == self.slots[i].tok else ' '
        return fit(mark + show(self.piece(ids[k])), self.col_width - len(right)) + right


# ---------------------------------------------------------------- watcher

def read(path):
    try:
        return Path(path).read_text(encoding='utf-8')
    except (FileNotFoundError, UnicodeDecodeError):  # editor mid-write
        return None


def write(path, text):
    tmp = Path(f'{path}.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)


def serve(model, tokenizer, path='mdm.txt', poll=0.2, **session_kwargs):
    """Watch `path`; after every save apply the edits, run the model, re-render. Returns the Session."""
    sess = Session(model, tokenizer, **session_kwargs)
    last = sess.render()
    write(path, last)
    print(f'watching {path}\n  open it with:  vim -S {VIM_SCRIPT} {path}\n  stop with Ctrl-C or `quit` on the CMD line')
    try:
        while not sess.stopped:
            time.sleep(poll)
            text = read(path)
            if text is None or text == last:
                continue
            time.sleep(0.05)
            if read(path) != text:  # still being written
                continue
            try:
                sess.handle(text)
            except Exception as e:
                traceback.print_exc()
                sess.notes = [f'ERROR {type(e).__name__}: {e} (this save was discarded)']
            out = sess.render()
            if read(path) != text:  # saved again meanwhile: handle that save before showing anything
                continue
            if len(out.encode()) == len(text.encode()):
                out += '\n'  # size change guarantees editors with coarse mtimes notice the rewrite
            write(path, out)
            last = out
            print(f'[{time.strftime("%H:%M:%S")}] step {sess.step}: {" · ".join(sess.notes)}')
    except KeyboardInterrupt:
        print('stopped')
    return sess


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='GSAI-ML/iLLaDA-8B-Instruct')
    ap.add_argument('--fake', action='store_true', help='fake tokenizer and fake model: no weights, no download')
    ap.add_argument('--fake-model', action='store_true', help='real tokenizer from --model, fake model')
    ap.add_argument('--file', default='mdm.txt')
    ap.add_argument('--prompt', default='What is the capital of France?')
    ap.add_argument('--gen-len', type=int, default=16)
    ap.add_argument('--mask-id', type=int, default=5, help='iLLaDA: 5, LLaDA: 126336')
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--wrap', type=int, default=6, help='positions per table block, 0 = one wide table')
    ap.add_argument('--col-width', type=int, default=20)
    ap.add_argument('--no-chat', action='store_true', help='feed the prompt as raw text, without the chat template')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    if args.fake:
        from fake_llada import FakeTokenizer
        tokenizer = FakeTokenizer()
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.fake or args.fake_model:
        from fake_llada import FakeModel
        model, title = FakeModel(tokenizer, mask_id=args.mask_id), f'FAKE model · mask id {args.mask_id}'
    else:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
        model, title = model.to(args.device).eval(), f'{args.model} · mask id {args.mask_id}'
    print(f'mask id {args.mask_id} decodes to {tokenizer.decode([args.mask_id])!r}')

    serve(model, tokenizer, args.file, prompt=args.prompt, gen_len=args.gen_len, mask_id=args.mask_id,
          topk=args.topk, wrap=args.wrap, col_width=args.col_width, chat=not args.no_chat, title=title)


if __name__ == '__main__':
    main()
