"""The same human-in-the-loop sampler as mdm_vim.py, as a web page.

On the GPU server:
  uv run python vim_demo/mdm_web.py                  # iLLaDA-8B-Instruct on http://127.0.0.1:8765
  uv run python vim_demo/mdm_web.py --fake           # no weights

On your Mac, tunnel the port through SSH, then open http://localhost:8765 :
  ssh -N -L 8765:localhost:8765 <the user@host you normally ssh to>

The server listens on 127.0.0.1 only, so nothing is exposed to the network; the SSH tunnel is
the way in. From a Python session that already holds the model:

  import sys; sys.path.insert(0, 'vim_demo')
  from mdm_web import serve
  sess = serve(model, tokenizer, port=8765, prompt='What is the capital of France?', gen_len=16)
"""
import argparse
import getpass
import json
import socket
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from mdm_vim import Session, add_model_args, load

PAGE = Path(__file__).with_name('mdm_web.html')
COMMANDS = {'auto', 'finish', 'undo', 'reset', 'len', 'topk', 'view'}


def state(sess):
    """Everything the page draws."""
    order = sess.order()
    special = set(getattr(sess.tokenizer, 'all_special_ids', ()))
    slots = []
    for i, s in enumerate(sess.slots):
        ids, ps = sess.dist(i)
        slots.append({
            'tok': s.tok, 'text': sess.piece(s.tok), 'masked': s.tok == sess.mask_id, 'step': s.step, 'how': s.how,
            'frozen': sess.frozen(i), 'conf': sess.conf(i), 'order': order.get(i),
            'cands': [{'id': t, 'text': sess.piece(t), 'p': p} for t, p in zip(ids, ps)],
        })
    return {
        'title': sess.title, 'prompt': sess.prompt, 'prompt_tokens': len(sess.prompt_ids),
        'prompt_pieces': [{'id': t, 'text': sess.piece(t), 'special': t in special} for t in sess.prompt_ids],
        'answer': ''.join(sess.answer_pieces()), 'slots': slots, 'step': sess.step, 'view': sess.view,
        'topk': sess.topk, 'notes': sess.notes, 'fwd_secs': sess.fwd_secs, 'can_undo': bool(sess.history),
        'mask_id': sess.mask_id, 'vocab': sess.probs.shape[-1],
    }


def act(sess, a):
    """One request from the page: a step of staged edits, or a command."""
    op = a.get('op')
    if op == 'step':
        L, V = len(sess.slots), sess.probs.shape[-1]
        edits = {}
        for e in a.get('edits', []):
            i, tok = int(e['pos']), int(e['id'])
            if not (0 <= i < L and 0 <= tok < V):
                raise ValueError(f'edit out of range: position {i}, id {tok}')
            edits[i] = tok, str(e.get('how', 'id'))[:16]

        def change():
            answer, notes = a.get('answer'), []
            if answer is not None and a.get('answer_base') != ''.join(sess.answer_pieces()):
                answer, notes = None, ['sentence edit ignored: it was made on an outdated view, redo it']
            return notes + sess.edit(edits, answer, a.get('prompt'))
        sess.run(change)
    elif op in COMMANDS:
        sess.run(lambda: sess.command(op, [str(x) for x in a.get('args', [])]))
    else:
        raise ValueError(f'unknown op {op!r}')


def make_handler(sess, lock):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            if url.path in ('/', '/index.html'):
                self.send(200, PAGE.read_bytes(), 'text/html; charset=utf-8')  # re-read: edit the page, refresh
            elif url.path == '/api/state':
                with lock:
                    self.send_json(state(sess))
            elif url.path == '/api/piece':
                with lock:
                    self.send_json({'text': sess.piece(int(parse_qs(url.query)['id'][0]))})
            else:
                self.send(404, b'not found', 'text/plain')

        def do_POST(self):
            if self.path != '/api/act':
                return self.send(404, b'not found', 'text/plain')
            with lock:
                try:
                    act(sess, json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0)))))
                except Exception as e:
                    traceback.print_exc()
                    sess.notes = [f'ERROR {type(e).__name__}: {e}']
                print(f'step {sess.step}: {" · ".join(sess.notes)}')
                self.send_json(state(sess))

        def send_json(self, obj):
            self.send(200, json.dumps(obj).encode(), 'application/json')

        def send(self, code, body, content_type):
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # keep the terminal for step logs
            pass

    return Handler


def serve(model, tokenizer, port=8765, host='127.0.0.1', **session_kwargs):
    """Serve the page until Ctrl-C. Returns the Session."""
    sess = Session(model, tokenizer, **session_kwargs)
    try:
        server = ThreadingHTTPServer((host, port), make_handler(sess, threading.Lock()))
    except OSError as e:
        raise SystemExit(f'cannot listen on {host}:{port} ({e.strerror}); try another --port')
    print(f'serving on http://{host}:{port}\n'
          f'  on your Mac:  ssh -N -L {port}:localhost:{port} {getpass.getuser()}@{socket.gethostname()}\n'
          f'                (or whatever user@host you normally ssh with), then open http://localhost:{port}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('stopped')
    finally:
        server.server_close()
    return sess


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--host', default='127.0.0.1',
                    help='keep 127.0.0.1 and use an SSH tunnel; 0.0.0.0 would expose the page, with no password, to the network')
    args = ap.parse_args()
    model, tokenizer, title = load(args)
    serve(model, tokenizer, port=args.port, host=args.host, prompt=args.prompt, gen_len=args.gen_len,
          mask_id=args.mask_id, topk=args.topk, chat=not args.no_chat, title=title)


if __name__ == '__main__':
    main()
