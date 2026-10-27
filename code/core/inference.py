"""Local-only categorical scoring and all-layer prompt readout.

No target labels are accepted by this module. Exact token-prefix caching is an
optimization of the same causal forward pass, never a change in the prompt.
"""
from __future__ import annotations

import hashlib
import os
import time
import numpy as np

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def span_tokens(offsets, start, end):
    indices = [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]
    if not indices:
        raise ValueError('Target text has no tokens')
    return indices


def shared_prefix_length(prompts):
    if not prompts:
        return 0
    maximum = min(min(p['targets']) for p in prompts)
    for i in range(maximum):
        if any(p['ids'][i] != prompts[0]['ids'][i] for p in prompts[1:]):
            return i
    return maximum


def score_choices(logits, experiment):
    logits = np.asarray(logits, dtype=np.float64)
    exponential = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exponential / exponential.sum(axis=1, keepdims=True)
    result = []
    for ls, ps in zip(logits, probs):
        row = {'logits': ls.tolist(), 'probabilities': ps.tolist()}
        if experiment == 'figure3':
            row.update(predicted_rating=int(np.argmax(ps)) + 1,
                       expected_rating=float(ps @ np.arange(1, 5)))
        elif experiment == 'figure4':
            row.update(click_probability=float(ps[1]))
        else:
            raise ValueError(experiment)
        result.append(row)
    return result


class PoolCapture:
    """Matches HF hidden_states: embedding, blocks 0..L-2, final norm."""
    def __init__(self, model):
        self.n_layers = model.config.num_hidden_layers + 1
        self.handles = []
        self.active = False
        self.handles.append(model.model.embed_tokens.register_forward_hook(self.hook(0)))
        for i, layer in enumerate(model.model.layers[:-1]):
            self.handles.append(layer.register_forward_hook(self.hook(i + 1)))
        self.handles.append(model.model.norm.register_forward_hook(self.hook(self.n_layers - 1)))

    def begin(self, width, last, targets):
        self.width, self.last, self.targets = width, last, targets
        self.states = {}
        self.active = True

    def hook(self, index):
        def capture(module, args, output):
            if not self.active:
                return
            tensor = output[0] if isinstance(output, tuple) else output
            if tensor.ndim != 3 or tensor.shape[1] != self.width:
                raise ValueError('Unexpected hidden tensor shape')
            import torch
            with torch.no_grad():
                last = torch.stack([tensor[b, position].float() for b, position in enumerate(self.last)])
                mean = torch.stack([tensor[b, positions].float().mean(0) for b, positions in enumerate(self.targets)])
                self.states[index] = (last, mean)
        return capture

    def finish(self):
        import torch
        self.active = False
        if set(self.states) != set(range(self.n_layers)):
            raise ValueError('Incomplete hidden-state capture')
        last = torch.stack([self.states[i][0] for i in range(self.n_layers)], dim=1).cpu().numpy()
        mean = torch.stack([self.states[i][1] for i in range(self.n_layers)], dim=1).cpu().numpy()
        self.states.clear()
        if not np.isfinite(last).all() or not np.isfinite(mean).all():
            raise ValueError('Nonfinite hidden states')
        return last, mean

    def close(self):
        for handle in self.handles:
            handle.remove()


class LocalScorer:
    def __init__(self, model_path, model_id, device='cuda:0', max_input_tokens=16384):
        import torch
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.model_id, self.device = model_id, device
        self.max_input_tokens = max_input_tokens
        torch.set_num_threads(4)
        torch.manual_seed(20261014)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True,
                                                       trust_remote_code=False, use_fast=True)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float16, attn_implementation='sdpa', device_map={'': device})
        self.model.eval()
        self.capture = PoolCapture(self.model)
        self.versions = {'torch': torch.__version__, 'transformers': transformers.__version__, 'numpy': np.__version__}
        self.choice_ids = {}
        for experiment, choices in [('figure3', '1234'), ('figure4', '01')]:
            ids = [self.tokenizer.encode(c, add_special_tokens=False) for c in choices]
            if any(len(x) != 1 for x in ids):
                raise ValueError('Choice is not a single tokenizer token')
            self.choice_ids[experiment] = [x[0] for x in ids]
        self.cache = None
        self.prefix_length = 0

    def encode(self, prefix, request, experiment):
        content = prefix + request['target_text'] + request['suffix']
        if experiment == 'figure3':
            text = self.tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
                                                     tokenize=False, add_generation_prompt=True)
            if text.endswith('<think>\n'):
                text += '</think>\n'
            text += '{"rating": '
            if text.count(content) != 1:
                raise ValueError('Chat template did not preserve the unique user content')
            start = text.index(content) + len(prefix)
            add_special = False
        else:
            text, start, add_special = content, len(prefix), True
        encoded = self.tokenizer(text, add_special_tokens=add_special, return_offsets_mapping=True)
        ids = encoded['input_ids']
        if len(ids) > self.max_input_tokens:
            raise ValueError(f'Input length {len(ids)} exceeds limit {self.max_input_tokens}; no truncation applied')
        targets = span_tokens(encoded['offset_mapping'], start, start + len(request['target_text']))
        return {'ids': ids, 'targets': targets, 'prompt_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
                'text': text, 'request_id': request['request_id']}

    def prepare_cache(self, prompts):
        """Cache only the common tokens that precede EVERY target token."""
        self.clear_cache()
        n = shared_prefix_length(prompts)
        if len(prompts) < 2 or n < 64:
            return 0
        torch = self.torch
        ids = torch.tensor([prompts[0]['ids'][:n]], device=self.device)
        with torch.inference_mode():
            out = self.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                             use_cache=True, logits_to_keep=1)
        self.cache = tuple((k.detach(), v.detach()) for k, v in out.past_key_values.to_legacy_cache())
        self.prefix_length = n
        self.prefix_ids = prompts[0]['ids'][:n]
        del out
        return n

    def clear_cache(self):
        self.cache = None
        self.prefix_length = 0

    def forward(self, prompts, experiment, cached=False, reference=False):
        torch = self.torch
        from transformers import DynamicCache
        n = self.prefix_length if cached else 0
        if cached and (self.cache is None or any(p['ids'][:n] != self.prefix_ids for p in prompts)):
            raise ValueError('Cached token prefix mismatch')
        seqs = [p['ids'][n:] for p in prompts]
        width, batch = max(map(len, seqs)), len(seqs)
        padded = [[self.tokenizer.pad_token_id] * (width - len(s)) + s for s in seqs]
        masks = [[0] * (width - len(s)) + [1] * len(s) for s in seqs]
        ids = torch.tensor(padded, dtype=torch.long, device=self.device)
        suffix_mask = torch.tensor(masks, dtype=torch.long, device=self.device)
        attention_mask = torch.cat([torch.ones((batch, n), dtype=torch.long, device=self.device), suffix_mask], dim=1)
        positions = attention_mask.cumsum(-1) - 1
        positions.masked_fill_(attention_mask == 0, 1)
        positions = positions[:, n:]
        targets = [[t - n + width - len(s) for t in p['targets']] for p, s in zip(prompts, seqs)]
        if any(min(x) < 0 for x in targets):
            raise ValueError('Target overlaps cached prefix')
        kwargs = {}
        if n:
            # DynamicCache concatenates, so original prefix tensors remain immutable.
            legacy = tuple((k.expand(batch, -1, -1, -1), v.expand(batch, -1, -1, -1)) for k, v in self.cache)
            kwargs['past_key_values'] = DynamicCache.from_legacy_cache(legacy)
            kwargs['cache_position'] = torch.arange(n, n + width, device=self.device)
        self.capture.begin(width, [width - 1] * batch, targets)
        started = time.monotonic()
        try:
            with torch.inference_mode():
                out = self.model(input_ids=ids, attention_mask=attention_mask, position_ids=positions,
                                 use_cache=bool(n), output_hidden_states=reference, logits_to_keep=1, **kwargs)
                logits = out.logits[:, -1, self.choice_ids[experiment]].float().cpu().numpy()
            last, mean = self.capture.finish()
            if reference:
                for i, tensor in enumerate(out.hidden_states):
                    expected_last = tensor[:, -1].float().cpu().numpy()
                    expected_mean = torch.stack([tensor[b, tt].float().mean(0) for b, tt in enumerate(targets)]).cpu().numpy()
                    np.testing.assert_array_equal(last[:, i], expected_last)
                    np.testing.assert_array_equal(mean[:, i], expected_mean)
            rows = score_choices(logits, experiment)
            for row, p in zip(rows, prompts):
                row.update(request_id=p['request_id'], model_id=self.model_id, experiment=experiment,
                           prompt_sha256=p['prompt_sha256'], input_tokens=len(p['ids']), n_input_tokens=len(p['ids']),
                           target_tokens=len(p['targets']), status='ok', prefix_cached_tokens=n,
                           elapsed_batch_seconds=time.monotonic() - started,
                           choice_token_ids=self.choice_ids[experiment])
            del out
            return rows, last, mean
        finally:
            self.capture.active = False

    def close(self):
        self.capture.close()
        self.clear_cache()
        del self.model
        self.torch.cuda.empty_cache()
