"""Frozen MIND history selection helpers; caller supplies original dataset tables."""
import hashlib
import numpy as np
import pandas as pd
SEED = 20261014

def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def positions(value):
    return [] if pd.isna(value) or str(value) == '' else [int(x) for x in str(value).split(';')]


def permuted_positions(user_id, split, imp_id, repeat, m, k):
    if not 0 <= k <= m:
        raise ValueError('Invalid click count')
    seed = int(digest(f'{SEED}|{user_id}|{split}|{imp_id}|{repeat}')[:16], 16)
    return np.sort(np.random.Generator(np.random.PCG64(seed)).choice(m, k, replace=False))


def validate_frames(users, history, targets, news):
    ids = set(users.user_id.astype(str))
    if len(ids) != len(users) or set(history.user_id) != ids or set(targets.user_id) != ids:
        raise ValueError('User cohort is duplicated or incomplete')
    if news.news_id.duplicated().any():
        raise ValueError('Duplicate news IDs')
    news_ids = set(news.news_id)
    if news.title.fillna('').str.strip().eq('').any():
        raise ValueError('Empty news title')
    for label, frame in [('history', history), ('targets', targets)]:
        if frame.duplicated(['user_id', 'source_split', 'imp_id']).any():
            raise ValueError(f'Duplicate {label} exposure')
        for row in frame.itertuples():
            candidates = str(row.candidate_news_ids).split(';')
            clicked = positions(row.original_clicked_positions)
            k = int(row.n_clicks if label == 'history' else row.n_positive)
            if len(candidates) != int(row.n_candidates) or len(clicked) != k or len(set(clicked)) != k:
                raise ValueError('Candidate or click count mismatch')
            if not all(0 <= x < len(candidates) for x in clicked) or not set(candidates) <= news_ids:
                raise ValueError('Invalid clicked position or missing news')
            if label == 'targets' and (len(candidates) < 5 or not 0 < k < len(candidates)):
                raise ValueError('Target eligibility failed')
    for uid in sorted(ids):
        h, t = history[history.user_id.eq(uid)], targets[targets.user_id.eq(uid)]
        if len(t) != 3 or set(t.impression_index.astype(int)) != {0, 1, 2}:
            raise ValueError('Exactly three target exposures are required')
        cutoff = pd.to_datetime(t.timestamp).min()
        if not (pd.to_datetime(h.timestamp) < cutoff).all():
            raise ValueError('History must be strictly before the earliest target')
        if len(h) < 3 or int(h.n_candidates.sum()) <= 20:
            raise ValueError('History eligibility failed')
        if int(h.loc[(h.n_clicks > 0)&(h.n_clicks < h.n_candidates), 'n_clicks'].sum()) < 1:
            raise ValueError('No permutable history clicks')


def trim_words(text, maximum):
    return ' '.join(str(text or '').split()[:maximum])


def candidate_text(item):
    return (f"- Category: {item['category']}\n- Subcategory: {item['subcategory']}\n"
            f"- Title: {item['title']}\n- Abstract: {trim_words(item['abstract'], 80)}")


def history_text(ids, news):
    return '\n'.join(f"{i}. [{news[n]['category']}/{news[n]['subcategory']}] {news[n]['title']}\n"
                     f"   Abstract: {trim_words(news[n]['abstract'], 50)}"
                     for i, n in enumerate(ids, 1))
