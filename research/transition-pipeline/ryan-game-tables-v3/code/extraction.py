"""Pure, inspectable nflverse extraction; never changes source down/distance.

A segment begins before a decision and ends before the next decision in the
same half, or at half end. Its rewards include *all* intervening raw rows.
This preserves defensive scores, conversions and kickoff-return scores.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np
import pandas as pd

DECISIONS = {'run', 'pass', 'punt', 'field_goal', 'qb_kneel', 'qb_spike', 'no_play'}
N_STATES = 72


def present(x):
    return x is not None and not pd.isna(x)


def state(down, distance, yardline):
    if not all(present(x) for x in (down, distance, yardline)):
        raise ValueError('missing decision state')
    if int(down) != down or not 1 <= down <= 4 or not 0 <= yardline <= 100 or distance < 0:
        raise ValueError('invalid decision state')
    d = 0 if distance <= 3 else 1 if distance <= 7 else 2
    y = int(np.searchsorted([5, 20, 40, 60, 80], yardline, side='left'))
    return (int(down) - 1) * 18 + d * 6 + y


def state_name(s):
    return f'{s//18+1}d_{["short","medium","long"][(s%18)//6]}_{["goal","red","opp40","mid","own40","own20"][s%6]}'


def context(seconds, lead):
    # Exactly known predecision conditions, not end-of-game labels.
    time_bin = 0 if seconds > 120 else 1 if seconds > 30 else 2
    score_bin = 0 if lead < 0 else 1 if lead == 0 else 2
    return time_bin * 3 + score_bin


@dataclass(frozen=True)
class Event:
    game_id: str
    week: int
    half: int
    play_id: float
    next_play_id: float | None
    offense: str
    defense: str
    home_offense: bool
    source: int
    destination: int  # -1 ends half, never silently maps to turnover
    switch: bool
    own_points: int
    opponent_points: int
    elapsed: float
    remaining: float
    lead: int
    play_type: str
    raw_rows: int

    @property
    def key(self):
        return (self.source, self.destination, int(self.switch), self.own_points, self.opponent_points)

    @property
    def ctx(self):
        return context(self.remaining, self.lead)


@dataclass
class ExtractedGame:
    game_id: str
    week: int
    date: str
    home: str
    away: str
    events: list[Event]
    starts: list[dict]
    regulation: tuple[int, int]
    final: tuple[int, int]
    overtime: bool


def is_decision(row):
    return (present(row.get('down')) and row.get('play_type') in DECISIONS
            and row.get('extra_point_attempt', 0) != 1
            and row.get('two_point_attempt', 0) != 1)


def replay(game):
    """Independent team attribution check against stored expected scoreboard."""
    h = a = 0
    for opening in game.starts:
        h += opening['home_points']; a += opening['away_points']
    for e in game.events:
        if e.home_offense:
            h += e.own_points; a += e.opponent_points
        else:
            h += e.opponent_points; a += e.own_points
    return int(h), int(a)


def extract_game(frame):
    g = frame.sort_values('play_id', kind='stable').reset_index(drop=True).copy()
    if g.play_id.duplicated().any():
        raise ValueError('duplicate play_id')
    first = g.iloc[0]
    gid, home, away = str(first.game_id), str(first.home_team), str(first.away_team)
    if home == away or not home or not away:
        raise ValueError('invalid team identity')
    # Administrative total_* rows can repeat an OLD score (e.g. a timeout
    # between TD and conversion). Never treat that as a negative scoring play.
    # Possession-relative pre/post fields identify both scoring teams; validate
    # the final result separately against game-level scores and final totals.
    score_rows=[];last=np.array([0.,0.])
    for _,r in g.iterrows():
        if r.posteam in (home,away) and all(present(r.get(k)) for k in ('posteam_score_post','defteam_score_post')):
            values=np.array([r.posteam_score_post,r.defteam_score_post],float)
            last=values if r.posteam==home else values[::-1]
        score_rows.append(last.copy())
    scores=np.asarray(score_rows)
    if not np.all(np.isfinite(scores)) or np.any(scores < 0) or np.any(scores != scores.astype(int)):
        raise ValueError('invalid scoreboard')
    if np.any(np.diff(np.vstack(([0, 0], scores)), axis=0) < 0):
        raise ValueError('scoreboard reversal needs explicit adjudication')
    pre = np.vstack(([0, 0], scores[:-1]))
    final = (int(first.home_score), int(first.away_score))
    if tuple(scores[-1].astype(int)) != final:
        raise ValueError('final scoreboard mismatch')
    final_totals=g[['total_home_score','total_away_score']].ffill().iloc[-1].to_numpy()
    if tuple(final_totals.astype(int))!=final:
        raise ValueError('independent final totals mismatch')
    decisions = [i for i, r in g.iterrows() if is_decision(r)]
    # Reject unclassified rows with a down rather than silently losing events.
    for i, r in g.iterrows():
        if (present(r.down) and r.get('extra_point_attempt', 0) != 1
                and r.get('two_point_attempt', 0) != 1 and i not in decisions
                and present(r.get('play_type'))):
            raise ValueError(f'unhandled down-bearing type {r.play_type}')
    for i in decisions:
        r = g.iloc[i]
        if r.posteam not in (home, away):
            raise ValueError('unknown possession team')
        state(r.down, r.ydstogo, r.yardline_100)
        if present(r.get('posteam_score')) and present(r.get('defteam_score')):
            declared = [r.posteam_score, r.defteam_score]
            if r.posteam != home:
                declared.reverse()
            if not np.array_equal(declared, pre[i]):
                raise ValueError('pre-play scoreboard metadata disagreement')
    events, starts = [], []
    previous_end = np.array([0, 0])
    for half in (1, 2):
        phase_rows = g.index[g.qtr.isin([2*half-1, 2*half])].tolist()
        ix = [i for i in decisions if i in phase_rows]
        if not ix:
            raise ValueError('empty regulation half')
        end = scores[phase_rows[-1]].astype(int)
        r0 = g.iloc[ix[0]]
        opening = pre[ix[0]].astype(int) - previous_end
        if np.any(opening < 0):
            raise ValueError('negative opening score')
        start_remaining = float(r0.half_seconds_remaining)
        if not math.isfinite(start_remaining) or not 0 <= start_remaining <= 1800:
            raise ValueError('invalid initial clock')
        starts.append(dict(half=half, state=state(r0.down,r0.ydstogo,r0.yardline_100),
                           home_offense=r0.posteam==home, remaining=start_remaining,
                           home_points=int(opening[0]), away_points=int(opening[1])))
        for k, i in enumerate(ix):
            r = g.iloc[i]
            j = ix[k+1] if k+1 < len(ix) else None
            nxt = g.iloc[j] if j is not None else None
            reward = (pre[j] if j is not None else end) - pre[i]
            rem = float(r.half_seconds_remaining)
            next_rem = float(nxt.half_seconds_remaining) if nxt is not None else 0.0
            if not all(math.isfinite(x) for x in (rem,next_rem)) or not 0 <= next_rem <= rem <= 1800:
                raise ValueError('nonmonotonic/missing half clock')
            if np.any(reward < 0):
                raise ValueError('negative segment score')
            home_off = r.posteam == home
            events.append(Event(gid,int(first.week),half,float(r.play_id),
                float(nxt.play_id) if nxt is not None else None,str(r.posteam),away if home_off else home,
                home_off,state(r.down,r.ydstogo,r.yardline_100),
                state(nxt.down,nxt.ydstogo,nxt.yardline_100) if nxt is not None else -1,
                bool(nxt is not None and r.posteam != nxt.posteam),
                int(reward[0 if home_off else 1]),int(reward[1 if home_off else 0]),
                rem-next_rem,rem,int(pre[i,0]-pre[i,1])*(1 if home_off else -1),
                str(r.play_type),(j if j is not None else phase_rows[-1]+1)-i))
        previous_end = end
    result = ExtractedGame(gid,int(first.week),str(first.game_date),home,away,events,starts,
                           tuple(previous_end),final,bool((g.qtr>4).any()))
    if replay(result) != result.regulation:
        raise ValueError('regulation replay mismatch')
    if not result.overtime and result.regulation != result.final:
        raise ValueError('regulation/final mismatch')
    return result


def extract_season(path):
    df = pd.read_parquet(path)
    games, excluded = [], []
    frame = df[df.season_type == 'REG']
    for gid, g in frame.groupby('game_id', sort=True):
        try:
            games.append(extract_game(g))
        except (ValueError, TypeError, KeyError) as e:
            excluded.append({'game_id':str(gid),'reason':str(e)})
    games.sort(key=lambda g:(g.date,g.game_id))
    audit = dict(raw_rows=len(df),regular_season_games=int(frame.game_id.nunique()),
                 reconciled_games=len(games),excluded=excluded,
                 regulation_events=sum(len(g.events) for g in games),
                 overtime_games=sum(g.overtime for g in games))
    return games, audit
