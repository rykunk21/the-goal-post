"""Whole-game ESPN summary fallback, preserving returned drive/play array order.

Never sort IDs, clocks or scores. Never splice feeds or repair score/clock
reversals. Keep a source-row map so canonical event ordinals are traceable.
"""
import pandas as pd
from .espn_college_adapter import extract_espn_college


def summary_frame(payload, meta):
    header = payload.get('header', {})
    if str(header.get('id')) != meta['game_id']:
        raise ValueError('raw ESPN event identity mismatch')
    competitions = header.get('competitions', [])
    if len(competitions) != 1:
        raise ValueError('raw ESPN ambiguous competition')
    competition = competitions[0]
    if competition.get('status', {}).get('type', {}).get('completed') is not True:
        raise ValueError('raw ESPN game incomplete')
    if str(competition.get('date', ''))[:10] != meta['game_date'][:10]:
        raise ValueError('raw ESPN schedule/date mismatch')
    sides = {}
    for competitor in competition.get('competitors', []):
        side = competitor.get('homeAway')
        if side not in ('home', 'away') or side in sides:
            raise ValueError('raw ESPN ambiguous competitors')
        if str(competitor.get('id')) != meta[side + '_team_id']:
            raise ValueError('raw ESPN schedule/team identity mismatch')
        score = float(competitor.get('score', 'nan'))
        if score != meta[side + '_final_score']:
            raise ValueError('raw ESPN schedule/final mismatch')
        sides[side] = competitor
    if set(sides) != {'home', 'away'}:
        raise ValueError('raw ESPN missing competitors')
    drives = payload.get('drives', {})
    if drives.get('current', {}).get('plays'):
        raise ValueError('raw ESPN unresolved current drive')
    previous = drives.get('previous', [])
    if not previous:
        raise ValueError('raw ESPN no drive play-by-play')
    records, mapping, seen = [], [], set()
    for di, drive in enumerate(previous):
        for pi, play in enumerate(drive.get('plays', [])):
            pid = play.get('id')
            if not isinstance(pid, str) or not pid or pid in seen:
                raise ValueError('raw ESPN missing/duplicate exact play ID')
            seen.add(pid)
            ordinal = len(records) + 1
            row = dict(game_play_number=ordinal, status_type_completed=True)
            for side in ('home', 'away'):
                row[side + 'TeamId'] = meta[side + '_team_id']
                row[side + 'FinalScore'] = meta[side + '_final_score']
            for name in ('homeScore', 'awayScore', 'period.number', 'type.text',
                         'start.down', 'start.team.id', 'start.distance',
                         'start.yardsToEndzone', 'clock.displayValue'):
                value = play
                for part in name.split('.'):
                    value = value.get(part) if isinstance(value, dict) else None
                row[name] = value
            records.append(row)
            mapping.append(dict(canonical_play_id=ordinal, source_play_id=pid,
                                drive_array_index=di, play_array_index=pi,
                                source_sequence_number=play.get('sequenceNumber')))
    if not records:
        raise ValueError('raw ESPN no plays')
    return pd.DataFrame(records), mapping


def extract_raw_espn(payload, meta):
    frame, mapping = summary_frame(payload, meta)
    return extract_espn_college(frame, meta), mapping
