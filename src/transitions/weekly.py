"""Weekly NFL empirical simulations. No bets, provider odds or VAE training."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import argparse
import hashlib
import json
import re

import numpy as np
import pandas as pd

from .paths import ARTIFACTS, RESET
from .simulator import run, context
from .simulate_reset import summarize

from .espn_schedule import load as load_espn_schedule
META = ['game_id', 'game_date', 'home_team_id', 'away_team_id', 'season',
        'matrix_status', 'matrix_shape']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def instant(value):
    at = pd.Timestamp(value)
    if pd.isna(at) or at.tzinfo is None:
        raise ValueError('as-of must include a timezone')
    return at.tz_convert('UTC')


def select_week(frame, as_of, season=None, week=None, game_type='REG'):
    """Nearest upcoming NFL week within seven days, or explicit season/week."""
    at = instant(as_of)
    required = {'game_id', 'season', 'week', 'game_type', 'gameday', 'gametime',
                'home_team', 'away_team', 'home_score', 'away_score'}
    if not required.issubset(frame.columns):
        raise ValueError('Schedule missing required fields: ' + ', '.join(sorted(required-set(frame.columns))))
    if frame.game_id.isna().any() or frame.game_id.duplicated().any():
        raise ValueError('Missing or duplicate schedule game identity')
    if (season is None) != (week is None):
        raise ValueError('Supply both season and week, or neither')
    data = frame.copy()
    data['season'] = pd.to_numeric(data.season, errors='raise').astype(int)
    data['week'] = pd.to_numeric(data.week, errors='raise').astype(int)
    # nflverse gametime is Eastern local time, including international games.
    local = pd.to_datetime(data.gameday.astype(str) + ' ' + data.gametime.astype(str), errors='coerce')
    data['kickoff'] = local.dt.tz_localize('America/New_York', ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC')
    if 'kickoff_utc' in data:
        data['kickoff'] = pd.to_datetime(data.kickoff_utc,utc=True,errors='coerce')
    if season is None:
        future = data[data.game_type.ne('PRE') & data.kickoff.gt(at) &
                      data.kickoff.le(at + pd.Timedelta(days=7)) &
                      data.home_score.isna() & data.away_score.isna() &
                      (data.source_status.eq('scheduled') if 'source_status' in data else True)].sort_values(['kickoff', 'game_id'])
        if future.empty:
            return [], [], {'status': 'no_upcoming_week_within_seven_days'}
        first = future.iloc[0]
        season, week, game_type = int(first.season), int(first.week), first.game_type
    selected = data[data.season.eq(season) & data.week.eq(week) & data.game_type.eq(game_type)]
    games, excluded = [], []
    for row in selected.sort_values(['kickoff','game_id']).itertuples():
        reason = None
        if hasattr(row,'source_status') and row.source_status != 'scheduled':
            reason = 'provider_status_' + str(row.source_status)
        elif pd.notna(row.home_score) or pd.notna(row.away_score):
            reason = 'score_recorded_already_played_or_in_progress'
        elif pd.isna(row.kickoff):
            reason = 'missing_or_invalid_kickoff'
        elif row.kickoff <= at:
            reason = 'kickoff_at_or_before_as_of'
        elif pd.isna(row.home_team) or pd.isna(row.away_team) or row.home_team == row.away_team:
            reason = 'invalid_team_identity'
        elif not re.fullmatch(r'[A-Za-z0-9_-]+', str(row.game_id)):
            reason = 'unsafe_game_identifier'
        if reason:
            excluded.append({'game_id': str(row.game_id), 'reason': reason})
        else:
            games.append(dict(game_id=str(row.game_id), season=int(row.season), week=int(row.week),
                              game_type=row.game_type, home=row.home_team, away=row.away_team,
                              kickoff_utc=row.kickoff.isoformat()))
    return games, excluded, dict(status='selected' if len(selected) else 'week_not_found',
                                 season=int(season), week=int(week), game_type=game_type,
                                 scheduled_rows=len(selected))


class Dataset:
    """Reuse frozen matrices; never load final-score labels for simulation."""
    def __init__(self, data):
        self.data = Path(data)
        self.keys = np.asarray(json.loads((self.data/'transition_set.json').read_text())['keys'], dtype=int)
        if self.keys.ndim != 2 or self.keys.shape[1] != 5 or np.any(self.keys[:,:2] < 0) or np.any(self.keys[:,:2] >= 72):
            raise ValueError('Football-only edge catalog required')
        if len({tuple(k) for k in self.keys}) != len(self.keys):
            raise ValueError('Duplicate edge keys')
        self.meta = pd.read_parquet(self.data/'nfl_games.parquet', columns=META)
        if self.meta.game_id.duplicated().any():
            raise ValueError('Duplicate dataset game identity')
        self.meta['day'] = pd.to_datetime(self.meta.game_date, utc=True, errors='raise').dt.floor('D')
        self.segments = pd.read_parquet(self.data/'parsed_segments.parquet', columns=[
            'league','game_id','source','destination','switch','own_points','opponent_points','remaining','lead','elapsed'])
        self.cache = {}
        self.timing_cache = {}
        self.input_hashes = {name: digest(self.data/name) for name in
                             ['transition_set.json','nfl_games.parquet','parsed_segments.parquet']}

    def prepare(self, game, as_of):
        # Conservative day cutoff: no same-day observations and no target game.
        cutoff = min(instant(as_of), instant(game['kickoff_utc'])).floor('D')
        eligible = self.meta[self.meta.day.lt(cutoff) & self.meta.matrix_status.eq('parsed_regulation') &
                             self.meta.game_id.ne(game['game_id'])]
        if self.meta.game_id.eq(game['game_id']).any():
            # The target is explicitly excluded even if its stored date is wrong.
            eligible = eligible[eligible.game_id.ne(game['game_id'])]
        blends, details = [], {}
        for team in [game['home'], game['away']]:
            select = eligible[eligible.season.eq(game['season']) &
                              (eligible.home_team_id.eq(team) | eligible.away_team_id.eq(team))].sort_values(['day','game_id'])
            if select.empty:
                raise ValueError('No earlier current-season parsed history for ' + str(team))
            cache_key = (team, tuple(select.game_id))
            if cache_key not in self.cache:
                rows = pd.read_parquet(self.data/'nfl_games.parquet', columns=META+['transition_probabilities_flat'],
                                       filters=[('game_id','in',select.game_id.tolist())]).set_index('game_id')
                matrices = []
                for gid in select.game_id:
                    row = rows.loc[gid]
                    shape = tuple(row.matrix_shape)
                    if shape != (2,9,len(self.keys)):
                        raise ValueError('Matrix/catalog shape mismatch')
                    p = np.asarray(row.transition_probabilities_flat, dtype=float).reshape(shape)
                    if not np.isfinite(p).all() or np.any(p < 0):
                        raise ValueError('Invalid probabilities')
                    matrices.append(p[0 if row.home_team_id == team else 1])
                blend = np.zeros_like(matrices[0]); contributors = Counter()
                for state in range(72):
                    ix = np.flatnonzero(self.keys[:,0] == state)
                    for ctx in range(9):
                        supported = [p[ctx,ix] for p in matrices if p[ctx,ix].sum() > 0]
                        contributors[len(supported)] += 1
                        if supported:
                            if not all(np.isclose(p.sum(),1,atol=1e-6) for p in supported):
                                raise ValueError('Bad probability normalization')
                            values = np.mean(supported, axis=0)
                            blend[ctx,ix] = values/values.sum()
                self.cache[cache_key] = blend, dict(contributors)
            blend, contributors = self.cache[cache_key]
            blends.append(blend)
            details[team] = dict(games=select[['game_id','game_date','home_team_id','away_team_id']].to_dict('records'),
                                 rows_by_contributing_games=contributors)
        # Historical timing also obeys cutoff; no future league segments.
        timing_key = tuple(sorted(eligible.game_id))
        if timing_key not in self.timing_cache:
            seg = self.segments[self.segments.league.eq('nfl') & self.segments.game_id.isin(eligible.game_id)]
            index = {tuple(k): j for j,k in enumerate(self.keys)}; records = defaultdict(list)
            for row in seg.itertuples():
                j = index[(row.source,row.destination,int(row.switch),row.own_points,row.opponent_points)]
                ctx = context(row.remaining,row.lead)
                records[j,ctx].append(float(row.elapsed))
                records[j,9+ctx//3].append(float(row.elapsed))
            self.timing_cache[timing_key] = records, sorted(seg.game_id.unique().tolist())
        records, timing_ids = self.timing_cache[timing_key]
        return self.keys, np.array(blends), records, dict(teams=details,
            earlier_than_utc_date=cutoff.isoformat(), timing_game_ids=timing_ids)



def game_seed(seed, game_id):
    return int.from_bytes(hashlib.sha256(f'{seed}:{game_id}'.encode()).digest()[:8], 'big')


def plot(scores, game, result, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    good = scores[(scores >= 0).all(1)]
    if not len(good):
        return
    fig, axes = plt.subplots(1,2,figsize=(12,6))
    fig.suptitle(f"{game['away']} at {game['home']} — regulation simulation", fontsize=17)
    for ax, values, title, label in [
        (axes[0],good.sum(1),'Total points','Home + away'),
        (axes[1],good[:,0]-good[:,1],'Home margin',f"{game['home']} − {game['away']}")]:
        x,n = np.unique(values,return_counts=True)
        ax.bar(x,n/len(good),color='#176b59' if ax is axes[0] else '#a33c48')
        ax.axvline(values.mean(),color='#172b3a',ls='--')
        ax.set(title=title,xlabel=label,ylabel='Simulated probability')
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.text(.97,.97,f'μ = {values.mean():.2f}\nσ = {values.std():.2f}',
                transform=ax.transAxes,ha='right',va='top',bbox=dict(facecolor='white',alpha=.9,edgecolor='#ddd'))
    diag = result['diagnostics']
    fig.text(.06,.07,f"Included {len(good):,}/{len(scores):,}; approximate completions {diag['grace_completed_games']:,}; "
             f"incomplete {len(scores)-len(good):,}.\nConditional on completion; empirical model, not a calibrated forecast. No overtime.",fontsize=10)
    fig.tight_layout(rect=(0,.15,1,.92))
    fig.savefig(folder/'distributions.png',dpi=160)
    fig.savefig(folder/'distributions.pdf')
    plt.close(fig)


def run_slate(dataset, games, output, as_of, n=50000, seed=20260924):
    if not isinstance(n,int) or n <= 0:
        raise ValueError('simulations must be positive')
    results = []
    for game in games:
        folder = Path(output)/game['game_id']; folder.mkdir()
        result = dict(game=game, seed=game_seed(seed,game['game_id']), attempted=n)
        try:
            keys,p,timing,provenance = dataset.prepare(game,as_of)
            scores,diag,events,rows = run(keys,p,timing,n=n,seed=result['seed'],grace_seconds=10)
            result.update(summarize(scores,diag))
            result['status'] = 'simulated' if result['n'] else 'no_completed_simulations'
            result['provenance'] = provenance
            np.save(folder/'scores.npy',scores)
            pd.DataFrame(rows).to_parquet(folder/'simulation-outcomes.parquet',index=False)
            pd.DataFrame(events).to_parquet(folder/'missing-row-events.parquet',index=False)
            good = scores[(scores>=0).all(1)]
            for name,values in [('total',good.sum(1)),('home_minus_away',good[:,0]-good[:,1])]:
                x,counts = np.unique(values,return_counts=True)
                pd.DataFrame(dict(points=x,count=counts,probability=counts/len(good) if len(good) else [])).to_csv(folder/(name+'-pmf.csv'),index=False)
            plot(scores,game,result,folder)
        except (ValueError,KeyError,IndexError) as exc:
            result.update(status='blocked',error_type=type(exc).__name__,reason=str(exc))
        # I/O/runtime exceptions are not disguised as sparse model coverage.
        save(folder/'result.json',result); results.append(result)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schedule',type=Path,help='Frozen compatible schedule CSV; omitted uses cached ESPN schedule')
    parser.add_argument('--schedule-cache',type=Path,default=ARTIFACTS/'schedule-cache')
    parser.add_argument('--cache-max-age-seconds',type=float,default=3600)
    parser.add_argument('--refresh-schedule',action='store_true')
    parser.add_argument('--data',type=Path,default=RESET/'data')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--as-of',default=datetime.now(timezone.utc).isoformat())
    parser.add_argument('--season',type=int); parser.add_argument('--week',type=int)
    parser.add_argument('--game-type',default='REG',choices=['REG','WC','DIV','CON','SB'])
    parser.add_argument('--simulations',type=int,default=50000)
    parser.add_argument('--seed',type=int,default=20260924)
    args = parser.parse_args(argv); at = instant(args.as_of)
    if args.simulations <= 0:
        parser.error('--simulations must be positive')
    if (args.season is None) != (args.week is None):
        parser.error('--season and --week must be supplied together')
    output = args.output or ARTIFACTS/'weekly'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True,exist_ok=False)
    report = dict(as_of_utc=at.isoformat(),created_at_utc=datetime.now(timezone.utc).isoformat(),
                  regulation_only=True,betting_admitted=False,requested_simulations=args.simulations,
                  root_seed=args.seed,score_labels_loaded_for_simulation=False,
                  code_sha256={name: digest(Path(__file__).with_name(name)) for name in
                               ['weekly.py','espn_schedule.py','simulator.py','simulate_reset.py']})
    try:
        if args.schedule:
            raw = args.schedule.read_bytes()
            report['schedule'] = dict(source=str(args.schedule.resolve()),sha256=hashlib.sha256(raw).hexdigest(),
                retrieved_at_utc=datetime.now(timezone.utc).isoformat(),freshness='local_snapshot_age_unknown')
        else:
            raw,espn_raw,metadata = load_espn_schedule(args.schedule_cache,args.season,args.week,args.game_type,
                max_age_seconds=args.cache_max_age_seconds,refresh=args.refresh_schedule)
            report['schedule'] = metadata
            (output/'schedule-espn.json').write_bytes(espn_raw)
        (output/'schedule.csv').write_bytes(raw)
        games,excluded,selection = select_week(pd.read_csv(BytesIO(raw)),at,args.season,args.week,args.game_type)
        report.update(selection=selection,excluded_games=excluded,selected_games=games)
        if not games:
            report.update(status='no_scheduled_matchups',results=[])
        else:
            dataset = Dataset(args.data); report['dataset_hashes'] = dataset.input_hashes
            report['results'] = run_slate(dataset,games,output,at,args.simulations,args.seed)
            report['status'] = 'complete' if all(r['status']=='simulated' for r in report['results']) else 'incomplete'
        report['limitations'] = [
            'Equal supported-row current-season offense blends; no home-field or opponent-defense adjustment.',
            'Historical imputation and same-time timing; catalog built from descriptive full snapshot, not a train-only vocabulary.',
            'Regulation only; censored final segments and under-ten-second grace can omit late scoring.',
            'Excluded paths are not missing at random. Probability masses condition on included simulations.',
            'A current schedule does not refresh the frozen matrix dataset. Source game IDs/dates remain visible.',
            'Historical as-of runs using a current schedule are retrospective diagnostics, not evidence of a forecast recorded then.']
    except Exception as exc:
        report.update(status='failed',error_type=type(exc).__name__,reason=str(exc))
        save(output/'report.json',report)
        raise
    save(output/'report.json',report)
    print(json.dumps(dict(output=str(output),status=report['status'],matchups=len(report['selected_games']))))
    return 0 if report['status']=='complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
