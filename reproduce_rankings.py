from pathlib import Path
import csv
root = Path(__file__).resolve().parent
rows = []
for role in ['bat', 'bowl']:
    with (root / 'results' / f'{role}_full_rankings.csv').open() as handle:
        board = list(csv.DictReader(handle))
    assert [int(row['rank']) for row in board] == list(range(1, len(board)+1))
    for row in board[:10]:
        rows.append((role, row['player_id'], row['rank'], row['player']))
with (root / 'results' / 'final_top_tens.csv').open() as handle:
    saved = [(r['role'], r['player_id'], r['rank'], r['player']) for r in csv.DictReader(handle)]
assert rows == saved
for row in rows:
    print(' | '.join(row))
print('All 20 ranks agree with the saved final submission.')
