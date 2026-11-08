"""Round-9 picks: embedded Julia openings (smooth), atrium pairs, canopy gardens."""
import json

EX = '/home/diatom/Pictures/fractals/explore/round9/'
allc = {json.loads(l)['id']: json.loads(l) for l in open(EX + 'candidates.jsonl')}
for l in open(EX + 'julia/candidates.jsonl'):
    c = json.loads(l); c['pool'] = 'julia'; allc[c['id']] = c
with open(EX + 'candidates.jsonl', 'w') as f:
    for c in allc.values():
        f.write(json.dumps(c) + '\n')
NAME = {'m00071_040_1k060': 'Top Antenna chain', 'm00051_008_2k060': 'Spiral Canopy chain', 'm00051_038_3k060': 'Palm Ring chain',
        'm00041_045_1k060': 'Palm Wheel chain', 'm00041_054_2k060': 'Star chain', 'm00041_045_2k060': 'Palm Wheel II chain',
        'm00041_037_2k060': 'Spiral Chain', 'm00041_006_1k060': 'Spiral Crossing chain', 'm00041_014_3k060': 'Snowflake chain',
        'm00041_002_1k060': 'Elephant Star chain', 'atr091_0148': 'Cranny 148', 'atr091_0238': 'Cranny 238', 'atr091_0090': 'Cranny 90',
        'm00041_084_2k060': 'Amber chain'}
DEPTH = {0.0: 'at the start', 0.125: 'just in', 0.25: 'a quarter of the way down', 0.375: 'three eighths down',
         0.5: 'halfway down', 0.625: 'past halfway', 0.75: 'three quarters down', 0.875: 'almost at the minibrot'}
J = [('m00071_040_1k060', 3), ('m00071_040_1k060', 4), ('m00071_040_1k060', 7), ('m00051_008_2k060', 0), ('m00051_008_2k060', 2),
     ('m00051_008_2k060', 4), ('m00051_038_3k060', 3), ('m00041_045_1k060', 2), ('m00041_054_2k060', 2), ('m00041_045_2k060', 1),
     ('m00041_037_2k060', 2), ('m00041_006_1k060', 1), ('m00041_014_3k060', 1), ('m00041_002_1k060', 4), ('atr091_0148', 3),
     ('atr091_0238', 4), ('atr091_0090', 3), ('m00041_084_2k060', 3)]
picks = []
for sid, j in J:
    c = allc[f'{sid}_j{j}']
    picks.append(dict(id=c['id'], palette='smooth', title=f"Julia opening: {NAME[sid]}, {DEPTH[c['jt']]}",
                      blurb='embedded Julia set on the way down to a minibrot'))
for sid, j in (('m00071_040_1k060', 4), ('m00051_008_2k060', 2)):
    picks.append(dict(id=f'{sid}_j{j}', palette='noctilucent', title=f"Julia opening: {NAME[sid]}, noctilucent",
                      blurb='embedded Julia set'))
for a in ('0090', '0148', '0155', '0232', '0141'):
    picks.append(dict(id=f'atr091_{a}_entrance', palette='smooth', title=f'Cranny {int(a)}: the entrance',
                      blurb='an unremarkable view; a minibrot hides at the bright spot'))
    picks.append(dict(id=f'atr091_{a}_a150', palette='smooth', title=f'Cranny {int(a)}: the atrium',
                      blurb='the minibrot hiding in the entrance view'))
for g in ('0175_f1', '0225_f0', '0167_f0', '0142_f1', '0043_f0', '0130_f0', '0060_f1', '0024_f2'):
    picks.append(dict(id=f'gar093_{g}', palette='canopy', title=f'Hidden Garden {g[:4].lstrip("0")}',
                      blurb='pink-lit garden, canopy trap scale 3'))
json.dump(picks, open(EX + 'picks_j.json', 'w'), indent=0)
print(len(picks))
