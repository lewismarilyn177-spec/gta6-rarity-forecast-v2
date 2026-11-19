# gta6-rarity-forecast-v2

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/lewismarilyn177-spec/gta6-rarity-forecast-v2/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/lewismarilyn177-spec/gta6-rarity-forecast-v2/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

GTA VI trophy rarity forecast entry for the make.fun bounty
[gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01): forecast, for
every GTA VI PlayStation 5 trophy, five quantiles of its Exophase earn rate at 2026-11-26T00:00:00Z.

`main.ts` reads each case (makefun/gta6-trophies-case/v1), builds per-trophy distinct-level series in
log space, fits a recent log-log slope per trophy, pools the slopes with shrinkage, projects the
center with a tanh-capped move, and writes logistic quantiles with a horizon-scaled spread.

Run: `mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node --experimental-strip-types main.ts`
