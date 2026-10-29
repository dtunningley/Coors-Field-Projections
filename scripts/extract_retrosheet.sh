#!/usr/bin/env bash
# Download Retrosheet event files (2012-2025) and Chadwick tools, then extract
# one row per plate appearance (ev/) and one row per game with its park (gm/).
set -euo pipefail
DATA="${DATA:-$HOME/data}"
mkdir -p "$DATA"/ev "$DATA"/gm && cd "$DATA"

# Chadwick command-line tools
[ -d chadwick ] || git clone --depth 1 https://github.com/chadwickbureau/chadwick
(cd chadwick && autoreconf -i && ./configure --prefix="$DATA/cw" && make -j8 && make install)
export LD_LIBRARY_PATH="$DATA/cw/lib"

# Retrosheet: sparse checkout of regular-season event and roster files only
if [ ! -d retrosheet ]; then
  git clone --depth 1 --filter=blob:none --no-checkout https://github.com/chadwickbureau/retrosheet
  cd retrosheet && git sparse-checkout init --no-cone
  { for y in $(seq 2012 2025); do
      echo "/seasons/$y/*.EVA"; echo "/seasons/$y/*.EVN"; echo "/seasons/$y/*.ROS"; echo "/seasons/$y/TEAM$y"
    done; echo "/reference/biofile.csv"; } > .git/info/sparse-checkout
  git checkout HEAD && cd ..
fi

for y in $(seq 2012 2025); do
  cd "$DATA/retrosheet/seasons/$y"
  # fields: game id, visiting team, batting-team flag, batter, event type, PA flag, sac-bunt flag
  "$DATA/cw/bin/cwevent" -y "$y" -f 0,1,3,10,34,35,38 *.EVA *.EVN 2>/dev/null \
    | awk -F, '$6=="\"T\""' > "$DATA/ev/$y.csv"
  # fields: game id, date, park id
  "$DATA/cw/bin/cwgame" -y "$y" -f 0,1,9 *.EVA *.EVN 2>/dev/null > "$DATA/gm/$y.csv"
done
echo "done: $(cat "$DATA"/ev/*.csv | wc -l) plate appearances"
