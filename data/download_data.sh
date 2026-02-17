#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw
cd data/raw

# MovieLens 100K (GroupLens official)
# Fonte: https://grouplens.org/datasets/movielens/100k/
URL="https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ZIP="ml-100k.zip"

if [ ! -f "$ZIP" ]; then
  echo "Downloading MovieLens 100K..."
  curl -L "$URL" -o "$ZIP"
else
  echo "MovieLens zip already exists: $ZIP"
fi

if [ ! -d "ml-100k" ]; then
  echo "Unzipping..."
  unzip -o "$ZIP" -d ml-100k
else
  echo "MovieLens already unzipped: data/raw/ml-100k"
fi

echo "Done."
