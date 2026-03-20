
# Globe generation pipeline — calls the 2 live scripts.
# Generate Atlases
python scripts/render_polygrids.py -f 3 --seed 42 -o exports/f3-1 --detail-rings 1
python scripts/render_polygrids.py -f 3 --seed 42 -o exports/f3-2 --detail-rings 2
python scripts/render_polygrids.py -f 3 --seed 42 -o exports/f3-3 --detail-rings 3
python scripts/render_polygrids.py -f 3 --seed 42 -o exports/f3-4 --detail-rings 4
python scripts/render_polygrids.py -f 3 --seed 42 -o exports/f3-5 --detail-rings 5

# Generate Debug Atlases
python scripts/render_polygrids.py -f 3 --seed 42 -o exports/f3-1-colour --detail-rings 1 --colour-debug --debug-labels
python scripts/render_polygrids.py -f 3 --seed 42 -o exports/f3-2-colour --detail-rings 2 --colour-debug --debug-labels
python scripts/render_polygrids.py -f 3 --seed 42 -o exports/f3-3-colour --detail-rings 3 --colour-debug --debug-labels
python scripts/render_polygrids.py -f 3 --seed 42 -o exports/f3-4-colour --detail-rings 4 --colour-debug --debug-labels
python scripts/render_polygrids.py -f 3 --seed 42 -o exports/f3-5-colour --detail-rings 5 --colour-debug --debug-labels

# Render Globes
python scripts/render_globe_from_tiles.py exports/f3-1
python scripts/render_globe_from_tiles.py exports/f3-2
python scripts/render_globe_from_tiles.py exports/f3-3
python scripts/render_globe_from_tiles.py exports/f3-4
python scripts/render_globe_from_tiles.py exports/f3-5

# Render Debug Globes
python scripts/render_globe_from_tiles.py exports/f3-1-colour
python scripts/render_globe_from_tiles.py exports/f3-2-colour
python scripts/render_globe_from_tiles.py exports/f3-3-colour
python scripts/render_globe_from_tiles.py exports/f3-4-colour
python scripts/render_globe_from_tiles.py exports/f3-5-colour

python scripts/render_globe_from_tiles.py exports/debug_labels_test