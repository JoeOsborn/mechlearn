ROM := mario.nes
MOVIE := Illustrative.fm2
MOVIE_BN := $(basename $(MOVIE))

LIMIT := 100
CHUNKS := 10

NUMBERS := $(shell seq 0 $$(($(CHUNKS) - 1)))
PLIKES := $(addprefix mario_$(LIMIT)_,${NUMBERS})
LIKES := $(addsuffix _$(CHUNKS).like.pkl,${PLIKES})

all : seg_reg.log

# TODO: change output file to a fixed filename so we don't
# have to pass variables into the notebooks. Alternately,
# generate a config file the notebooks can use to get
# paths for things

#.PHONY: sprite_data
#$(MOVIE_BN).txt id2sprites.pkl : sprite_data 

# HACK: This also generates $(MOVIE_BN).txt but I don't want to
# use a fake target and have this generated every time.
id2sprites.pkl : PPU\ playground.ipynb $(MOVIE) $(ROM)
	jupyter nbconvert --to html --execute --ExecutePreprocessor.timeout=-1 "PPU Playground.ipynb"

mario_tracks.pkl : $(MOVIE_BN).txt id2sprites.pkl Tracking\ and\ Filtering.ipynb
	jupyter nbconvert --to html --execute --ExecutePreprocessor.timeout=-1 "Tracking and Filtering.ipynb"

mario_collisions.pkl : mario_tracks.pkl Collisions.ipynb
	jupyter nbconvert --to html --execute --ExecutePreprocessor.timeout=-1 "Collisions.ipynb"

$(LIKES): seg_gen_models.py mario_tracks.pkl $(MOVIE_BN).txt mario_collisions.pkl
	python seg_gen_models.py "$@"

.PHONY: likes
likes: $(LIKES)

seg_reg.log: seg_reg.py mario_tracks.pkl mario_collisions.pkl $(LIKES)
	python seg_reg.py $(LIMIT) $(CHUNKS) > seg_reg.log
