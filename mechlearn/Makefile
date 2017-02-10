LIMIT = 100
CHUNKS = 10

NUMBERS := $(shell seq 0 $$(($(CHUNKS) - 1)))
PLIKES := $(addprefix mario_$(LIMIT)_,${NUMBERS})
LIKES := $(addsuffix _$(CHUNKS).like.pkl,${PLIKES})

#all : seg_reg.log

#%.py: %.ipynb
#	jupyter nbconvert $< --to python --ClearOutputPreprocessor.enabled=True

#Illustrative.txt id2sprites.pkl: PPU\ playground.py
#	python $<

#mario_tracks.pkl : Illustrative.txt id2sprites.pkl Tracking\ and\ Filtering.py
#	python Tracking\ and\ Filtering.py

#mario_collisions.pkl : mario_tracks.pkl Collisions.py
#	python Collisions.py

$(LIKES): seg_gen_models.py mario_tracks.pkl Illustrative.txt mario_collisions.pkl
	python seg_gen_models.py $@

seg_reg.log: seg_reg.py mario_tracks.pkl mario_collisions.pkl $(LIKES)
	python seg_reg.py $(LIMIT) $(CHUNKS) > seg_reg.log
