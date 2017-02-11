ROM := mario.nes
MOVIE := Illustrative

# TODO: generate a config file the notebooks can use to get
# paths for things

%.id2sprites.pkl %.txt : PPUPlayground.ipynb %.fm2 $(ROM)
	jupyter nbconvert --to html --execute --ExecutePreprocessor.timeout=-1 PPUPlayground.ipynb

%_tracks.pkl : %.txt %.id2sprites.pkl TrackingAndFiltering.ipynb
	jupyter nbconvert --to html --execute --ExecutePreprocessor.timeout=-1 TrackingAndFiltering.ipynb

%_collisions.pkl : %_tracks.pkl Collisions.ipynb
	jupyter nbconvert --to html --execute --ExecutePreprocessor.timeout=-1 Collisions.ipynb

%_modes.pkl : %_tracks.pkl %.fm2 LeastSquares.ipynb
	jupyter nbconvert --to html --execute --ExecutePreprocessor.timeout=-1 LeastSquares.ipynb

%_edges.pkl : %_modes.pkl %_collisions.pkl %.fm2 Causes.ipynb
	jupyter nbconvert --to html --execute --ExecutePreprocessor.timeout=-1 Causes.ipynb

.SECONDARY: 

all : $(MOVIE)_modes.pkl $(MOVIE)_edges.pkl
