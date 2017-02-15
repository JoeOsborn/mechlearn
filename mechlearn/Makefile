id2sprites.pkl : PPUPlayground.ipynb movie.fm2 rom.nes
	jupyter nbconvert --to html --execute --ExecutePreprocessor.timeout=-1 PPUPlayground.ipynb

tile2colorized.pkl : id2sprites.pkl
	@

ppu_output.txt :  tile2colorized.pkl
	@

tracks.pkl : ppu_output.txt id2sprites.pkl TrackingAndFiltering.ipynb
	jupyter nbconvert --to html --execute --ExecutePreprocessor.timeout=-1 TrackingAndFiltering.ipynb

collisions.pkl : tracks.pkl Collisions.ipynb
	jupyter nbconvert --to html --execute --ExecutePreprocessor.timeout=-1 Collisions.ipynb

modes.pkl : tracks.pkl movie.fm2 LeastSquares.ipynb
	jupyter nbconvert --to html --execute --ExecutePreprocessor.timeout=-1 LeastSquares.ipynb

edges.pkl : modes.pkl collisions.pkl movie.fm2 tile2colorized.pkl Causes-Adams.ipynb
	jupyter nbconvert --to html --execute --ExecutePreprocessor.timeout=-1 Causes.ipynb

.SECONDARY: 

all : modes.pkl edges.pkl
