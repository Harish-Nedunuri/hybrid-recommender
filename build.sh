python3 setup.py clean --all
rm -rf build
rm -rf dist
python3 setup.py bdist_egg
pip3 uninstall -y hybrid_recommender
python3 -m pip install .

