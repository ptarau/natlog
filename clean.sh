rm -r -f dist build
rm -r -f *.egg-info
pushd .
cd natlog
rm -r -f __pycache__
cd test
rm -r -f __pycache__
popd
pushd .
cd natlog
cd app
rm -r -f __pycache__
popd
