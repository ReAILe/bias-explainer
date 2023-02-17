rm -rf __pycache__/
rm .DS_Store
rm */*.sdimacs*
rm data/raw/repaired* data/raw/reduced*
rm temp*
rm qelim.res
find . -type d -name  "__pycache__" -exec rm -r {} +
rm tests/*.csv
rm ./*.sdimacs*
rm ./*.fr*
rm Fairsquare/src/*.fr*
rm verifair/python/verifair/main/*temp*
rm -r .ipynb_checkpoints
rm data/model.wcnf
rm data/model_out.txt