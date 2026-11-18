
for file in $(ls ./src/xai/*.py)
do 
    ./run_lint.sh $file -e
done