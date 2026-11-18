
for file in $(ls ./src/*.py)
do 
    ./run_lint.sh $file -e
done