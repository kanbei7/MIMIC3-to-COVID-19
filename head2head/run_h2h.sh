#for i in {0..4}
#do
#	date
#	for j in {1..1000}
#	do
#		python3 -W ignore split.py $i
#		python3 -W ignore benchmark.py $i
#		python3 -W ignore allpheno.py $i
#		python3 -W ignore selectedpheno.py $i
#	done
#done

date
for j in {1..1000}
do
	python3 -W ignore split.py 5
	python3 -W ignore benchmark.py 5
	python3 -W ignore allpheno.py 5
	python3 -W ignore selectedpheno.py 5
done
date

date
for j in {1..1000}
do
	python3 -W ignore split.py 0
	python3 -W ignore benchmark.py 0
	python3 -W ignore allpheno.py 0
	python3 -W ignore selectedpheno.py 0
done
date