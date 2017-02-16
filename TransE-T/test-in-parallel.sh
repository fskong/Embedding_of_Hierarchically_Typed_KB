#!/usr/bin/env bash

init(){
	splitDir=$1
	midDir=$2
	
	if [ ! -d $midDir ]; then
		#Tip: The bracket expression here is equivalent to "test -d" command, which checks if
		#	the corresponding directory exists. Type "man test" to see more info. 
		mkdir $midDir
	fi
	
	if [ ! -d $splitDir ]; then
		mkdir $splitDir
	fi
	
	rm $splitDir/* $midDir/*
}

splitTest(){
	testFile=$1
	splitNum=$2
	splitDir=$3
	
	totalSize=$(cat $testFile | wc -l)
	splitSize=$(($totalSize / $splitNum))
	echo Split the test file into $splitNum portions, with $splitSize instances in each split. 

	split -l $splitSize $testFile  $splitDir/split-
}

launchThreads(){
	dataDir=$1
	resultDir=$2
	logDir=$3
	splitDir=$4
	midDir=$5 

	inputSps=$(ls -1 $splitDir)
	
	for split in $inputSps
	do
		bash one_test.sh $dataDir $resultDir $splitDir/$split $logDir/detail.direct.without.raw.* > $midDir/$split &
	done
	wait
}

sumUp(){
	midDir=$1
	rm $midDir/sumup.txt
	
	midSps=$(ls -1 $midDir)
	for split in $midSps
	do
		str=$(tail -n 2 $midDir/$split)
		echo ${str/"hit@10"/""} | grep -Eo '[0-9.]+' >> $midDir/sumup.txt
	done
	
	g++ -o sum_up_rank sum_up_rank.cpp
	./sum_up_rank $midDir/sumup.txt > final_result.txt
}

#Arguments:
splitNum=8
dataDir=../Del-FB15K
splitDir=inputSplits
midDir=mid
resultDir=result
logDir=testing_log


cd $(dirname $0)
init $splitDir $midDir
splitTest $dataDir/test.txt $splitNum $splitDir
launchThreads $dataDir $resultDir $logDir $splitDir $midDir
sumUp $midDir
