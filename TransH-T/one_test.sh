dirname $0
cd `dirname $0`

llambda=1
dataPrefix=$1
resultPrefix=$2
splitTest=$3

echo $workPath
./TransH-Type-test \
		$dataPrefix/entity.txt \
		$dataPrefix/relation.txt \
		$dataPrefix/type.txt \
		$dataPrefix/entitytype.txt \
		$resultPrefix/entity2vec \
		$resultPrefix/bias2vec \
		$resultPrefix/normal2vec \
		$resultPrefix/tranf2vec \
		$resultPrefix/ae2vec \
		$resultPrefix/de2vec \
		$splitTest \
		splited_relation \
		$dataPrefix/entity.txt \
		$4 \
		$llambda \
		$dataPrefix/rel-type-constraint.txt \
echo "" 
