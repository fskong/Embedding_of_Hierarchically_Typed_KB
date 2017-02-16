make

lambda=0.0015
batchsize=1000
dim=100
eta=1
llambda=1
delta=0.0001
type_margin=0.5
epoch=1000
output_per_epoch=100
rellambda=1
relmargin=0.5
reldelta=0.0015
distance=0.01
dataPrefix=../FB15K/

resultPrefix=result


time ./TransH-Type-train -dim $dim -epoch $epoch -batch $batchsize -lr $lambda -margin 0.5 -epsilon 0.01 -beta 0.001 -ll $llambda -del $delta\
	-entity $dataPrefix/entity.txt \
	-rel $dataPrefix/relation.txt \
	-train $dataPrefix/train.txt \
	-type $dataPrefix/type.txt \
	-relationtype $dataPrefix/rel-type-constraint.txt\
	-entitytype $dataPrefix/entitytype.txt \
	-bias_out $resultPrefix/bias2vec \
	-entity_out $resultPrefix/entity2vec \
	-normal_out $resultPrefix/normal2vec \
	-a_out $resultPrefix/tranf2vec \
	-ae_out $resultPrefix/ae2vec \
	-de_out $resultPrefix/de2vec \
	-eta $eta \
	-type_margin $type_margin\
	-output_per_epoch $output_per_epoch\
	-relmargin $relmargin \
	-distance $distance \
	-rellambda $rellambda\
	-reldelta $reldelta

bash test-in-parallel.sh &
