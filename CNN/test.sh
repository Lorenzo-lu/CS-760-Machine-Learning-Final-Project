python CNN.py --embedding_strategy multi --pooling_chunk 1 --k_max 1 > multi_1_1.txt
python CNN.py --embedding_strategy multi --pooling_chunk 1 --k_max 2 > multi_1_2.txt
python CNN.py --embedding_strategy multi --pooling_chunk 1 --k_max 3 > multi_1_3.txt
python CNN.py --embedding_strategy multi --pooling_chunk 1 --k_max 4 > multi_1_4.txt

python CNN.py --embedding_strategy multi --pooling_chunk 2 --k_max 1 > multi_2_1.txt
python CNN.py --embedding_strategy multi --pooling_chunk 3 --k_max 1 > multi_3_1.txt
python CNN.py --embedding_strategy multi --pooling_chunk 4 --k_max 1 > multi_4_1.txt

python CNN.py --embedding_strategy static --pooling_chunk 1 --k_max 1 > static_1_1.txt
python CNN.py --embedding_strategy non_static --pooling_chunk 1 --k_max 1 > non_stat_1_1.txt
