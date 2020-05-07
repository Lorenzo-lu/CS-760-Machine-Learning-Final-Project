### Introduction
|  folder   | meaning  |
|  ----  | ----  |
| datset  | Original data set including instruction. |
| CNN  | All CNN related code. If preprocessed dataset is needed, please put them here. |
| RNN  | All RNN related code. If preprocessed dataset is needed, please put them here. |
| doc  | .tex and .sty file only. |


### Result
##### CNN
Please see txt in CNN. For each `XXX_m_n.txt`, `XXX` means how to handle input word embedding, `multi`: use 2 channels, one static and one non static channel; `stat` and `non_stat` means static and non static respectively. `m` means the number of chunks while pooling(corresponding to chunk max pooling). `n` means how many max values are kept during pooling(corresponding to k max pooling).

##### RNN

### How to use

##### CNN
```bash ./test.sh```
Config runtime parameters in script.

##### RNN
