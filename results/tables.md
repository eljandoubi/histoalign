### a1_block: ['V1_Breast_Cancer_Block_A_Section_1'] to V1_Breast_Cancer_Block_A_Section_1
train 2876 spots, test 761 spots, 1008 genes. Mean and standard deviation over seeds.

| variant | method | all genes | top 50 genes | per spot |
|---|---|---|---|---|
| default | aligned_knn | 0.135 ± 0.004 | 0.119 ± 0.005 | 0.141 ± 0.006 |
| default | raw_knn | 0.148 ± 0.000 | 0.154 ± 0.000 | 0.163 ± 0.000 |
| default | ridge | 0.135 ± 0.000 | 0.132 ± 0.000 | 0.167 ± 0.000 |
| default | aligned_knn_shuffled_pairs | 0.021 ± 0.010 | 0.038 ± 0.036 | 0.021 ± 0.011 |
| indrop | aligned_knn | 0.121 ± 0.008 | 0.114 ± 0.005 | 0.119 ± 0.010 |
| indrop | raw_knn | 0.148 ± 0.000 | 0.154 ± 0.000 | 0.163 ± 0.000 |
| indrop | ridge | 0.135 ± 0.000 | 0.132 ± 0.000 | 0.167 ± 0.000 |
| indrop | aligned_knn_shuffled_pairs | 0.030 ± 0.028 | 0.030 ± 0.012 | 0.018 ± 0.027 |
| nbr2 | aligned_knn | 0.135 ± 0.004 | 0.118 ± 0.005 | 0.141 ± 0.006 |
| nbr2 | raw_knn | 0.148 ± 0.000 | 0.154 ± 0.000 | 0.163 ± 0.000 |
| nbr2 | ridge | 0.135 ± 0.000 | 0.132 ± 0.000 | 0.167 ± 0.000 |
| nbr2 | aligned_knn_shuffled_pairs | 0.021 ± 0.010 | 0.038 ± 0.036 | 0.021 ± 0.011 |
| sigmoid | aligned_knn | 0.101 ± 0.009 | 0.090 ± 0.019 | 0.075 ± 0.020 |
| sigmoid | raw_knn | 0.148 ± 0.000 | 0.154 ± 0.000 | 0.163 ± 0.000 |
| sigmoid | ridge | 0.135 ± 0.000 | 0.132 ± 0.000 | 0.167 ± 0.000 |
| sigmoid | aligned_knn_shuffled_pairs | 0.016 ± 0.025 | 0.005 ± 0.027 | 0.014 ± 0.016 |

| variant | epochs | recall@10 | near recall@10 | median rank |
|---|---|---|---|---|
| default | 3.333 ± 0.471 | 0.054 ± 0.001 | 0.278 ± 0.014 | 172.000 ± 8.165 |
| indrop | 5.333 ± 0.471 | 0.042 ± 0.006 | 0.271 ± 0.015 | 174.333 ± 2.357 |
| nbr2 | 3.333 ± 0.471 | 0.055 ± 0.002 | 0.283 ± 0.010 | 171.333 ± 8.179 |
| sigmoid | 3.667 ± 0.471 | 0.037 ± 0.004 | 0.267 ± 0.043 | 181.667 ± 2.494 |
| chance | | 0.022 | 0.167 | 382 |

### a1_to_a2: ['V1_Breast_Cancer_Block_A_Section_1'] to V1_Breast_Cancer_Block_A_Section_2
train 3798 spots, test 3987 spots, 1008 genes. Mean and standard deviation over seeds.

| variant | method | all genes | top 50 genes | per spot |
|---|---|---|---|---|
| default | aligned_knn | 0.140 ± 0.005 | 0.221 ± 0.008 | 0.170 ± 0.007 |
| default | raw_knn | 0.138 ± 0.000 | 0.210 ± 0.000 | 0.174 ± 0.000 |
| default | ridge | 0.137 ± 0.000 | 0.196 ± 0.000 | 0.145 ± 0.000 |
| default | aligned_knn_shuffled_pairs | -0.006 ± 0.006 | -0.001 ± 0.003 | -0.012 ± 0.006 |
| indrop | aligned_knn | 0.131 ± 0.002 | 0.211 ± 0.004 | 0.166 ± 0.001 |
| indrop | raw_knn | 0.138 ± 0.000 | 0.210 ± 0.000 | 0.174 ± 0.000 |
| indrop | ridge | 0.137 ± 0.000 | 0.196 ± 0.000 | 0.145 ± 0.000 |
| indrop | aligned_knn_shuffled_pairs | 0.006 ± 0.020 | -0.003 ± 0.016 | -0.011 ± 0.009 |
| nbr2 | aligned_knn | 0.139 ± 0.005 | 0.220 ± 0.008 | 0.169 ± 0.007 |
| nbr2 | raw_knn | 0.138 ± 0.000 | 0.210 ± 0.000 | 0.174 ± 0.000 |
| nbr2 | ridge | 0.137 ± 0.000 | 0.196 ± 0.000 | 0.145 ± 0.000 |
| nbr2 | aligned_knn_shuffled_pairs | -0.006 ± 0.006 | -0.002 ± 0.004 | -0.012 ± 0.006 |
| sigmoid | aligned_knn | 0.134 ± 0.011 | 0.204 ± 0.029 | 0.159 ± 0.014 |
| sigmoid | raw_knn | 0.138 ± 0.000 | 0.210 ± 0.000 | 0.174 ± 0.000 |
| sigmoid | ridge | 0.137 ± 0.000 | 0.196 ± 0.000 | 0.145 ± 0.000 |
| sigmoid | aligned_knn_shuffled_pairs | 0.014 ± 0.017 | 0.022 ± 0.028 | 0.019 ± 0.037 |

| variant | epochs | recall@10 | near recall@10 | median rank |
|---|---|---|---|---|
| default | 3.667 ± 0.471 | 0.029 ± 0.003 | 0.180 ± 0.014 | 415.667 ± 23.171 |
| indrop | 6.667 ± 0.943 | 0.026 ± 0.002 | 0.157 ± 0.004 | 397.667 ± 14.055 |
| nbr2 | 3.667 ± 0.471 | 0.029 ± 0.004 | 0.182 ± 0.015 | 419.667 ± 23.471 |
| sigmoid | 10.000 ± 5.715 | 0.023 ± 0.007 | 0.162 ± 0.036 | 636.333 ± 66.595 |
| chance | | 0.003 | 0.029 | 1999 |

### a2_to_a1: ['V1_Breast_Cancer_Block_A_Section_2'] to V1_Breast_Cancer_Block_A_Section_1
train 3987 spots, test 3798 spots, 1008 genes. Mean and standard deviation over seeds.

| variant | method | all genes | top 50 genes | per spot |
|---|---|---|---|---|
| default | aligned_knn | 0.134 ± 0.013 | 0.164 ± 0.025 | 0.173 ± 0.009 |
| default | raw_knn | 0.159 ± 0.000 | 0.172 ± 0.000 | 0.183 ± 0.000 |
| default | ridge | 0.134 ± 0.000 | 0.122 ± 0.000 | 0.143 ± 0.000 |
| default | aligned_knn_shuffled_pairs | -0.013 ± 0.007 | -0.035 ± 0.007 | -0.008 ± 0.016 |
| indrop | aligned_knn | 0.129 ± 0.004 | 0.163 ± 0.000 | 0.169 ± 0.004 |
| indrop | raw_knn | 0.159 ± 0.000 | 0.172 ± 0.000 | 0.183 ± 0.000 |
| indrop | ridge | 0.134 ± 0.000 | 0.122 ± 0.000 | 0.143 ± 0.000 |
| indrop | aligned_knn_shuffled_pairs | -0.025 ± 0.021 | -0.028 ± 0.035 | -0.033 ± 0.012 |
| nbr2 | aligned_knn | 0.132 ± 0.013 | 0.155 ± 0.024 | 0.173 ± 0.009 |
| nbr2 | raw_knn | 0.159 ± 0.000 | 0.172 ± 0.000 | 0.183 ± 0.000 |
| nbr2 | ridge | 0.134 ± 0.000 | 0.122 ± 0.000 | 0.143 ± 0.000 |
| nbr2 | aligned_knn_shuffled_pairs | -0.008 ± 0.015 | -0.007 ± 0.036 | 0.003 ± 0.017 |
| sigmoid | aligned_knn | 0.118 ± 0.015 | 0.143 ± 0.016 | 0.120 ± 0.047 |
| sigmoid | raw_knn | 0.159 ± 0.000 | 0.172 ± 0.000 | 0.183 ± 0.000 |
| sigmoid | ridge | 0.134 ± 0.000 | 0.122 ± 0.000 | 0.143 ± 0.000 |
| sigmoid | aligned_knn_shuffled_pairs | 0.009 ± 0.022 | -0.000 ± 0.030 | -0.002 ± 0.027 |

| variant | epochs | recall@10 | near recall@10 | median rank |
|---|---|---|---|---|
| default | 1.333 ± 0.471 | 0.019 ± 0.007 | 0.135 ± 0.024 | 634.500 ± 114.668 |
| indrop | 3.333 ± 0.471 | 0.026 ± 0.001 | 0.166 ± 0.005 | 420.167 ± 26.288 |
| nbr2 | 1.667 ± 0.471 | 0.023 ± 0.006 | 0.150 ± 0.022 | 535.000 ± 93.964 |
| sigmoid | 1.667 ± 0.943 | 0.014 ± 0.010 | 0.108 ± 0.050 | 961.500 ± 398.055 |
| chance | | 0.003 | 0.031 | 1938 |

