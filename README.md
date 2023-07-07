# ei-src4search
Corresponding codes for paper `Robust Interaction-based Relevance Modeling for Online E-Commerce Search`

## abstract 
`Semantic relevance calculation` plays a vital role in e-commerce search engine, as it aims to select the most suitable items to match customers' intents. Neglecting this aspect can have detrimental effects on user experience and engagement. In this paper we present a robust interaction-based modeling paradigm. It encompasses:

1) a dynamic length representation scheme to reduce inference time

2) a professional terms recognition method to identify subjects and core attributes from words stacked sentences

3) a contrastive adversarial training mechanism to enhance the robustness for representing and matching

Extensive offline evaluations demonstrate the superior robustness and effectiveness of our approach, and online A/B testing confirms its ability to improve relevance in the same exposure position, resulting in more clicks and conversions. 

To the best of our knowledge, this method is ***the first interaction-based approach for large e-commerce search relevance calculation***. Notably, we have deployed it for the entire search traffic on [alibaba.com](www.alibaba.com), the largest B2B e-commerce platform in the world.

<p align="center">
  <img width="740.6" height="364" src="https://github.com/benchen4395/ei-src4search/blob/main/model_implement_structures/overall_model_structure.jpg">
</p>
<h5 align="center">
The overview of the ei-SRC, the proposed e-commerce interaction-based semantic relevance calculation method.
</h5>

## Content in it:

- 1\. 20 pairs annotated [testing data cases](https://github.com/benchen4395/ei-src4search/blob/main/query_title_testdata_cases/20_query_title_pairs.csv) (random sampled from 100k query-item pairs, and text descriptions of items are the given titles).

- 2\. Key code implementations of the following parts, and corresponding description of corresponding config.
    - **D**ynamic-length **R**epresentation **S**cheme [(__DRS__)](https://github.com/benchen4395/ei-src4search/blob/main/model_implement_details/DRS.py)
    - **C**ontrastive **A**dversarial **T**raining mechanism [(__CAT__)](https://github.com/benchen4395/ei-src4search/blob/main/model_implement_details/CAT.py)

- 3\. More testing results of offline experiments, which contains the ablation expermient results of ***BERT_mini*** with different: 
    - number of layers, self-attention heads, and activation function
    - dropout rate of R-Drop
    - loss weighting parameters of the loss function

#### As a side note:
All data, code implements may be released to public through [official repository](https://github.com/alibaba), as well as more testing results of offline experiments.

## Offline Testings
We take AUC, Micro / Macro F1, Spearmanr and Pearsonr as the evaluation metrics, and list all results of base ($X_{base}$) and mini ($X_{mini}$) version for representation / interaction based models respectively.

- Table 1. Improvements in computational performance for each strategy combination of **ei-SRC**

|Method|GPU utilization|reaction latency|ROC AUC|
|---|:---:|:---:|:---:|
|DRS|-34.61\%|-36.26\%|+0.4\%|
|$\backslash$+Cache|-48.08\%|-40.66\%|-|
|$\backslash$+Vocab.|-53.84\%|-46.15\%|+1.5\%|
|$\backslash$+CAT|-51.25\%|-44.83\%|+6.3\%|

- Table 2. Comparison with representation and interaction-based methods on manual annotated data

|Strategy|AUC|Macro F1|Spearmanr|Pearsonr|
|---|:---:|:---:|:---:|:---:|
|$sBERT_{mini}$|0.8184|0.61/0.71|0.5515|0.3934|
|$sBERT_{base}$|0.8554|0.60/0.76|0.6061|0.4751|
|$BERT_{mini}$|0.8500|0.65/0.77|0.6147|0.5079|
|$BERT_{base}$|0.8907|0.72/0.78|0.6742|0.6360|
|$RoBERTa_{base}$|0.8964|0.78/0.82|0.6865|0.6926|
|$StructBERT_{base}$|0.9011|0.78/0.81|0.6947|0.7096|
|ei-SRC|**0.9033**|**0.81/0.83**|**0.6984**|**0.7181**|
