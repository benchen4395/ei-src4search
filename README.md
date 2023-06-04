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
  <img width="500" height="171" src="https://github.com/benchen4395/ei-src4search/blob/main/model_implement_structures/model_structure.png">
</p>
<h5 align="center">
The overview of the ei-SRC, the proposed e-commerce interaction-based semantic relevance calculation method.
</h5>

#### Content in it:

- 1\. 10 pairs annotated testing data cases (random sampled from 100k query-item pairs, and text descriptions of items are the given titles).

- 2\. Key code implementations of the following parts, and corresponding description of corresponding config.
    - **D**ynamic-length **R**epresentation **S**cheme [(__DRS__)](https://github.com/benchen4395/ei-src4search/blob/main/model_implement_details/DRS.py)
    - **C**ontrastive **A**dversarial **T**raining mechanism [(__CAT__)](https://github.com/benchen4395/ei-src4search/blob/main/model_implement_details/CAT.py)

- 3\. More testing results of offline experiments, which contains the ablation expermient results of ***BERT_mini*** with different: 
    - number of layers, self-attention heads, and activation function
    - dropout rate of R-Drop
    - loss weighting parameters of the loss function
