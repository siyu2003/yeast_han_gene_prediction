# Cross-Species Inductive Heterogeneous Graph ML for Alopecia Candidate Discovery

## 0. Project Summary

본 프로젝트는 **맥주효모(Saccharomyces cerevisiae) 유래 유전자/대사산물**이 인간의 **남성형 탈모 관련 표적 단백질(SRD5A1, SRD5A2, AR 등)** 과 어떤 방식으로 연결될 수 있는지를 **이종 그래프(Heterogeneous Graph) 기반 Link Prediction**으로 탐색하는 학교 과제용 proof-of-concept 프로젝트이다.

```text
Yeast Gene
  → Yeast Metabolite
    → Structurally Similar Human Drug
      → Human Target Protein
        → Alopecia Disease
```

즉, 효모 유전자와 인간 탈모 질환 사이에 직접적인 실험 데이터가 없더라도,  
**효모 유전자가 만드는 대사산물**과 **인간 탈모 관련 약물** 사이의 **화학 구조 유사도**를 latent edge로 추가하여 cross-species graph connectivity를 만든다.

이 프로젝트의 최종 목표는 다음과 같다.

```text
1. Yeast gene / metabolite / human drug / human target / disease로 이루어진 heterogeneous graph 구성
2. RDKit Morgan fingerprint 기반 Tanimoto similarity로 metabolite-drug latent edge 생성
3. Inductive learning이 가능한 HAN 기반 link prediction 모델 구현
4. 숨겨둔 edge를 test set으로 평가
5. 학습된 representation을 이용해 yeast gene 또는 unseen compound의 alopecia relevance score 예측
```

> 중요한 주의: 이 프로젝트는 실제 치료 효과를 증명하는 wet-lab 연구가 아니라, 제한된 curated dataset을 이용한 **in silico proof-of-concept**이다. 결과는 “후보 우선순위화(candidate prioritization)”로 해석해야 한다.

---

## 1. Problem

> 맥주효모의 유전자와 대사산물 중, 인간의 탈모 관련 표적 단백질과 graph-based path를 통해 연결될 가능성이 높은 후보는 무엇인가?

---

## 2. Model

본 프로젝트의 핵심 contribution은 다음과 같다.

### 2.1 Cross-species graph construction

서로 다른 생물종의 데이터를 하나의 graph 안에 연결한다.

```text
Yeast side:
- Yeast gene
- Yeast metabolite

Human side:
- Human drug
- Human target protein
- Human disease
```

### 2.2 Latent chemical similarity edge

Yeast metabolite와 human drug 사이에는 직접적인 biological edge가 없다.  
따라서 RDKit 기반 화학 구조 유사도를 이용해 다음 edge를 계산한다.

```text
(Metabolite, similar_to, Drug)
```

이는 데이터 단절을 극복하기 위한 latent edge이다.

### 2.3 Inductive heterogeneous GNN

node feature 기반 model로 바꾸어 **inductive learning이 가능하도록 설계**한다.

```text
Transductive version:
node_id → nn.Embedding → HANConv

Inductive version:
node feature → Linear encoder → HANConv
```

즉, 새로운 metabolite나 drug가 들어와도 SMILES로부터 fingerprint를 만들 수 있으므로, 학습된 model parameter를 적용할 수 있다.

---

## 3. Graph Schema

### 3.1 Node Types

본 프로젝트의 heterogeneous graph는 총 5종류의 node type을 가진다.

```text
1. gene(??)
   - Yeast gene
   - 예: ERG11, ERG7, HMG1, ERG20

2. metabolite(1024)
   - Yeast metabolite 또는 sterol/isoprenoid-related compound
   - 예: lanosterol, ergosterol, squalene, testosterone

3. drug(1200)
   - Human drug
   - 예: finasteride, dutasteride, spironolactone, ketoconazole

4. target(20/50)
   - Human target protein
   - 예: SRD5A1, SRD5A2, AR, CYP51A1

5. disease(1)
   - Human disease
   - 본 프로젝트에서는 ALOPECIA 하나만 사용
```

---

### 3.2 Edge Types

총 4개의 biological / latent relation을 사용한다.

```text
1. (gene, produces, metabolite)
   - Yeast gene이 특정 metabolite와 pathway상 연결됨
   - curated edge

2. (metabolite, similar_to, drug)
   - RDKit Morgan fingerprint 기반 Tanimoto similarity로 계산
   - latent chemical similarity edge
   - 각 metabolite당 Top-K drug만 연결

3. (drug, inhibits, target)
   - human drug가 target protein을 억제한다고 가정한 curated edge
   - 과제용 confidence score 포함

4. (target, associated_with, disease)
   - target protein이 alopecia와 관련됨
   - association score 포함
```

message passing을 위해 모든 edge에는 reverse edge를 추가한다.

```text
(gene, produces, metabolite)
(metabolite, rev_produces, gene)

(metabolite, similar_to, drug)
(drug, rev_similar_to, metabolite)

(drug, inhibits, target)
(target, rev_inhibits, drug)

(target, associated_with, disease)
(disease, rev_associated_with, target)
```

reverse edge는 반대 작용을 위함이 아니다. undirected graph를 구현하기 위함이다.

---

## 4. Data Design

### 4.1 Yeast Gene → Metabolite Curated Data


### 5.2 Human Drug → Target Curated Data



### 5.3 Target → Disease Curated Data



## 6. Chemical Feature Engineering

### 6.1 SMILES Fetching

대형 database를 다운로드하지 않고, compound name을 PubChem PUG-REST로 검색하여 SMILES를 가져온다.

```text
Input:
compound name

Output:
CanonicalSMILES 또는 IsomericSMILES
```

예:

```text
compound name: finasteride
SMILES: PubChem에서 가져온 canonical/isomeric SMILES
```

SMILES를 가져오지 못한 compound는 제거한다.

---

### 6.2 Morgan Fingerprint

RDKit의 Morgan fingerprint를 사용한다.

```text
radius = 2
fpSize = 2048
```

각 molecule은 2048차원 binary fingerprint로 변환된다.

```text
SMILES → RDKit Mol → Morgan fingerprint → 2048-dimensional vector
```

---

### 6.3 Tanimoto Similarity

Metabolite와 drug의 fingerprint 사이에 Tanimoto similarity를 계산한다.

```text
Tanimoto(A, B) = |A ∩ B| / |A ∪ B|
```

각 metabolite마다 similarity가 높은 Top-K drug만 edge로 연결한다.

```text
TOP_K = 5
```

결과 edge:

```text
(metabolite, similar_to, drug)
```

예:

```text
lanosterol     similar_to  finasteride
ergosterol     similar_to  spironolactone
testosterone   similar_to  dutasteride
```

Tanimoto 값이 매우 높지 않아도 본 프로젝트에서는 latent weak edge로 사용한다.  
목적은 “완전히 같은 구조”를 찾는 것이 아니라, graph connectivity를 확보하는 것이다.

---

## 7. Inductive Node Feature Design

이 프로젝트에서 inductive learning을 가능하게 하는 핵심은 **node ID embedding을 사용하지 않는 것**이다.

### 7.1 Metabolite Node Feature

```text
metabolite.x = Morgan fingerprint
dimension = 2048
```

새 metabolite가 들어와도 SMILES만 있으면 동일한 방식으로 feature를 만들 수 있다.

---

### 7.2 Drug Node Feature

```text
drug.x = Morgan fingerprint
dimension = 2048
```

새 drug가 들어와도 SMILES만 있으면 feature를 만들 수 있다.

---

### 7.3 Gene Node Feature



### 7.4 Target Node Feature



### 7.5 Disease Node Feature


## 8. Model Architecture

### 8.1 Why HAN?

이 프로젝트는 node type과 edge type이 여러 개인 heterogeneous graph이다.  
따라서 homogeneous GCN보다 heterogeneity를 처리할 수 있는 HAN을 사용한다.

```text
Model:
Inductive Heterogeneous Graph Attention Network
```

사용 layer:

```text
torch_geometric.nn.HANConv
```

---

### 8.2 Overall Architecture

```text
Input HeteroData
  ├── gene.x
  ├── metabolite.x
  ├── drug.x
  ├── target.x
  └── disease.x

Node-type-specific Linear Projection
  ├── gene:       Linear(input_dim, hidden_dim)
  ├── metabolite: Linear(2048, hidden_dim)
  ├── drug:       Linear(2048, hidden_dim)
  ├── target:     Linear(input_dim, hidden_dim)
  └── disease:    Linear(input_dim, hidden_dim)

HANConv Layer 1
  → heterogeneous message passing

HANConv Layer 2
  → final node representation

Dot Product Decoder
  → link score
```

---

### 8.3 Input Projection

각 node type은 feature dimension이 다르다.

```text
gene.x       = 6
metabolite.x = 2048
drug.x       = 2048
target.x     = 6
disease.x    = 6
```

따라서 node type별로 Linear projection을 사용한다.

```python
self.input_proj = nn.ModuleDict({
    node_type: nn.Linear(in_dim, hidden_channels)
    for node_type, in_dim in in_channels_dict.items()
})
```

### 8.4 HAN Layers

```python
self.conv1 = HANConv(
    in_channels=hidden_channels,
    out_channels=hidden_channels,
    metadata=metadata,
    heads=4,
    dropout=0.1,
)

self.conv2 = HANConv(
    in_channels=hidden_channels,
    out_channels=out_channels,
    metadata=metadata,
    heads=4,
    dropout=0.1,
)
```

hidden dimension은 64를 사용한다.

```text
hidden_channels = 64
out_channels = 64
attention heads = 4
dropout = 0.1
```

---

### 8.5 Link Prediction Decoder

두 node embedding의 dot product를 사용한다.

```text
score(u, v) = sigmoid(z_u · z_v)S
```

PyTorch code:

```python
src_z = z_dict[src_type][edge_label_index[0]]
dst_z = z_dict[dst_type][edge_label_index[1]]
logits = (src_z * dst_z).sum(dim=-1)
```

BCEWithLogitsLoss를 사용하므로 training 중에는 sigmoid 이전의 logit을 loss에 넣는다.

---

## 9. Training Strategy

### 9.1 Link Prediction Tasks

다음 두 edge type을 학습 task로 사용한다.

```text
1. (metabolite, similar_to, drug)
2. (drug, inhibits, target)
```

즉, 모델은 다음을 학습한다.

```text
- 어떤 metabolite-drug pair가 구조적으로 유사한가?
- 어떤 drug-target pair가 inhibitory relation을 가지는가?
```

`gene → disease`는 직접 학습하지 않는다.  
학습 후 downstream prediction으로 사용한다.

---

### 9.2 RandomLinkSplit

PyTorch Geometric의 `RandomLinkSplit`으로 train/validation/test를 나눈다.

```text
validation ratio = 0.2
test ratio = 0.2
negative sampling ratio = 1.0
```

즉, known positive edge 일부를 숨기고 test set으로 사용한다.

---

### 9.3 Positive and Negative Samples

Positive sample:

```text
실제 존재하는 edge
예: finasteride → SRD5A2
```

Negative sample:

```text
graph에 존재하지 않는 random pair
예: finasteride → ESR1
```

이렇게 만들어진 label:

```text
edge exists     → label = 1
edge not exists → label = 0
```

---

### 9.4 Loss Function

```text
Binary Cross Entropy with Logits Loss
```

목표:

```text
positive edge score → high
negative edge score → low
```

---

### 9.5 Optimizer

```text
Adam
learning rate = 0.01
weight_decay = 1e-4
epochs = 120
```

---

## 10. Evaluation

### 10.1 Test Data Meaning

`test_data`는 새로운 gene-disease pair를 test하는 것이 아니다.  
`RandomLinkSplit`이 숨겨둔 known edge를 복원할 수 있는지를 보는 것이다.

평가 대상:

```text
1. metabolite-drug similarity edge 복원
2. drug-target inhibition edge 복원
```

---

### 10.2 AUC

AUC는 positive edge가 negative edge보다 더 높은 score를 받는지 평가한다.

```text
AUC 질문:
실제 edge가 가짜 edge보다 높은 점수를 받는가?
```

---

### 10.3 Average Precision

AP는 positive edge를 ranking 상위에 잘 올리는지를 본다.

```text
AP 질문:
모델이 실제 edge를 상위 후보로 잘 랭킹하는가?
```

---

### 10.4 Recall@K

Recall@K는 하나의 source node에 대해 가능한 destination node 전체를 ranking하고, 실제 target이 Top-K 안에 들어오는지 본다.

예:

```text
source = finasteride

candidate targets:
- SRD5A1
- SRD5A2
- AR
- CYP51A1
- HMGCR
- ESR1

true target = SRD5A2

SRD5A2가 Top-K 안에 있으면 hit
```

---

## 11. Discovery Prediction

### 11.1 Existing Gene → Alopecia Ranking

학습 후에는 직접 학습하지 않은 relation을 예측한다.

```text
(gene, ?, disease)
```

구체적으로:

```text
score(gene, ALOPECIA) = sigmoid(z_gene · z_ALOPECIA)
```

이 값이 높은 gene을 candidate로 ranking한다.

해석:

```text
높은 score = graph embedding space에서 alopecia node와 가까움
```

주의:

```text
이 score는 실제 치료 효과가 아니라 graph-based relevance score이다.
```

---

### 11.2 Path-based Explanation

GNN score만 보고하면 생물학적 설명력이 약하다.  
따라서 각 gene에 대해 다음 경로를 같이 출력한다.

```text
Gene
  → Metabolite
    → Similar Drug
      → Target
        → ALOPECIA
```

path score는 다음처럼 계산할 수 있다.

```text
path_score =
Tanimoto similarity
× drug-target confidence
× target-disease association score
```

최종 ranking은 GNN score와 path score를 함께 사용할 수 있다.

```text
final_score =
0.5 × GNN score
+ 0.5 × scaled path score
```

---

## 12. Inductive Inference

이 프로젝트의 최종 요구사항은 **unseen node에 대해서도 prediction이 가능해야 한다**는 것이다.

이를 위해 모델은 node ID embedding이 아니라 feature encoder를 사용한다.

---

### 12.1 Unseen Metabolite

새로운 metabolite가 들어왔을 때 필요한 정보:

```text
- compound name
- SMILES
```

처리 과정:

```text
1. PubChem에서 SMILES 가져오기
2. RDKit Morgan fingerprint 생성
3. metabolite.x에 새 node feature 추가
4. 기존 drug들과 Tanimoto similarity 계산
5. Top-K drug와 similarity edge 추가
6. 재학습 없이 trained HAN parameter로 message passing
7. unseen metabolite와 ALOPECIA node의 score 계산
```

중요:

```text
재학습 없음
model.eval() 상태에서 inference
```

---

### 12.2 Unseen Drug

새로운 drug가 들어왔을 때 필요한 정보:

```text
- drug name
- SMILES
```

처리 과정:

```text
1. PubChem에서 SMILES 가져오기
2. RDKit Morgan fingerprint 생성
3. drug.x에 새 node feature 추가
4. 기존 metabolite들과 Tanimoto similarity 계산
5. Top-K metabolite와 similarity edge 추가
6. 재학습 없이 trained HAN parameter로 message passing
7. unseen drug와 기존 target node들의 score 계산
```

예측 결과:

```text
unseen drug → predicted target ranking
```

---

### 12.3 Unseen Gene

새로운 gene은 화학 구조가 없으므로, 연결 metabolite 정보가 필요하다.

필요한 정보:

```text
- new gene name
- new gene이 연결될 metabolite list
```

처리 과정:

```text
1. new gene이 연결된 metabolite들의 fingerprint 수집
2. 평균 fingerprint로 gene feature 생성
3. gene.x에 새 node feature 추가
4. new gene → metabolite edge 추가
5. 재학습 없이 trained HAN parameter로 message passing
6. unseen gene과 ALOPECIA node의 score 계산
```

조건:

```text
unseen gene이 아무 metabolite와도 연결되지 않으면 의미 있는 prediction이 어렵다.
```

---

## 13. Why This Model Is Inductive

본 모델이 inductive한 이유:

```text
- node ID embedding을 사용하지 않음
- node feature를 입력으로 사용
- 같은 Linear projection과 HANConv parameter를 새로운 node에도 적용 가능
- 새 node가 graph에 연결되면 message passing으로 representation 생성 가능
```

비교:

```text
Transductive model:
new node index가 embedding table에 없으므로 직접 예측 불가

Inductive model:
new node feature가 있으면 같은 parameter를 적용 가능
```