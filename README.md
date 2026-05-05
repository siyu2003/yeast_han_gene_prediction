# Cross-Species Inductive Heterogeneous Graph ML for Alopecia Candidate Discovery

## 0. Project Summary

본 프로젝트는 **맥주효모(Saccharomyces cerevisiae) 유래 유전자/대사산물**이 인간의 **남성형 탈모 관련 표적 단백질(SRD5A1, SRD5A2, AR 등)** 과 어떤 방식으로 연결될 수 있는지를 **이종 그래프(Heterogeneous Graph) 기반 Link Prediction**으로 탐색하는 학교 과제용 proof-of-concept 프로젝트이다.

핵심 아이디어는 다음과 같다.

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

## 1. Original Biological Motivation

### 1.1 Problem

탈모 시장에서 맥주효모는 영양 보조제 또는 마케팅 용어로 널리 소비되고 있다.  
하지만 남성형 탈모의 핵심 생물학적 축인 다음 표적들에 대해 맥주효모가 직접적으로 작용한다는 기전은 명확히 증명되어 있지 않다.

```text
- AR: Androgen Receptor
- SRD5A1: Steroid 5-alpha-reductase type 1
- SRD5A2: Steroid 5-alpha-reductase type 2
```

따라서 본 프로젝트는 다음 질문에서 출발한다.

> 맥주효모의 유전자와 대사산물 중, 인간의 탈모 관련 표적 단백질과 graph-based path를 통해 연결될 가능성이 높은 후보는 무엇인가?

---

## 2. Project Scope

초기 설계는 YMDB, STITCH, Open Targets 전체 bulk dump를 사용하는 대규모 프로젝트였다.

```text
Original large-scale design:
- YMDB 전체 yeast metabolite / gene data
- STITCH 전체 human drug-target interaction
- Open Targets 전체 target-disease association
- 수만~수십만 node/edge
- Colab에서 압축 해제, 파싱, preprocessing 시간이 매우 오래 걸림
```

하지만 본 프로젝트는 **대학교 4학년 학교 과제 수준**에 맞게 다음과 같이 단순화하였다.

```text
Final school-project design:
- 대형 bulk dump 사용하지 않음
- 소형 curated dataset 직접 생성
- PubChem API로 필요한 compound의 SMILES만 가져옴
- RDKit으로 fingerprint와 Tanimoto similarity 계산
- PyTorch Geometric의 HeteroData와 HANConv 사용
- Colab에서 빠르게 실행 가능
```

이 단순화는 핵심 아이디어를 유지하면서도 실행 가능성을 높이기 위한 것이다.

---

## 3. Core Contribution

본 프로젝트의 핵심 contribution은 다음과 같다.

### 3.1 Cross-species graph construction

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

### 3.2 Latent chemical similarity edge

Yeast metabolite와 human drug 사이에는 직접적인 biological edge가 없다.  
따라서 RDKit 기반 화학 구조 유사도를 이용해 다음 edge를 계산한다.

```text
(Metabolite, similar_to, Drug)
```

이는 데이터 단절을 극복하기 위한 latent edge이다.

### 3.3 Inductive heterogeneous GNN

처음 만든 모델은 `nn.Embedding(node_id)` 기반이었기 때문에 transductive 성격이 강했다.  
최종 버전은 node ID embedding을 버리고, node feature 기반으로 바꾸어 **inductive learning이 가능하도록 설계**한다.

```text
Transductive version:
node_id → nn.Embedding → HANConv

Inductive version:
node feature → Linear encoder → HANConv
```

즉, 새로운 metabolite나 drug가 들어와도 SMILES로부터 fingerprint를 만들 수 있으므로, 학습된 model parameter를 적용할 수 있다.

---

## 4. Graph Schema

### 4.1 Node Types

본 프로젝트의 heterogeneous graph는 총 5종류의 node type을 가진다.

```text
1. gene
   - Yeast gene
   - 예: ERG11, ERG7, HMG1, ERG20

2. metabolite
   - Yeast metabolite 또는 sterol/isoprenoid-related compound
   - 예: lanosterol, ergosterol, squalene, testosterone

3. drug
   - Human drug
   - 예: finasteride, dutasteride, spironolactone, ketoconazole

4. target
   - Human target protein
   - 예: SRD5A1, SRD5A2, AR, CYP51A1

5. disease
   - Human disease
   - 본 프로젝트에서는 ALOPECIA 하나만 사용
```

---

### 4.2 Edge Types

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

reverse edge는 생물학적 반대 작용을 주장하는 것이 아니라, GNN message passing이 양방향으로 흐르도록 하기 위한 기술적 edge이다.

---

## 5. Data Design

### 5.1 Yeast Gene → Metabolite Curated Data

학교 과제용으로 sterol biosynthesis, mevalonate pathway, isoprenoid pathway 중심의 curated data를 사용한다.

예시:

```text
ERG10  → acetoacetyl-CoA
HMG1   → mevalonic acid
ERG20  → farnesyl pyrophosphate
ERG9   → squalene
ERG7   → lanosterol
ERG11  → lanosterol
ERG3   → ergosterol
ERG5   → ergosterol
ERG4   → ergosterol
```

추가적으로 구조 비교를 위해 steroid-like compound도 일부 포함한다.

```text
ERG11 → testosterone
ERG11 → dihydrotestosterone
ERG11 → androstenedione
ERG11 → progesterone
```

이 edge는 실제 효모 생합성 경로 전체를 완벽히 재현하는 것이 아니라,  
project demonstration을 위한 curated biological approximation이다.

---

### 5.2 Human Drug → Target Curated Data

탈모, androgen axis, steroid/sterol metabolism 관련 약물을 포함한다.

예시:

```text
finasteride       → SRD5A2
dutasteride       → SRD5A1 / SRD5A2
spironolactone    → AR
bicalutamide      → AR
flutamide         → AR
enzalutamide      → AR
ketoconazole      → CYP51A1
abiraterone       → CYP17A1
minoxidil         → KCNJ8
```

비교용 decoy target도 포함한다.

```text
simvastatin  → HMGCR
atorvastatin → HMGCR
tamoxifen    → ESR1
```

---

### 5.3 Target → Disease Curated Data

질병 node는 `ALOPECIA` 하나만 사용한다.

```text
SRD5A2  → ALOPECIA: high relevance
SRD5A1  → ALOPECIA: high relevance
AR      → ALOPECIA: high relevance
CYP17A1 → ALOPECIA: medium relevance
CYP51A1 → ALOPECIA: weak relevance
KCNJ8   → ALOPECIA: weak/alternative relevance
HMGCR   → ALOPECIA: decoy weak relevance
ESR1    → ALOPECIA: decoy weak relevance
```

---

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

Gene은 화학 구조가 없기 때문에, 연결된 metabolite fingerprint의 평균을 사용한다.

```text
gene.x = mean(fingerprint of connected metabolites)
dimension = 2048
```

예:

```text
ERG11 is connected to:
- lanosterol
- testosterone
- dihydrotestosterone

ERG11 feature =
mean(fp(lanosterol), fp(testosterone), fp(dihydrotestosterone))
```

이렇게 하면 unseen gene도 다음 조건을 만족하면 feature를 만들 수 있다.

```text
unseen gene → at least one connected metabolite
```

---

### 7.4 Target Node Feature

Target node는 소형 biological category vector를 사용한다.

예시 feature column:

```text
[
  androgen_axis,
  steroid_metabolism,
  hair_growth,
  cholesterol_metabolism,
  estrogen_axis,
  alopecia_relevance
]
```

예:

```text
SRD5A2 = [1, 1, 0, 0, 0, 1.00]
AR     = [1, 0, 0, 0, 0, 0.90]
HMGCR  = [0, 0, 0, 1, 0, 0.10]
```

---

### 7.5 Disease Node Feature

Disease node도 작은 biological feature vector를 사용한다.

예:

```text
ALOPECIA = [
  androgen_related,
  steroid_related,
  hair_related,
  druggable_axis,
  human_disease,
  relevance
]
```

구체적으로:

```text
ALOPECIA = [1, 1, 1, 1, 1, 1.0]
```

---

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
gene.x       = 2048
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

이 구조 덕분에 새로운 node도 같은 feature dimension만 맞으면 학습된 projection layer를 통과할 수 있다.

---

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

hidden dimension은 학교 과제용으로 64를 사용한다.

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
score(u, v) = sigmoid(z_u · z_v)
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

---

## 14. Important Implementation Rules for Future AI Agents

이 README를 기반으로 코드를 다시 작성하는 AI agent는 다음 규칙을 반드시 지켜야 한다.

### 14.1 Do Not Use Node ID Embeddings

다음 구조를 사용하면 안 된다.

```python
nn.Embedding(num_nodes, hidden_channels)
```

이 방식은 unseen node에 대해 inductive inference가 어렵다.

반드시 다음 구조를 사용한다.

```python
node feature → Linear projection → HANConv
```

---

### 14.2 Molecule Nodes Must Use RDKit Features

`metabolite`와 `drug` node는 반드시 SMILES 기반 fingerprint를 사용한다.

```text
SMILES → RDKit Mol → Morgan fingerprint → 2048-dim vector
```

---

### 14.3 Gene Nodes Must Be Feature-based

Gene node도 단순 one-hot이나 embedding으로 처리하지 않는다.  
최종 inductive 버전에서는 다음 방식을 사용한다.

```text
gene feature = mean fingerprint of connected metabolites
```

이 방식은 new gene에도 적용 가능하다.

---

### 14.4 Reverse Edges Are Required

모든 relation에 reverse edge를 추가해야 한다.

```text
gene → metabolite
metabolite → gene

metabolite → drug
drug → metabolite

drug → target
target → drug

target → disease
disease → target
```

이걸 하지 않으면 gene node가 disease 쪽 정보를 충분히 받을 수 없다.

---

### 14.5 Training Target Should Not Include Gene-Disease

학교 과제 버전에서는 `gene → disease` edge가 supervised label로 존재하지 않는다.

따라서 training task는 다음 두 개만 사용한다.

```text
1. metabolite-drug link prediction
2. drug-target link prediction
```

`gene → disease`는 downstream ranking으로만 사용한다.

---

### 14.6 Test Data Interpretation Must Be Correct

`test_data`는 다음을 검증한다.

```text
- 숨겨둔 metabolite-drug edge 복원
- 숨겨둔 drug-target edge 복원
```

`test_data`가 gene-disease prediction을 직접 검증한다고 설명하면 안 된다.

---

### 14.7 Unseen Node Inference Must Not Retrain

inductive inference를 보여줄 때는 다음을 지켜야 한다.

```text
1. trained model load
2. model.eval()
3. new node feature 생성
4. graph에 new node와 edge 추가
5. forward pass
6. score 계산
```

재학습 없이 수행해야 inductive inference demonstration이 된다.

---

## 15. Recommended Colab Execution Order

Colab notebook은 다음 순서로 구성한다.

```text
Cell 1. Install packages
Cell 2. Import and setup
Cell 3. Curated dataset
Cell 4. Fetch SMILES from PubChem
Cell 5. Generate RDKit fingerprints
Cell 6. Build metabolite-drug similarity edges
Cell 7. Build node mappings
Cell 8. Build inductive node features
Cell 9. Build HeteroData
Cell 10. RandomLinkSplit
Cell 11. Define Inductive HAN model
Cell 12. Define loss and evaluation functions
Cell 13. Train model
Cell 14. Test evaluation
Cell 15. Recall@K
Cell 16. Existing gene → Alopecia ranking
Cell 17. Define unseen metabolite data
Cell 18. Add unseen metabolite nodes to graph
Cell 19. Predict unseen metabolite → Alopecia
Cell 20. Define unseen drug data
Cell 21. Add unseen drug nodes to graph
Cell 22. Predict unseen drug → target
Cell 23. Predict unseen gene → Alopecia
Cell 24. Save results
```

---

## 16. Output Files

최종 결과는 다음 파일로 저장한다.

```text
inductive_han_project_results/
├── genes.csv
├── metabolites.csv
├── drugs.csv
├── gene_metabolite_edges.csv
├── metabolite_drug_similarity_edges.csv
├── drug_target_edges.csv
├── target_disease_edges.csv
├── existing_gene_ranking.csv
├── unseen_metabolite_disease_prediction.csv
├── unseen_drug_target_prediction.csv
├── unseen_gene_disease_prediction.csv
└── inductive_han_model.pt
```

---

## 17. Expected Results

### 17.1 Training Result

소형 curated dataset이므로 AUC/AP는 데이터 split에 따라 변동될 수 있다.  
성능 수치는 절대적인 생물학적 정확도라기보다 pipeline validation으로 해석한다.

예상 output:

```text
Epoch 010 | Loss ...
Epoch 020 | Loss ...
...
Test Result:
('metabolite', 'similar_to', 'drug') AUC/AP
('drug', 'inhibits', 'target') AUC/AP
```

---

### 17.2 Existing Gene Ranking

예상 output:

```text
rank | gene_id | score
1    | ERG11   | ...
2    | ERG7    | ...
3    | ERG20   | ...
...
```

해석:

```text
ERG 계열 gene은 sterol-related metabolite와 많이 연결되어 있고,
이 metabolite들이 androgen/sterol-related drugs와 latent edge를 형성하므로
ALOPECIA node와 가까운 embedding을 가질 가능성이 있다.
```

---

### 17.3 Unseen Metabolite Prediction

예상 output:

```text
unseen_metabolite | predicted_disease | score
dihydrotestosterone | ALOPECIA | ...
cortisol            | ALOPECIA | ...
estradiol           | ALOPECIA | ...
```

해석:

```text
새로운 metabolite도 SMILES 기반 fingerprint를 통해 graph에 추가할 수 있고,
학습된 HAN parameter로 disease relevance score를 계산할 수 있다.
```

---

### 17.4 Unseen Drug Prediction

예상 output:

```text
unseen_drug | predicted_target | score
RU58841     | AR               | ...
topilutamide| AR               | ...
turosteride | SRD5A2           | ...
```

해석:

```text
새로운 drug도 fingerprint와 similarity edge를 통해 graph에 추가되며,
기존 target node와의 link score를 예측할 수 있다.
```

---

### 17.5 Unseen Gene Prediction

예상 output:

```text
unseen_gene       | predicted_disease | score
NEW_STEROID_LIKE_1| ALOPECIA          | ...
NEW_ERG_LIKE_1    | ALOPECIA          | ...
```

해석:

```text
새로운 gene도 연결 metabolite가 주어지면 평균 fingerprint 기반 feature를 생성할 수 있고,
ALOPECIA node와의 graph-based score를 계산할 수 있다.
```

---

## 18. Limitations

본 프로젝트의 한계는 다음과 같다.

```text
1. Curated small dataset이므로 실제 생물학적 결론을 내리기 어렵다.
2. Drug-target edge와 target-disease score는 과제용으로 단순화한 값이다.
3. Tanimoto similarity가 낮아도 Top-K edge로 연결하므로 weak edge가 포함될 수 있다.
4. Gene feature를 metabolite fingerprint 평균으로 두는 것은 단순화된 approximation이다.
5. Disease node가 하나뿐이므로 disease classification model은 아니다.
6. Wet-lab validation이나 docking validation은 수행하지 않았다.
```

따라서 결과는 다음처럼 해석해야 한다.

```text
"치료 효과 증명"이 아니라
"candidate prioritization을 위한 graph-based computational hypothesis generation"
```

---

## 19. Suggested Report Wording

보고서에는 다음 문장을 사용할 수 있다.

> 본 프로젝트는 맥주효모 유래 유전자 및 대사산물이 인간 탈모 관련 표적 단백질과 직접 연결되어 있지 않다는 데이터 단절 문제를 해결하기 위해, 화학 구조 유사도를 latent edge로 도입한 inductive heterogeneous graph neural network pipeline을 구현하였다. Metabolite와 drug node는 SMILES 기반 RDKit Morgan fingerprint를 feature로 사용하였고, gene node는 연결된 metabolite fingerprint의 평균으로 표현하였다. 이를 통해 학습 시점에 존재하지 않았던 새로운 metabolite, drug, gene에 대해서도 feature를 생성하고, 학습된 HAN parameter를 적용하여 link score를 예측할 수 있도록 설계하였다.

그리고 결과 해석은 다음처럼 제한적으로 표현한다.

> 본 결과는 실제 치료 효과나 직접적인 생화학적 억제 효과를 증명하는 것이 아니라, 제한된 curated dataset에서 graph representation learning과 chemical similarity를 이용해 후보 gene/metabolite/drug를 우선순위화한 in silico proof-of-concept이다.

---

## 20. Minimal Code Skeleton

아래는 전체 구조를 요약한 pseudo-code이다.

```python
# 1. Build curated data
gene_metabolite_edges = ...
drug_target_edges = ...
target_disease_edges = ...

# 2. Fetch SMILES
metabolites_df["smiles"] = fetch_from_pubchem(...)
drugs_df["smiles"] = fetch_from_pubchem(...)

# 3. RDKit features
metabolites_df["fp"] = smiles_to_fp(...)
drugs_df["fp"] = smiles_to_fp(...)

# 4. Build latent similarity edges
metabolite_drug_edges = topk_tanimoto(metabolites, drugs)

# 5. Build mappings
gene2idx, met2idx, drug2idx, target2idx, disease2idx = ...

# 6. Build inductive node features
metabolite.x = Morgan fingerprint
drug.x = Morgan fingerprint
gene.x = mean connected metabolite fingerprint
target.x = biological category vector
disease.x = disease category vector

# 7. Build HeteroData
data["gene"].x = ...
data["metabolite"].x = ...
data["drug"].x = ...
data["target"].x = ...
data["disease"].x = ...

data["gene", "produces", "metabolite"].edge_index = ...
data["metabolite", "similar_to", "drug"].edge_index = ...
data["drug", "inhibits", "target"].edge_index = ...
data["target", "associated_with", "disease"].edge_index = ...

# 8. Add reverse edges
...

# 9. RandomLinkSplit
train_data, val_data, test_data = RandomLinkSplit(...)

# 10. Model
model = InductiveHANLinkPredictor(
    metadata=data.metadata(),
    in_channels_dict=...,
)

# 11. Train
for epoch in range(EPOCHS):
    loss = BCEWithLogitsLoss(pred_edge_logits, edge_labels)
    loss.backward()
    optimizer.step()

# 12. Evaluate
AUC, AP, Recall@K

# 13. Existing gene ranking
score(gene, ALOPECIA)

# 14. Unseen node inference
new node → feature → graph extension → model.eval() → score
```

---

## 21. Final One-line Description

> An inductive heterogeneous graph neural network pipeline that connects yeast genes to human alopecia targets through metabolite-drug structural similarity edges and predicts candidate genes, metabolites, and drugs using feature-based HAN link prediction.

---

## 22. Korean One-line Description

> 효모 유전자와 인간 탈모 표적 단백질 사이의 직접적인 데이터 단절을 화합물 구조 유사도 기반 latent edge로 연결하고, feature 기반 inductive HAN을 이용해 기존 및 신규 후보 node의 질병 관련성을 예측하는 이종 그래프 머신러닝 프로젝트이다.
