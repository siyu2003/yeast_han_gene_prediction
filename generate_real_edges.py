import os
import json
import pandas as pd
from tqdm import tqdm
from chembl_webresource_client.new_client import new_client

import requests

def build_target_disease_edges():
    print("[1/3] OpenTargets API에서 타겟-질병(Alopecia) 연관성 데이터 생성 중...")
    
    # EFO_0007324: Androgenetic alopecia (남성형 탈모)
    query = """
    query {
      disease(efoId: "EFO_0007324") {
        associatedTargets(page: {index: 0, size: 50}) {
          rows {
            target {
              approvedSymbol
            }
            score
          }
        }
      }
    }
    """
    
    try:
        response = requests.post('https://api.platform.opentargets.org/api/v4/graphql', json={'query': query})
        data = response.json()
        
        records = []
        rows = data['data']['disease']['associatedTargets']['rows']
        for row in rows:
            target_symbol = row['target']['approvedSymbol']
            score = row['score']
            records.append({
                "target_id": target_symbol,
                "disease_id": "ALOPECIA",
                "disease_name": "Alopecia",
                "association_score": score
            })
            
        df = pd.DataFrame(records)
        df.to_csv("real_target_disease_edges.csv", index=False)
        print(f"  -> 성공! 'real_target_disease_edges.csv' 생성 완료 (총 {len(df)}개 타겟)\n")
        return df
    except Exception as e:
        print(f"  [오류] API 호출 실패: {e}")
        return pd.DataFrame()

def build_drug_target_edges(max_records_per_target=30):
    print("[2/3] ChEMBL API에서 핵심 타겟을 억제하는 약물 데이터를 가져오는 중...")
    
    key_targets = {
        'CHEMBL1871': 'AR',       
        'CHEMBL1787': 'SRD5A1',   
        'CHEMBL1788': 'SRD5A2',   
        'CHEMBL2047': 'CYP17A1',
        'CHEMBL1936': 'CYP51A1'
    }
    
    records = []
    for chembl_id, symbol in tqdm(key_targets.items(), desc="타겟별 약물 검색"):
        activities = new_client.activity.filter(
            target_chembl_id=chembl_id,
            standard_type='IC50'
        ).only(['molecule_chembl_id', 'standard_value'])
        
        count = 0
        for act in activities:
            if count >= max_records_per_target:
                break
            val = act.get('standard_value')
            if val is not None:
                try:
                    ic50_val = float(val)
                    if ic50_val <= 100: conf = 0.95
                    elif ic50_val <= 1000: conf = 0.80
                    elif ic50_val <= 10000: conf = 0.60
                    else: conf = 0.50
                    
                    records.append({
                        "drug_name": act['molecule_chembl_id'],
                        "target_id": symbol, 
                        "confidence": conf
                    })
                    count += 1
                except ValueError:
                    continue
                    
    df = pd.DataFrame(records).sort_values('confidence', ascending=False).drop_duplicates(subset=['drug_name', 'target_id'])
    df.to_csv("real_drug_target_edges.csv", index=False)
    print(f"\n  -> 성공! 'real_drug_target_edges.csv' 생성 완료 (총 {len(df)}개 약물-타겟 연결)\n")
    return df

def build_gene_metabolite_edges(json_path="ymdb_full.json"):
    print("[3/3] YMDB 로컬 파일에서 유전자-대사체 데이터를 파싱하는 중...")
    
    if not os.path.exists(json_path):
        print(f"  [경고] '{json_path}' 파일이 없습니다. YMDB에서 다운로드해주세요.")
        return pd.DataFrame()

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
            
        # YMDB JSON 파일은 때때로 쉼표(,) 없이 객체가 이어지는 형태(JSON Lines 또는 깨진 List)로 다운로드됩니다.
        # json.JSONDecoder().raw_decode를 사용하여 수동으로 객체들을 하나씩 추출합니다.
        data = []
        decoder = json.JSONDecoder()
        
        # 파일이 '['로 시작한다면 리스트 괄호를 제거해줍니다.
        text = raw_text.strip()
        if text.startswith('['):
            text = text[1:]
            
        while text:
            text = text.lstrip()
            if not text or text.startswith(']'):
                break
                
            # 쉼표가 중간에 껴있을 수도 있으니 제거
            if text.startswith(','):
                text = text[1:].lstrip()
                if not text: break
                
            try:
                obj, idx = decoder.raw_decode(text)
                data.append(obj)
                text = text[idx:]
            except json.JSONDecodeError as e:
                # 더 이상 파싱할 수 없다면 중단
                print(f"  [알림] 파일 후반부 손상 감지. 현재까지 {len(data)}개 항목 파싱 완료.")
                break
                
    except Exception as e:
        print(f"  [치명적 오류] '{json_path}' 파일을 열 수 없습니다: {e}")
        return pd.DataFrame()

    records = []
    for met in data:
        met_name = met.get('name', '')
        # YMDB JSON 구조 상 유전자 정보는 'enzymes'가 아니라 'proteins' 키 아래에 있습니다.
        proteins = met.get('proteins', [])
        if met_name and proteins:
            for p in proteins:
                gene_name = p.get('gene_name')
                if gene_name:
                    records.append({"gene_id": gene_name, "metabolite_name": met_name})
                    
    df = pd.DataFrame(records).drop_duplicates()
    df.to_csv("real_gene_metabolite_edges.csv", index=False)
    print(f"  -> 성공! 'real_gene_metabolite_edges.csv' 생성 완료 (총 {len(df)}개 유전자-대사체 연결)\n")
    return df

if __name__ == "__main__":
    print("="*60)
    print("🚀 이질적 그래프(Heterogeneous Graph) 엣지 데이터 생성 시작")
    print("="*60 + "\n")
    
    build_target_disease_edges()
    build_drug_target_edges(max_records_per_target=40) 
    build_gene_metabolite_edges("ymdb_full.json")
    
    print("="*60)
    print("🎉 데이터 파이프라인 실행 종료!")