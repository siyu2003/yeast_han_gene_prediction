import os
import requests
import pandas as pd
from tqdm import tqdm
from chembl_webresource_client.new_client import new_client

def get_uniprot_id(symbol):
    """
    유전자 심볼을 이용해 UniProt Accession ID를 가져옵니다.
    """
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f"(gene_exact:{symbol}) AND (organism_id:9606)",
        "fields": "accession",
        "format": "json",
        "size": 1
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("results"):
                return data["results"][0]["primaryAccession"]
    except Exception:
        pass
    return None

def get_chembl_target_id_robust(symbol):
    """
    3가지 하이브리드 전략을 사용하여 ChEMBL Target ID를 집요하게 찾습니다.
    성공 시 (ChEMBL ID, 성공한 전략명)을 튜플로 반환합니다.
    """
    # ---------------------------------------------------------
    # 전략 1: UniProt 고유 ID 매핑 (정확도 1순위)
    # ---------------------------------------------------------
    uniprot_id = get_uniprot_id(symbol)
    if uniprot_id:
        try:
            targets = new_client.target.filter(target_components__accession=uniprot_id)
            for t in targets:
                if t.get('target_type') == 'SINGLE PROTEIN' and t.get('organism') == 'Homo sapiens':
                    return t.get('target_chembl_id'), "UniProt"
        except Exception:
            pass

    # ---------------------------------------------------------
    # 전략 2: ChEMBL 동의어(Synonym) 및 공식 명칭(Pref Name) 매핑
    # ---------------------------------------------------------
    try:
        # 동의어 정확히 일치 검색
        targets_syn = new_client.target.filter(
            target_synonyms__target_synonym__iexact=symbol,
            target_type='SINGLE PROTEIN',
            organism='Homo sapiens'
        )
        if targets_syn:
            return targets_syn[0].get('target_chembl_id'), "Synonym"
            
        # 공식 명칭 정확히 일치 검색
        targets_pref = new_client.target.filter(
            pref_name__iexact=symbol,
            target_type='SINGLE PROTEIN',
            organism='Homo sapiens'
        )
        if targets_pref:
            return targets_pref[0].get('target_chembl_id'), "Synonym"
    except Exception:
        pass

    # ---------------------------------------------------------
    # 전략 3: 기존의 일반 텍스트 검색 (Search)
    # ---------------------------------------------------------
    try:
        targets_search = new_client.target.search(symbol)
        for t in targets_search:
            if t.get('target_type') == 'SINGLE PROTEIN' and t.get('organism') == 'Homo sapiens':
                return t.get('target_chembl_id'), "General Search"
    except Exception:
        pass

    # 3가지 방법을 다 썼는데도 없으면 실패
    return None, "Fail"


def build_drug_target_edges(target_disease_path="real_target_disease_edges.csv", max_records_per_target=30):
    print("[2/3] 3중 하이브리드 탐색(UniProt + Synonym + 일반검색)으로 약물 데이터를 수집합니다...")
    
    if not os.path.exists(target_disease_path):
        print(f"  [오류] '{target_disease_path}' 파일이 없습니다. 1단계를 먼저 실행해주세요.")
        return pd.DataFrame()
        
    df_targets = pd.read_csv(target_disease_path)
    if 'target_symbol' not in df_targets.columns:
        print("  [오류] 입력 데이터에 'target_symbol' 컬럼이 없습니다.")
        return pd.DataFrame()
        
    target_symbols = df_targets['target_symbol'].dropna().unique()
    total_targets = len(target_symbols)
    print(f"  -> 검색할 고유 타겟 수: {total_targets}개\n")

    records = []
    
    # 전략별 성공 현황과 실패 원인을 추적하는 딕셔너리
    status_tracker = {
        "success_uniprot": [],
        "success_synonym": [],
        "success_search": [],
        "fail_mapping": [],  # DB에서 타겟 자체를 못 찾은 경우
        "fail_no_drug": []   # 타겟은 찾았으나 IC50 약물이 없는 경우
    }
    
    for symbol in tqdm(target_symbols, desc="타겟별 약물 검색"):
        # 3중 검색 시작
        chembl_id, strategy = get_chembl_target_id_robust(symbol)
        
        # 타겟 매핑에 아예 실패한 경우
        if not chembl_id:
            status_tracker["fail_mapping"].append(symbol)
            continue
            
        # 매핑된 타겟에 대해 약물 검색
        try:
            activities = new_client.activity.filter(
                target_chembl_id=chembl_id,
                standard_type='IC50'
            ).only(['molecule_chembl_id', 'standard_value'])
            
            count = 0
            found_drug = False
            
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
                            "drug_id": act['molecule_chembl_id'],
                            "target_symbol": symbol,
                            "target_chembl_id": chembl_id,
                            "mapping_strategy": strategy, # 어떤 방식으로 찾았는지 기록
                            "confidence": conf,
                            "ic50_value": ic50_val
                        })
                        count += 1
                        found_drug = True
                    except ValueError:
                        continue
                        
            # 약물 검색 결과에 따라 상태 분류
            if found_drug:
                if strategy == "UniProt":
                    status_tracker["success_uniprot"].append(symbol)
                elif strategy == "Synonym":
                    status_tracker["success_synonym"].append(symbol)
                elif strategy == "General Search":
                    status_tracker["success_search"].append(symbol)
            else:
                status_tracker["fail_no_drug"].append(f"{symbol} (ID: {chembl_id})")
                
        except Exception as e:
            status_tracker["fail_no_drug"].append(f"{symbol} (API 에러)")
            continue

    # ---------------------------------------------------------
    # 📊 종합 결과 요약 보고서
    # ---------------------------------------------------------
    total_success = len(status_tracker['success_uniprot']) + len(status_tracker['success_synonym']) + len(status_tracker['success_search'])
    
    print("\n" + "="*70)
    print("📊 [3중 하이브리드 검색 결과 요약 보고서]")
    print(f"  - 총 검색 대상 타겟 : {total_targets}개")
    print(f"  - 🟢 약물 검색 성공 : {total_success}개")
    print(f"      ├─ UniProt 매핑으로 찾음: {len(status_tracker['success_uniprot'])}개")
    print(f"      ├─ Synonym 매핑으로 찾음: {len(status_tracker['success_synonym'])}개")
    print(f"      └─ 일반 검색으로 찾음   : {len(status_tracker['success_search'])}개")
    print(f"  - 🔴 약물 검색 실패 : {total_targets - total_success}개")
    print("-" * 70)
    
    if status_tracker["fail_mapping"]:
        print(f"  [❌ 완전 매핑 실패] 3가지 방법으로도 DB에서 단백질을 찾을 수 없음 ({len(status_tracker['fail_mapping'])}개)")
        for i in range(0, len(status_tracker["fail_mapping"]), 8):
            print(f"    {', '.join(status_tracker['fail_mapping'][i:i+8])}")
            
    if status_tracker["fail_no_drug"]:
        print(f"\n  [⚠️ 약물 부재] 단백질은 찾았으나 IC50 활성 조건을 만족하는 약물이 없음 ({len(status_tracker['fail_no_drug'])}개)")
        # 이름 문제가 아니라 순수하게 아직 연구/데이터가 부족한 타겟들
        
    print("="*70 + "\n")

    if not records:
         print("  [알림] 조건에 맞는 약물 데이터를 찾지 못했습니다.")
         return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values('confidence', ascending=False).drop_duplicates(subset=['drug_id', 'target_symbol'])
    
    output_file = "real_drug_target_edges.csv"
    df.to_csv(output_file, index=False)
    print(f"  -> 성공! '{output_file}' 생성 완료 (총 {len(df)}개 엣지)\n")
    return df

# 실행 코드
if __name__ == "__main__":
    df_drugs = build_drug_target_edges()
