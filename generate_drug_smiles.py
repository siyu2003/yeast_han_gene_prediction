import os
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote
from tqdm.auto import tqdm

def fetch_single_smiles(compound_name):
    """
    화합물 이름이 CHEMBL ID인 경우 ChEMBL API를, 일반 이름인 경우 PubChem API를 사용하여 SMILES를 조회합니다.
    """
    if compound_name.upper().startswith("CHEMBL"):
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{compound_name.upper()}.json"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                structs = data.get("molecule_structures")
                if structs and isinstance(structs, dict):
                    smiles = structs.get("canonical_smiles") or structs.get("standard_smiles")
                    if smiles:
                        return compound_name, smiles
        except Exception:
            pass
            
    # 일반 이름인 경우 PubChem API 사용
    safe_name = quote(compound_name)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{safe_name}/property/CanonicalSMILES,IsomericSMILES/JSON"
    
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props:
                smiles = props[0].get("CanonicalSMILES") or props[0].get("SMILES") or props[0].get("IsomericSMILES") or props[0].get("ConnectivitySMILES")
                if smiles:
                    return compound_name, smiles
    except Exception:
        pass
        
    return compound_name, None


def fetch_smiles_concurrently(compound_list, max_workers=10):
    """
    주어진 화합물 리스트를 멀티스레딩으로 매우 빠르게 조회합니다.
    """
    unique_names = list(set(compound_list))
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_single_smiles, name): name for name in unique_names}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="SMILES 추출 중"):
            name, smiles = future.result()
            results.append({"drug_id": name, "drug_smiles": smiles}) # Merge를 위해 컬럼명 변경
            time.sleep(0.05) # API 서버 과부하 방지
            
    df = pd.DataFrame(results)
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 약물(Drug) SMILES 멀티스레딩 추출 파이프라인 시작")
    print("=" * 60)
    
    # 1. 입력 파일 경로 설정
    input_csv = "real_drug_target_edges.csv"  # 실제 파일 이름으로 변경하세요
    output_csv = "real_drug_target_edges_with_smiles.csv"
    
    if not os.path.exists(input_csv):
        print(f"[오류] '{input_csv}' 파일을 찾을 수 없습니다.")
        exit(1)
        
    # 2. 데이터 로드
    df_edges = pd.read_csv(input_csv)
    print(f"\n[1/3] '{input_csv}' 파일 로드 완료 (총 {len(df_edges)}개 엣지)")
    
    # 3. 고유한 약물 ID 추출
    if 'drug_id' not in df_edges.columns:
        print("[오류] CSV 파일에 'drug_id' 컬럼이 없습니다.")
        exit(1)
        
    unique_drugs = df_edges['drug_id'].dropna().unique().tolist()
    print(f"[2/3] API 검색 대상 고유 약물 수: {len(unique_drugs)}개")
    
    # 4. SMILES 다중 스레드 검색
    df_smiles = fetch_smiles_concurrently(unique_drugs, max_workers=10)
    
    # 5. 검색 결과 통계 확인
    success_count = df_smiles['drug_smiles'].notna().sum()
    print(f"\n  -> 추출 성공: {success_count}개 / 실패: {len(unique_drugs) - success_count}개")
    
    # 6. 원본 데이터와 병합 (Left Join)
    print("\n[3/3] 원본 데이터와 SMILES 데이터 병합 중...")
    df_final = pd.merge(df_edges, df_smiles, on="drug_id", how="left")
    
    # 7. 결과 저장
    df_final.to_csv(output_csv, index=False)
    print("=" * 60)
    print(f"🎉 성공! 모든 결과가 '{output_csv}'에 저장되었습니다.")