import pandas as pd
import requests

def build_target_disease_edges(efo_id="MONDO_0004907", disease_name="Alopecia", max_size=50):
    """
    OpenTargets API를 사용하여 특정 질병(Alopecia)과 연관된 타겟(유전자/단백질) 데이터를 가져옵니다.
    """
    print(f"[1/3] OpenTargets API에서 타겟-질병({disease_name}) 연관성 데이터 생성 중...")
    
    # GraphQL 쿼리: Target ID(Ensembl)와 Symbol을 모두 가져오도록 개선
    query = """
    query targetDiseaseQuery($efoId: String!, $size: Int!) {
      disease(efoId: $efoId) {
        associatedTargets(page: {index: 0, size: $size}) {
          rows {
            target {
              id
              approvedSymbol
            }
            score
          }
        }
      }
    }
    """
    
    variables = {
        "efoId": efo_id,
        "size": max_size
    }
    
    try:
        response = requests.post(
            'https://api.platform.opentargets.org/api/v4/graphql', 
            json={'query': query, 'variables': variables}
        )
        response.raise_for_status()
        data = response.json()
        
        records = []
        rows = data['data']['disease']['associatedTargets']['rows']
        
        for row in rows:
            target_ensembl_id = row['target']['id']
            target_symbol = row['target']['approvedSymbol']
            score = row['score']
            
            records.append({
                "target_ensembl_id": target_ensembl_id,
                "target_symbol": target_symbol,
                "disease_id": efo_id,
                "disease_name": disease_name,
                "association_score": score
            })
            
        df = pd.DataFrame(records)
        output_file = "real_target_disease_edges.csv"
        df.to_csv(output_file, index=False)
        print(f"  -> 성공! '{output_file}' 생성 완료 (총 {len(df)}개 타겟)\n")
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"  [오류] API 네트워크 호출 실패: {e}")
        return pd.DataFrame()
    except KeyError as e:
        print(f"  [오류] API 응답 데이터 구조가 예상과 다릅니다: {e}")
        return pd.DataFrame()

# 단독 테스트용 실행 코드
if __name__ == "__main__":
    df_alopecia = build_target_disease_edges()
    if not df_alopecia.empty:
        print(df_alopecia.head())
