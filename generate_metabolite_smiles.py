import csv

def extract_metabolite_smiles(sdf_path, output_path):
    print(f"[{sdf_path}] 파일에서 데이터 추출을 시작합니다...")
    
    results = []
    
    # 파일을 한 줄씩 읽어 메모리 사용량을 최소화합니다.
    with open(sdf_path, 'r', encoding='utf-8') as f:
        current_smiles = ""
        current_name = ""
        
        iterator = iter(f)
        for line in iterator:
            line = line.strip()
            
            # SMILES 필드를 만나면 다음 줄을 읽어 저장
            if line == "> <SMILES>":
                try:
                    current_smiles = next(iterator).strip()
                except StopIteration:
                    break
                    
            # 대사체 이름 필드를 만나면 다음 줄을 읽어 저장
            elif line == "> <GENERIC_NAME>":
                try:
                    current_name = next(iterator).strip()
                except StopIteration:
                    break
                    
            # 분자 데이터의 끝($$$$)을 만나면 결과 리스트에 추가하고 초기화
            elif line == "$$$$":
                if current_name and current_smiles:
                    results.append({
                        "Metabolite": current_name, 
                        "SMILES": current_smiles
                    })
                current_smiles = ""
                current_name = ""
                
    # 추출한 데이터를 CSV 파일로 저장 (utf-8-sig로 저장하여 엑셀에서 한글 깨짐 방지)
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Metabolite", "SMILES"])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"🎉 추출 완료! 총 {len(results)}개의 대사체-SMILES 쌍이 '{output_path}' 파일에 저장되었습니다.")

if __name__ == "__main__":
    # 입력 SDF 파일 경로와 출력할 CSV 파일 경로를 지정합니다.
    INPUT_SDF = 'ymdb.sdf'
    OUTPUT_CSV = 'ymdb_metabolite_smiles.csv'
    
    extract_metabolite_smiles(INPUT_SDF, OUTPUT_CSV)
