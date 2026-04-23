import pandas as pd

def export_missing_abstract_dois(null_report_path, source_data_path, output_path):
    try:
        # 1. Đọc file báo cáo giá trị null
        df_null = pd.read_csv(null_report_path)
        
        # 2. Đọc file gốc để lấy cột DOI
        df_source = pd.read_csv(source_data_path)

        # 3. Lọc lấy những dòng mà 'abstract' là NaN (rỗng) trong file report
        # Sử dụng .isna() để bắt chính xác các ô rỗng
        missing_abstracts = df_null[df_null['abstract'].isna()].copy()

        # 4. Kết hợp (Merge) với file gốc dựa trên cột 'id'
        # Chúng ta chỉ lấy cột 'id' và 'doi' từ file gốc để ghép vào
        result = pd.merge(
            missing_abstracts[['id']], 
            df_source[['id', 'doi']], 
            on='id', 
            how='left'
        )

        # 5. Xuất ra file mới (chỉ lấy 2 cột id và doi)
        result.to_csv(output_path, index=False)
        
        print(f"Thành công! Đã tìm thấy {len(result)} dòng thiếu abstract.")
        print(f"File đã được lưu tại: {output_path}")
        print("\nPreview 5 dòng đầu tiên:")
        print(result.head())

    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

# Thực thi code
export_missing_abstract_dois('null_values_test_report.csv', 'test (2).csv', 'missing_abstracts_doi_test.csv')
