import pandas as pd
import os
import glob

def main():
    raw_dir = '/home/felipe/Projeto/Portfolio/Portfolio2/multi_brand_marketing/data/raw'
    processed_dir = '/home/felipe/Projeto/Portfolio/Portfolio2/multi_brand_marketing/data/processed'

    # Cria o diretório processado se não existir
    os.makedirs(processed_dir, exist_ok=True)

    # Lista todos os arquivos CSV usando o caminho absoluto
    csv_files = glob.glob(os.path.join(raw_dir, '*.csv'))

    if not csv_files:
        print("Nenhum arquivo CSV encontrado diretório raw.")
        return

    dfs = []
    for file in csv_files:
        # Extrai o nome da marca a partir do nome do arquivo
        filename = os.path.basename(file)
        brand = filename.split('_')[0].capitalize()
        
        # Lê o dataframe
        df = pd.read_csv(file)
        
        # Opcional: Adiciona a coluna da marca, caso não exista no df original, para identificar a origem
        df['Brand'] = brand
        
        dfs.append(df)

    # Unifica os dataframes
    unified_df = pd.concat(dfs, ignore_index=True)

    # Salva o arquivo unificado
    output_path = os.path.join(processed_dir, 'unified_campaign_data.csv')
    unified_df.to_csv(output_path, index=False)

    print(f"Bases unificadas com sucesso!")
    print(f"Arquivo salvo em: {output_path}")
    print(f"Total de registros: {len(unified_df)}")

if __name__ == '__main__':
    main()
