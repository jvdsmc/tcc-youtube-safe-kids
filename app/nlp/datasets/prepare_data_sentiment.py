"""
Script de preparação de dados para Análise de Sentimentos (AS).

Este script prepara os dados para treinamento do modelo de análise de sentimentos,
dividindo o dataset em conjuntos de treino, validação e teste.

*** Versão Modificada com Oversampling para balancear o treino ***
"""

import pandas as pd
from sklearn.utils import resample  # <-- ADICIONADO PARA OVERSAMPLING
from sklearn.model_selection import train_test_split
import logging
import sys
import os

# Adiciona o diretório raiz ao path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from app.nlp.config import PATHS, DATA_SPLIT
from app.nlp.utils.data_utils import DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mapeamento de labels de texto para números
LABEL_MAPPING = {
    'Negativo': 0,
    'Neutro': 1,
    'Positivo': 2
}

def map_labels_to_numbers(labels: list) -> list:
    """
    Converte labels de texto para números.
    
    Args:
        labels: Lista de labels em formato texto
        
    Returns:
        Lista de labels em formato numérico
    """
    numeric_labels = []
    for label in labels:
        if isinstance(label, str):
            numeric_labels.append(LABEL_MAPPING.get(label, -1))
        else:
            numeric_labels.append(label)
    return numeric_labels

def prepare_data():
    """
    Prepara os dados para treinamento do modelo de análise de sentimentos.
    
    Returns:
        Tuple com (train_texts, train_labels, val_texts, val_labels, test_texts, test_labels)
    """
    # Caminho do corpus
    corpus_path = PATHS['corpus_file']
    
    # Cria processador de dados
    processor = DataProcessor(corpus_path)
    
    # Carrega corpus
    processor.load_corpus()
    
    # Extrai dados para a tarefa AS
    texts, labels = processor.get_task_data('AS')
    
    # Converte labels de texto para números
    numeric_labels = map_labels_to_numbers(labels)
    
    # Remove amostras com labels inválidos (-1)
    valid_indices = [i for i, label in enumerate(numeric_labels) if label != -1]
    texts = [texts[i] for i in valid_indices]
    numeric_labels = [numeric_labels[i] for i in valid_indices]
    
    logger.info(f"Total de amostras válidas: {len(texts)}")
    logger.info(f"Distribuição de classes (antes da divisão): {pd.Series(numeric_labels).value_counts().to_dict()}")
    
    # Divisão: 80% treino, 10% validação, 10% teste
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, numeric_labels,
        test_size=0.2,
        random_state=DATA_SPLIT['random_state'],
        stratify=numeric_labels if DATA_SPLIT['stratify'] else None
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels,
        test_size=0.5,
        random_state=DATA_SPLIT['random_state'],
        stratify=temp_labels if DATA_SPLIT['stratify'] else None
    )

    # --- INÍCIO DO CÓDIGO DE OVERSAMPLING (BALANCEAMENTO) ---
    logger.info("Balanceando o conjunto de treino com Oversampling...")
    logger.info(f"Distribuição de treino (antes): {pd.Series(train_labels).value_counts().to_dict()}")

    # Converter para DataFrame para facilitar
    train_df = pd.DataFrame({'text': train_texts, 'label': train_labels})

    # Separar classes
    df_neg = train_df[train_df['label'] == 0]
    df_neu = train_df[train_df['label'] == 1]
    df_pos = train_df[train_df['label'] == 2]

    # Pegar a contagem da classe majoritária (Negativo)
    # No seu output, o treino Negativo (0) tinha 843 amostras
    max_count = len(df_neg)
    logger.info(f"Classe majoritária (Negativo) tem {max_count} amostras. Balanceando outras classes para este número.")

    # Fazer Upsample (reamostragem com reposição) das classes minoritárias
    df_neu_upsampled = resample(df_neu,
                              replace=True,     # Amostragem com reposição
                              n_samples=max_count,  # Para igualar à classe majoritária
                              random_state=DATA_SPLIT['random_state']) # Reprodutibilidade

    df_pos_upsampled = resample(df_pos,
                              replace=True,
                              n_samples=max_count,
                              random_state=DATA_SPLIT['random_state'])

    # Combinar de volta (agora balanceado)
    train_df_balanced = pd.concat([df_neg, df_neu_upsampled, df_pos_upsampled])
    
    # Embaralhar os dados combinados
    train_df_balanced = train_df_balanced.sample(frac=1, random_state=DATA_SPLIT['random_state']).reset_index(drop=True)

    # Converter de volta para listas
    train_texts = train_df_balanced['text'].tolist()
    train_labels = train_df_balanced['label'].tolist()

    logger.info(f"Tamanho do conjunto de treino após balanceamento: {len(train_texts)}")
    logger.info(f"Nova distribuição de treino (depois): {pd.Series(train_labels).value_counts().to_dict()}")
    # --- FIM DO CÓDIGO DE OVERSAMPLING ---
    
    logger.info(f"Divisão dos dados (final):")
    logger.info(f"  Treino: {len(train_texts)} amostras")
    logger.info(f"  Validação: {len(val_texts)} amostras")
    logger.info(f"  Teste: {len(test_texts)} amostras")
    
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

if __name__ == "__main__":
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = prepare_data()
    
    print(f"\n=== Resumo dos Dados Preparados ===")
    print(f"Treino: {len(train_texts)} amostras")
    print(f"Validação: {len(val_texts)} amostras")
    print(f"Teste: {len(test_texts)} amostras")
    
    print(f"\n=== Distribuição de Classes ===")
    print(f"Treino: {pd.Series(train_labels).value_counts().to_dict()}")
    print(f"Validação: {pd.Series(val_labels).value_counts().to_dict()}")
    print(f"Teste: {pd.Series(test_labels).value_counts().to_dict()}")
    
    print("\n=== Exemplos de Textos ===")
    # Pode dar erro se o treino estiver vazio, mas com oversampling é seguro
    if len(train_texts) > 0:
        print(f"Treino - Texto 1: {train_texts[0]}")
        print(f"Treino - Label 1: {train_labels[0]} ({['Negativo', 'Neutro', 'Positivo'][train_labels[0]]})")
    if len(val_texts) > 0:
        print(f"\nValidação - Texto 1: {val_texts[0]}")
        print(f"Validação - Label 1: {val_labels[0]} ({['Negativo', 'Neutro', 'Positivo'][val_labels[0]]})")