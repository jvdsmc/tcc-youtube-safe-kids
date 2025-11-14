"""
Script de preparação de dados para Análise de Sentimentos (AS).

Este script prepara os dados para treinamento do modelo de análise de sentimentos,
dividindo o dataset em conjuntos de treino, validação e teste.
"""

import pandas as pd
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
    logger.info(f"Distribuição de classes: {pd.Series(numeric_labels).value_counts().to_dict()}")
    
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
    
    logger.info(f"Divisão dos dados:")
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
    print(f"Treino - Texto 1: {train_texts[0]}")
    print(f"Treino - Label 1: {train_labels[0]} ({['Negativo', 'Neutro', 'Positivo'][train_labels[0]]})")
    print(f"\nValidação - Texto 1: {val_texts[0]}")
    print(f"Validação - Label 1: {val_labels[0]} ({['Negativo', 'Neutro', 'Positivo'][val_labels[0]]})")

