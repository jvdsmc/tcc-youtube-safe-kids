"""
Script de treinamento para Análise de Sentimentos (AS).

Este script treina o modelo BERTimbau para análise de sentimentos usando
os dados preparados e a configuração AS_best obtida através de grid search.
"""

import logging
import sys
import os

# Adiciona o diretório raiz ao path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from app.nlp.models.bertimbau_sentiment import BertimbauSentiment
from app.nlp.datasets.prepare_data_sentiment import prepare_data

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Função principal de treinamento."""
    logger.info("=" * 60)
    logger.info("Iniciando treinamento do modelo de Análise de Sentimentos")
    logger.info("=" * 60)
    
    # Prepara dados
    logger.info("Preparando dados...")
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = prepare_data()
    
    # Cria modelo
    logger.info("Criando modelo BERTimbau para Análise de Sentimentos...")
    model = BertimbauSentiment()
    
    # Treina modelo usando a configuração AS_best (melhores parâmetros obtidos via grid search)
    logger.info("Iniciando treinamento com configuração AS_best...")
    logger.info("Parâmetros: epochs=5, batch_size=8, lr=5e-5, warmup_steps=500")
    
    results = model.train_model(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        config_name='AS_best',  # Usa a configuração otimizada
        experiment_name='sentiment_v1'
    )
    
    logger.info("=" * 60)
    logger.info("Treinamento concluído!")
    logger.info("=" * 60)
    logger.info(f"Modelo salvo em: {results['model_path']}")
    logger.info(f"Métricas finais: {results['final_metrics']}")
    logger.info(f"Configuração usada: {results['training_config']}")
    
    print("\n" + "=" * 60)
    print("RESUMO DO TREINAMENTO")
    print("=" * 60)
    print(f"Modelo salvo em: {results['model_path']}")
    print(f"\nMétricas no conjunto de validação:")
    for metric, value in results['final_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()

