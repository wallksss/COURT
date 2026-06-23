# Plano de execucao do projeto

## Direcionamento extraido do PDF

O projeto deve classificar paginas/documentos juridicos do STF em cinco categorias: Acordao (0), ARE (1), Despacho (2), RE (3) e Sentenca (4). A implementacao deve ficar em um unico `main.ipynb`, com funcoes auxiliares em scripts Python, e precisa cobrir analise de dados, pre-processamento, representacao esparsa e densa, tarefas basicas de PLN, modelos classicos, modelos profundos/Transformers e analise de resultados. A metrica central da competicao e F1-Score.

## Inspiracao do repositorio de referencia

O repositorio `loren-gc/SDUWPS-ML-Classifier` separa a implementacao em `IMPLEMENTACAO/main.ipynb`, `dataset/`, `figs/` e `scripts/`, mantendo o notebook como narrativa e os scripts como camada reutilizavel. A mesma ideia foi aplicada aqui: o notebook chama funcoes em `scripts/`, os dados ficam em `data/`, modelos em `models/` e tabelas/submissoes em `outputs/`.

## Arquitetura modular

1. Entrada de dados:
   `analise_exploratoria.load_competition_data` carrega `train.csv`, `test.csv` e `sample_submission.csv`. Esta etapa valida o contrato de entrada sem depender do restante do fluxo.

2. Analise exploratoria:
   `validate_schema`, `summarize_dataset`, `add_text_statistics`, `text_length_percentiles`, `duplicate_text_report`, `get_top_terms` e funcoes de plotagem produzem diagnosticos de classe, tamanho, duplicatas, ruido e vocabulario. Se a analise mudar, os modelos continuam recebendo os mesmos dataframes.

3. Pre-processamento:
   `fix_mojibake`, `clean_body`, `TextPreprocessConfig`, `clean_text`, `add_clean_body_column` e `add_clean_text_column` encapsulam a limpeza minima do CSV e a limpeza juridica completa. A limpeza minima preserva tokens como `ARTIGO_102`, `EMAIL` e `RECURSO_EXTRAORDINARIO`; a coluna limpa (`Body_clean`) continua sendo a interface entre pre-processamento e treino.

4. Features juridicas e tarefas basicas de PLN:
   `extract_legal_signal_features` cria sinais leves de dominio. `extract_basic_nlp_features` usa spaCy para PoS/NER quando disponivel e cai para features juridicas quando o modelo de lingua nao esta instalado.

5. Representacao esparsa:
   `build_sparse_model_zoo` testa TF-IDF de palavras, caracteres e uma uniao word+char. Esse bloco e robusto para OCR porque char n-grams toleram erros de digitacao/extracao.

6. Modelos classicos:
   `evaluate_model_zoo` compara Dummy, ComplementNB, Regressao Logistica, LinearSVC, LinearSVC calibrado e SGD. Cada modelo pode ser trocado sem alterar o notebook. O TF-IDF forte usa n-gramas de palavras 1-3 e caracteres 3-5.

7. Predicoes OOF, ensemble e sequencia:
   `make_oof_probabilities` salva probabilidades fora da dobra para treino e probabilidades medias para teste. `optimize_ensemble_weights` combina TF-IDF e, quando existirem, probabilidades LegalBERT/BERTimbau. `duplicate_prior_probabilities` e `make_oof_duplicate_prior_probabilities` adicionam prior por texto sem usar o rotulo da propria validacao. `optimize_transition_lambda_oof`, `estimate_transition_matrix`, `viterbi_decode` e `transductive_viterbi_submission` implementam a etapa Viterbi/HMM em ordem de `Id` com rotulos de treino fixos.

8. Representacao densa e Transformers:
   `auto_runtime_profile` detecta CPU, CUDA/NVIDIA ou Apple MPS e ajusta batch, precisao e dispositivo sem exigir flags manuais. O pipeline completo continua sendo executado; em CPU ele apenas fica mais lento. `build_chunked_sentence_embedding_matrix` divide documentos longos em chunks sobrepostos antes de gerar embeddings com SentenceTransformers, reduzindo truncamento. `fine_tune_transformer_classifier` isola o fine-tuning com HuggingFace para modelos como BERTimbau, Albertina, ModernBERT e Longformer.
   O fine-tuning agora usa truncamento `head+tail` para manter o inicio e o final dos documentos em janelas de 512 tokens.

9. Auditoria de rotulos e pseudo-rotulagem:
   `detect_label_issues_with_cv` procura possiveis rotulos incorretos com predicoes fora da dobra. `pseudo_label_unlabeled_samples` reaproveita amostras com `Category = -1` apenas quando a confianca do modelo e alta. `build_training_set_with_pseudo_labels` combina rotulos originais, pseudo-rotulos e correcoes conservadoras mantendo rastreabilidade.

10. Avaliacao e analise:
   `analise_resultados` ranqueia modelos, plota metricas, matriz de confusao, relatorio por classe e erros. `save_experiment_report` salva rankings em CSV, JSON e Markdown. A etapa usa as predicoes geradas, portanto pode ser refeita sem retreinar tudo.

11. Submissao:
   `generate_submission` treina o melhor modelo em todo o treino, prediz `test.csv` e salva `outputs/submission.csv` com colunas `Id,Category`. A nova rota CSV-only salva `outputs/submission_sota_csv_viterbi.csv`, que aplica ensemble, prior por duplicata e Viterbi transdutivo sem acessar rotulos de teste.

## Experimentos recomendados

1. Baseline obrigatorio:
   Dummy e ComplementNB com TF-IDF para estabelecer o minimo.

2. Forte baseline classico:
   Regressao Logistica com TF-IDF word n-grams e LinearSVC com char n-grams.

3. Modelo combinado:
   Regressao Logistica com `FeatureUnion` de word TF-IDF + char TF-IDF.

4. Rota CSV-only competitiva:
   Gerar `oof_tfidf.npy` e `test_tfidf.npy`, incluir probabilidades Transformer opcionais quando existirem, otimizar pesos por macro F1, combinar com prior de duplicata e aplicar Viterbi por `Id`.

5. Embeddings:
   SentenceTransformer multilingue com chunking e classificador linear, util quando houver recursos para baixar modelos. Essa versao e preferivel ao encoding direto do texto inteiro porque documentos juridicos longos tendem a exceder a janela do encoder.

6. Transformer:
   Fine-tuning de BERTimbau/Albertina para textos truncados em 512 tokens. Quando houver CUDA, o treino usa GPU NVIDIA; quando houver Apple Silicon, usa MPS; quando nenhum acelerador estiver disponivel, usa CPU com batch menor. Assim o mesmo notebook roda sem flags em qualquer maquina.

7. Ajuste eficiente:
   Quando houver GPU, testar PEFT/LoRA/QLoRA para reduzir custo, principalmente se um encoder/LLM maior for usado.

8. Dados sem rotulo e rotulos ruidosos:
   Usar pseudo-rotulagem com limiar conservador para `Category = -1` e salvar uma auditoria dos possiveis erros de rotulagem. A correcao automatica deve ficar restrita a casos de confianca muito alta, mantendo os arquivos de auditoria para revisao manual.

## Referencias consultadas

- Repositorio de referencia: https://github.com/loren-gc/SDUWPS-ML-Classifier
- Kaggle da competicao: https://www.kaggle.com/competitions/ufscar-pln2026-pf/data
- BERT-CRF em portugues / BERTimbau: https://arxiv.org/abs/1909.10649
- Longformer para documentos longos: https://arxiv.org/abs/2004.05150
- SetFit e sentence embeddings eficientes: https://arxiv.org/abs/2209.11055
- Albertina PT-BR: https://arxiv.org/abs/2305.06721
- QLoRA: https://arxiv.org/abs/2305.14314
- ModernBERT: https://arxiv.org/abs/2412.13663
- NorBERTo / ModernBERT em portugues: https://arxiv.org/abs/2605.00086
