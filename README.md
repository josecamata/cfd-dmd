# CFD-DMD

## Sobre o Projeto
Este repositório implementa a **Decomposição Modal Dinâmica (Dynamic Mode Decomposition - DMD)**, uma técnica usada para analisar sistemas dinâmicos complexos. A DMD é amplamente aplicada na dinâmica de fluidos computacional (CFD), processamento de sinais e aprendizado de padrões em sistemas dinâmicos.

## O que é Decomposição Modal Dinâmica (DMD)?
A Decomposição Modal Dinâmica (DMD) é uma técnica baseada em análise de dados que decompõe um sistema dinâmico em modos oscilatórios, permitindo capturar padrões temporais e espaciais a partir de medições discretas.

A DMD fornece informações sobre:
- Estruturas modais dominantes no sistema
- Frequências e taxas de crescimento ou decaimento dos modos
- Previsão da dinâmica do sistema com base nos modos extrídos

## Aplicabilidade da DMD
A DMD tem sido amplamente utilizada em diversas áreas, incluindo:
- **Dinâmica de fluidos computacional (CFD)**: Para identificar padrões em escoamentos turbulentos e sistemas aerodinâmicos.
- **Processamento de sinais e imagens**: Para extração de informações essenciais de dados temporais e espaciais.
- **Sistemas biológicos**: Para modelagem de processos celulares e fisiológicos.
- **Engenharia de controle**: Para redução de ordem de modelos de sistemas dinâmicos complexos.

---
## Estrutura do Repositório
```
CFD-DMD/
├── python/                   # Scripts em Python para execução da DMD
├── .gitignore                 # Arquivos ignorados pelo Git
├── README.md                  # Documento atual
├── clean_output.sh            # Script para limpar arquivos de saída
├── setup.sh                   # Script para configurar o ambiente
```

## Configuração do Ambiente
Para configurar o ambiente de execução, siga os seguintes passos:

1. Instale as bibliotecas necessárias:
   ```bash
   pip install opencv-python gdown
   ```

2. Execute o script para importar o conjunto de dados:
   ```bash
   python3 python/import_dataset.py
   ```

3. Configure o ambiente com o script `setup.sh`:
   ```bash
   bash setup.sh
   ```

4. Caso precise limpar os arquivos de saída gerados durante a execução, utilize:
```bash
bash clean_output.sh
```

## Contribuição
Contribuições são bem-vindas! Se desejar adicionar novos recursos ou corrigir erros, siga os passos:
1. Faça um fork do repositório.
2. Crie uma branch com suas alterações:
   ```bash
   git checkout -b minha-nova-feature
   ```
3. Commit suas modificações:
   ```bash
   git commit -m "Adicionando nova feature"
   ```
4. Envie para seu repositório:
   ```bash
   git push origin minha-nova-feature
   ```
5. Abra um Pull Request.

## Licença
Este projeto é distribuído sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## Agradecimentos

Este projeto recebu suporte financeiro da FAPEMIG (APQ-01123-21).

