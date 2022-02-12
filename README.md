# Algoritmo de Machine Learning para Previsão de Ratings

<p>Este projeto foi realizado no âmbito da cadeira de Aprendizagem automárica em Sistemas Empresariais no curso de MiEGSI. O objetivo do projeto é a familiarização com conceitos de machine learning utilizando a metodologia CRISP-DM de forma a criar um algoritmo capaz de prever o Rating ESRB de um jogo dados o resto dos dados como o caráter assuntual do jogo.</p>
<p>Para dar o projeto como finalizado tomamos a liberdade de implementar o modelo desenvolvido numa API Web de forma a poder ser utilizada por toda a gente e como "Proof of Concept" do nosso trabalho</p>
<p>Pode encontrar a implementação atual do nosso modelo em <a>https://esrb-rating-predictor.herokuapp.com/</a></p>

## Description

<p>Decidimos realizar o projeto utilizando a linguagem de programação Python, que é popular por ser particularmente boa e ter muito apoio em projetos de Machine Learning. Dentro da linguagem Python utilizamos o, MUITO, popular Pandas para manusear todos os dados em forma de DataFrame e o scikit-learn para todas as atividades de machine learning como modelos e métodos de avaliação.</p>
<p>Sendo assim seguimos a metodologia CRISP-DM para realizar este projeto.</p>

## CRISP-DM
### Business Understanding
```
O Business Understanding trata-se da fase onde explorámos a empresa em estudo de forma a consolidar os seus objetivos e o porquê de se estar a realizar um projeto de Machine Learning.

De seguida, baseando-se nos objetivos da empresa criamos os nossos objetivo de Data Mining onde vamos especificar exatamente o que queremos que o nosso algoritmo seja capaz de fazer, assim como definimos metas ideais para a gestão de sucesso.

Por fim realizamos um plano de projeto de forma a situar no tempo cada uma das fases do projeto.

No nosso caso a empresa em questão trata-se da sony, que quer prever os ESRB_Ratings dos seus jogos antes de estes serem lançados de forma a poderem realizar campanhas de marketing de forma mais eficiente e livres de erros.

Sendo assim o nosso objetivo de Data Mining é, dado as flags de um dado jogo, prever qual será o rating que lhe será atribuido.
```
### Data Understanding 
```
Esta fase serve para nos familiarizarmos com os dados que vamos tentar prever. Isto é essencial visto que quanto mais familiarizado estivermos com os dados melhores decisões de inclusão ou exclusão de certos parâmetros iremos fazer.

Para realizar este estudo fizemos uma descrição de todos os dados disponibilizados onde explorámos aprofundadamente o que cada parâmetro do nosso dataset significava

Fizémos também uma vistoria inicial dos dados ao qual chamámos Exploração e Verificação da qualidade dos dados onde verificamos o nosso dataset para valores nulos ou outliers(valores fora da norma), a sua cardinalidade e de uma forma geral com que se parece cada tipo de dado.
```
### Data Preparation
```
A fase de Preparação dos Dados é das mais importantes, senão a mais importante em todo o projeto, sendo que é nesta fase que trancamos virtualmente o desempenho do nosso algoritmo(mesmo sem saber).
Nesta fase iremos fazer o tratamento completo aos dados de forma a que estes estejam nas melhores condições para serem passados pelos modelos de machine learning.

Sendo assim fizemos a seleção dos atributos que consideramos relevantes para processamento nos modelos. Para isto realizamos um estudo das correlaçoes dos atributos, quer entre si, quer com o atributo que pretendemos prever(sobretudo com o atributo que pretendemos prever). Para realizar este estudo utilizamos técnicas quer manuais (como analisar a nuvem volumétrica das relações dos atributos) quer automáticas (como analisar os resultados de um GridSearch com todos os atributos utilizando os modelos).

Esta técnica automática da seleção dos atributos resultou em criarmos 3 cenários diferentes, cada um com um grupo diferente(mas, na mesma relevante) de dados.
```
### Modeling 
```
Para o processo de modelação utilizamos a técnica de Cross Validation de forma a correr múltiplos modelos em sequência e avaliar os resultados deles.

O cross validation é uma técnica utilizada para validar a estabilidade do nosso modelo de machine learning. É necessário ter uma forma de garantia de que o modelo resolveu a maioria dos padrões dos dados corretamente sem captar muito ruído obtendo um modelo com alta variância (para ser consistente com todos os datasets de teste possíveis) e com baixa “pré-configuração” (bias).
Sendo assim utilizamos a técnica de cross validation para avaliar o desempenho dos nossos modelos face a dados nunca antes vistos por eles (dados de teste).
Existem várias abordagens para o cross validation porém a que iremos usar é o Stratified KFold que se trata de uma extensão do KFold especialmente para problemas de classificação.
Esta técnica permite que, em vez de as divisões do dataset serem realizadas de forma aleatória, é assegurado que a percentagem de cada classe da variável target é mantida a mesma para cada fold, assim como no dataset original. Por exemplo, a partir da etapa de Exploração dos Dados, foi possível verificar que existe uma percentagem de 37% de dados da classe T, 23% da classe E, 20% da classe M e 20% da classe ET, sendo que em cada fold irá ser preservada essa mesma percentagem de dados de cada classe.
Esta técnica de cross validation é geralmente utilizada quando se verificam duas condições, sendo elas:
    • Quando se pretende preservar a percentagem de cada classe da variável target;
    • Quando possuímos poucos dados de treino.
Posto isto, como o grupo pretende manter a percentagem de cada classe da variável target e como possuímos poucos dados de treino decidimos utilizar esta técnica de cross validation, que permite assegurar que cada fold apresenta a mesma percentagem de dados de cada classe como no dataset original, evitando o problema de algum fold não apresentar dados de uma determinada classe, o que iria prejudicar a viabilidade dos resultados do modelo.

```
### Evaluation 
```
Na avaliação dos resultados iremos analisar os resultados obtidos e retirar as conclusões que forem necessárias. Comparar os resultados obtidos com os resultados previstos/esperados é necessário para averiguar o sucesso obtido.
```

### Deplyment 
```
A fase de deployment consiste em utilizar o/s modelo/s obtido/s nas fases anteriores, completos, treinados, otimizados num ambiente de produção onde os seus outputs criarão valor para a organização em questão.
Em relação ao nosso projeto iremos exportar o modelo que obtivémos através das fases anteriores do CRISP-DM, modelo este que está pronto a ser utilizado para fazer previsões de dados.
No nosso caso exportámos o modelo e utilizámo-lo, quer na previsão dos dados acima de 2019 (para poder analisar os resultados), quer na criação de uma Web API que recebe parâmetros de jogos e utiliza o modelo para prever o respetivo rating para poder enviar a resposta prevista.
```

## Getting Started

### Dependencies

* Windows 10/11, Linux, MacOS
* Python instalado na máquina que irá correr
* Jupyter Notebooks
* Pandas
* Scikit-learn (SKLearn)
* Flask

### Installing (Jupyter Notebook)

* Clonar o repositório
```
git clone https://github.com/JoaoGuedes01/ESRB_Rating_Predictor_CRISPDM.git
```
* Iniciar o serviço do Jupyter
```
cd jupyter-notebooks
jupyter notebook
```
No final é apenas necessário navegar até ao notebook onde terá disponível todos os métodos separados convenientemente por divisões que poderá correr e explorar.

### Installing (Web API)

* Clonar o repositório
```
git clone https://github.com/JoaoGuedes01/ESRB_Rating_Predictor_CRISPDM.git
```

* Correr a aplicação app
```
python app.py
```
<p>Esta aplicação irá importar e carregar o modelo da pasta <b>model</b> e torna-lo utilizável através de pedidos HTTP, assim como disponibilizar uma interface gráfica onde é possível experimentar o modelo.</p>
<p>Após correr o ficheiro <b>app.py</b> pode dirigir-se a <b>http://localhost:33507</b> para aceder à interface gráfica</p>


## Help

Caso tenha problemas ao correr as funções do notebook proceda à instalação dos modulos em falta utilizando o comando:
```
pip install <modulo_em_falta>
```

## Authors
- [João Guedes](https://github.com/JoaoGuedes01)
- [João Pedro](https://github.com/joaopedrofg7)
- [José Melo](https://www.linkedin.com/in/jos%C3%A9pmelo/)
- [Rui Gomes](https://github.com/ruigomes99)

## Contribution
 - [Pedro Pereira](https://www.linkedin.com/in/pedrojosepereira/)

## Version History
* 0.1
    * Initial Release

## License

This project is licensed under the MIT License - see the LICENSE.md file for details