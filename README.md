Valores utilizados:
- Alpha: 0.12
- b: 4
- w: -1

Considerações:
- Alpha de 0.01 até 0.11 vai melhorando progressivamente a convergência do EQM. A partir de 0.12, EQM começa a divergir, portanto o melhor valor para alpha parece ser 0.11;
- EQM final converge em 0,575;
- b converge para 4;
- w converge para -0.3;
- Se forem utilizados valores b=4 e w=-1, o que não seria um chute muito irrealista, visto que são números inteiros, o modelo levaria 50 iterações para ter uma convergência com uma rpecisão razoavelmente boa.