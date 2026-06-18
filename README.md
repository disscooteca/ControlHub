# ControlHub - Simulador e Controlador de Sistemas Dinâmicos

O **ControlHub** é uma plataforma integrada de simulação e controle para fins educacionais. O projeto consiste em uma interface gráfica desenvolvida em **Python (Streamlit)** que se comunica via **Bluetooth Low Energy (BLE)** com uma **ESP32**, permitindo, além de simular teoricamente e estudar os sistemas **Pêndulo Invertido** e **Bola e Bastão**, também permite monitorar e controlar esses sistemas físicos reais.

---

## ⚙️ Funcionalidades do ControlHub

Permite simular, testar e aprender o comportamento dos sistemas para, a partir disso, aplicar a teoria de controle a fim de alcançar a resposta desejada.

Isso é feito usando ambientes virtuais que simulam os sistemas e permitem que o programa dê feedback dos inputs do usuário, como gráficos de desempenho e dados relevantes a depender do que o usuário quiser.

---

## 🚀 Funcionalidades dos Kits

### 1. Pêndulo Invertido
* **Modo Gamificação:** O usuário controla o motor manualmente através de um potenciômetro no console físico enquanto o sistema monitora o erro de equilíbrio.
* **Modo Feedback:** Controle automático para manter o pêndulo na posição vertical (180°) baseado em ganho configurável recebido via BLE.

### 2. Bola e Bastão
* **Modo Gamificação:** Controle dinâmico do servo motor via potenciômetro para tentar equilibrar a bola na posição desejada.
* **Modo PID:** Implementação de controle PID clássico e em tempo real. Os parâmetros $K_p$, $K_i$ e $K_d$ são enviados diretamente da interface Python para a ESP32.

---

## 🛠️ Esquemático de Hardware (ESP32)

Se você deseja replicar o kit físico, faça as conexões da ESP32 utilizando os pinos definidos no firmware:

| Componente | Elemento | Pino ESP32 |
| :--- | :--- | :--- |
| **Motor JGA (Pêndulo)** | PWM (Velocidade) | GPIO 4 |
| | Entrada A (mla) | GPIO 18 |
| | Entrada B (mlb) | GPIO 19 |
| **Encoder Pêndulo** | Canal A | GPIO 22 *(Com Interrupção)* |
| | Canal B | GPIO 21 |
| **Servo Motor (Bola Bastão)**| Sinal PWM | GPIO 13 |
| **Sensor Ultrassônico** | Trigger (trig) | GPIO 27 |
| | Echo (echo) | GPIO 33 |
| **Console Físico** | Chave Início/Fim | GPIO 17 *(Pull-down)* |
| | LED Indicador | GPIO 23 |
| | Potenciômetro | GPIO 34 |

---

## 📦 Configuração e Instalação

### 1. Firmware da ESP32
1. Abra o código da ESP32 na Arduino IDE.
2. Certifique-se de ter instalado as seguintes bibliotecas:
   * `ESP32Servo`
   * `HCSR04` (Ultrasonic Distance Sensor)
3. Altere os UUIDs de serviço e característica no código caso utilize um aplicativo mobile genérico, ou mantenha os padrões para conectar com o app em Streamlit:
   * **Service UUID:** `4fafc201-1fb5-459e-8fcc-c5c9c331914b`
   * **Characteristic UUID:** `beb5483e-36e1-4688-b7f5-ea07361b26a8`
4. Carregue o código na ESP32.

### 2. Interface Python (Streamlit)
Certifique-se de ter o Python 3.9+ instalado em sua máquina.

* Execute ControlHub.exe
* * Encontrado nos releases

 ou
 
1. Clone o repositório:

   git clone [https://github.com/disscooteca/ControlHub.git](https://github.com/disscooteca/ControlHub.git)
   cd ControlHub
   
3. Instale as dependências necessárias:

  pip install -r requirements.txt
  
4. Execute o servidor do Streamlit:
  streamlit run main.py

---

### 📈 Como Funciona o Fluxo
1. O usuário seleciona o sistema desejado na interface gráfica (Streamlit).

2. O usuário segue as perguntas nas ordens apresentadas, fazendo as simulações e plotando gráficos.

3. Para interagir com o kit, mande o comando via ControlHub ao clicar o botão para o modo escolhido.
   
4. A ESP32 recebe o comando, decodifica os parâmetros e aguarda o acionamento da Chave do Console (GPIO 17).

5. Ao ligar a chave, a ESP32 inicia uma coleta rígida de dados temporizada de 666 leituras a cada 30ms.

6. Finalizada a coleta, a ESP32 calcula o erro total acumulado e devolve o resultado via Notificação BLE para o Streamlit plotar os gráficos de desempenho.
