import streamlit as st
import plotly.graph_objects as go
import numpy as np
import control as ctl
from control import (TransferFunction)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import requests
import sys
import os


def obter_caminho_arquivo(nome_arquivo):
    """
    Retorna o caminho absoluto do ficheiro.
    Funciona tanto no ambiente de desenvolvimento como no .exe compilado.
    """
    if hasattr(sys, '_MEIPASS'):
        # Se estiver a correr pelo .exe, procura na pasta temporária
        return os.path.join(sys._MEIPASS, nome_arquivo)
    
    # Se estiver a correr no VS Code, procura na pasta normal
    return os.path.join(os.path.abspath("."), nome_arquivo)

def enunciado_questao1(type):
    if type == "Bola bastão":
        st.markdown("### Questão 1: Bola Bastão - Pygame")

        st.markdown(f"""
        Cada aluno do grupo deverá jogar o jogo da bola bastão desenvolvido em Pygame e coletar os dados da partida.
        """)

        st.info("O objetivo é deixar a bola no meio do bastão. Quem obtiver o menor erro médio vence!")

        st.warning("⚠️ A tentativa não vale se a bolinha cair do bastão.")

    if type == "Pêndulo simples invertido":
        st.markdown("### Questão 1: Pêndulo Simples Invertido - Pygame")

        st.markdown(f"""
        Cada aluno do grupo deverá jogar o jogo do pêndulo simples invertido desenvolvido em Pygame e coletar os dados da partida.
        """)

        st.info("O objetivo é manter o pêndulo em posição vertical para cima. Quem obtiver o menor erro médio vence!")

def enunciado_questao2(type):
    if type == "Bola bastão":
        st.markdown("### Questão 2: Modelagem e Função de Transferência")

        st.markdown("a)")

        st.markdown(f"""
        sendo:\n
                    a: aceleração (m/s²)
                    m: massa da bola (kg)
                    g: gravidade (m/s²)
                    d: offset do braço da alavanca (m)
                    L: comprimento do bastão (m)
                    R: raio da bola (m)
                    r: distância da bola até extremidade fixa (m)
                    x: distância da bola até o centro da barra (m)
                    J: momento de inércia da bola maciça 2*M*R²/5 (kg*m²)
                    alpha: ângulo do bastão (rad)
                    theta: ângulo do motor (rad)
        """)
        
        st.markdown("""
        O comportamento do sistema bola-bastão pode ser representado pela equação diferencial:
        """)

        # Equação principal da imagem
        st.latex(r"a \cdot m + a \cdot \frac{J}{R^2} + m \cdot g \cdot \operatorname{sen}(\alpha) = 0")
        
        st.markdown("""
        **Pergunta:** \n\nComo obter a função de transferência considerando a entrada como o seno do ângulo do bastão $(sen(alpha))$ e a saída sendo a posição da bola ($x$)?      
        """)

        st.markdown("b)")

        st.markdown(r"""
        Considerando que $b = \frac{m*g}{m+\frac{J}{R^2}}$, como fica a função de transferência em malha aberta?
        """)

        st.markdown("c)")

        st.markdown(r"""
        Considerando que $G_{mf} = \frac{G}{1 + G}$, como fica a função de transferência em malha fechada tendo como G a função de malha aberta da questão (b)?
        """)

        st.markdown("d)")

        st.markdown(r"""
        A forma generalizada de se calcular a função de transferência em malha fechada considera um controlador C(s) em série com a planta e a realimentação como sendo H(s), obtendo assim:
        $G_{mf} = \frac{C(s)*G(s)}{1 + C(s)*G(s)*H(s)}$. Pensando no comportamento do sistema, como devem ser os valores de C(s) e H(s)?
        """)

        st.info("💡 **Dica:** Considere o exemplo da bolinha estando a direita do ponto de referência, o erro, que é desejado - atual, será negativo e isso deve mandar um sinal positivo para o motor, uma vez que ele precisa de inclinação positiva para movimentar a bola para a esqueda")

    if type == "Pêndulo simples invertido":
        st.markdown("### Questão 2: Modelagem e Função de Transferência")

        st.markdown("a)")

        st.markdown(f"""
        sendo:\n
        * **$\\theta$**: ângulo do bastão (rad)
        * **$\dot{{\\theta}}$**: velocidade angular (rad/s)
        * **$\ddot{{\\theta}}$**: aceleração angular (rad/s²)
        * **$m$**: massa do bastão (kg)
        * **$L$**: comprimento do bastão (m)
        * **$g$**: aceleração da gravidade (m/s²)   
        * **$b$**: coeficiente de viscosidade (N·m·s/rad)
        * **$I$**: momento de inércia do bastão de comprimento L e massa m distribuída uniformemente ($m \cdot L^2 / 3$)
        * **$\\mathcal{{T}}_{{ext}}$**: torque externo aplicado pelo motor (N·m)
        """)

        st.markdown("""
        O comportamento dinâmico do braço robótico (pêndulo invertido) é representado pela seguinte equação diferencial, considerando a força da gravidade aplicada no centro de massa ($L/2$):
        """)

        # Equação Diferencial Completa
        st.latex(r"\mathcal{T}_{total} = \mathcal{T}_{externo} - b \cdot \dot{\theta} + \mathcal{T}_{gravidade} = I \cdot \ddot{\theta} ")

        st.info("💡 **Dica:** Utilize a aproximação para pequenos ângulos onde $\operatorname{sen}(\\theta) \\approx \\theta$ para linearizar o sistema.")

        st.markdown(r"""
        **Pergunta:** Como obter a **função de transferência** em malha aberta $G(s) = \frac{\Theta(s)}{\mathcal{T}_{ext}(s)}$ , sendo a entrada como o torque externo e a saída como o ângulo $\theta$?
        """)

def enunciado_questao3(type):
    if type == "Bola bastão":
        st.markdown("### Questão 3: Linearização de Sistemas")
        
        st.markdown("""
        Sistemas de controle não lineares lidam com dinâmicas que não seguem o princípio da superposição, 
        onde a resposta não é sempre proporcional à entrada. Apesar desses sistemas serem muito comuns e 
        representarem diversos comportamentos cotidianos, essa não linearidade torna a análise e o 
        controle deles muito complexos. 
        
        Diante disso, uma estratégia é a **linearização**, que busca representar um sistema não-linear 
        por meio de um sistema linear em um ponto de operação.
        
        Levando em consideração que o $sen(\\alpha)$ transforma o sistema em não linear 
        e de acordo com o gráfico abaixo, que representa os polinômios da Série de Taylor para o seno, 
        qual é a opção válida para se **linearizar** esse sistema?
        """)

        # 1. Gerar os dados
        x = np.linspace(-np.pi, np.pi, 201)
        
        # Funções de Taylor
        f_sin = np.sin(x)
        f1 = x
        f3 = x - x**3/6
        f5 = x - x**3/6 + x**5/120
        f7 = x - x**3/6 + x**5/120 - x**7/5040
        
        # 2. Criar a figura do Plotly
        fig = go.Figure()

        # Adicionar cada linha (Trace)
        fig.add_trace(go.Scatter(x=x, y=f_sin, name="Original (seno)", line=dict(width=3, color='blue')))
        fig.add_trace(go.Scatter(x=x, y=f1, name="Primeira Ordem (Linear)"))
        fig.add_trace(go.Scatter(x=x, y=f3, name="Terceira Ordem"))
        fig.add_trace(go.Scatter(x=x, y=f5, name="Quinta Ordem"))
        fig.add_trace(go.Scatter(x=x, y=f7, name="Sétima Ordem", line=dict(color='yellow')))

        # 3. Configurações de Layout
        fig.update_layout(
            title="Polinômios de Taylor aplicados ao seno de α",
            xaxis_title="Ângulo [rad]",
            yaxis_title="sin(α)",
            xaxis=dict(range=[-np.pi, np.pi]),
            yaxis=dict(range=[-3.5, 3.5]), # Ajustado para manter a escala da imagem original
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            template="plotly_white", # Fundo limpo
            hovermode="x unified"    # Mostra todos os valores ao passar o mouse em um ponto X
        )

        col1, col2, col3 = st.columns([1,3,1])
        # 4. Mostrar no Streamlit
        col2.plotly_chart(fig)

        # Opções formatadas com LaTeX
        st.markdown("""
            \na) A melhor forma de se linearizar o seno é utilizar a função original $F_{original} = \sin(\\alpha)$, pois ela representa com exatidão o comportamento do sistema ao longo do tempo.
            \nb) A melhor forma de se linearizar o seno é utilizar o polinômio de primeira ordem $F_1(x) = x$, pois ele respeita o princípio da superposição. Porém, ele se aproxima da função original só para ângulos próximos de zero.
            \nc) A melhor forma de se linearizar o seno é utilizar o polinômio de terceira ordem $F_3(x) = x - \\frac{x^3}{6}$ pois, por mais que ele não respeite o princípio da superposição, sua resposta se assemelha ao original.
            \nd) A melhor forma de se linearizar o seno é utilizar o polinômio de sétima ordem $F_7(x) = x - \\frac{x^3}{6} + \\frac{x^5}{120} - \\frac{x^7}{5040}$ pois ele é o que melhor representa o sinal original sem criar distorção.
            \ne) A melhor forma de se linearizar o seno é utilizar o polinômio de maior ordem que o computador permite calcular, pois quanto maior a ordem, mais próximo da resposta do seno o polinômio estará.
        """)

    if type == "Pêndulo simples invertido":
        st.markdown("### Questão 3: Estabilidade e a Aproximação Linear")

        st.markdown(r"""
            Caso solte essa haste em qualquer posição que não seja ela perfeitamente alinhada para cima de forma, posição essa em que não há  
            torque causado pela gravidade, o que acontecerá com a bolinha? Considerando esse fato e que o desejado é que a 
            haste fique para cima, qual é um grande problema em se considerar sen(theta) aproximadamente igual a theta?
        """)

        # --- Gráfico Plotly ---

        # Gerando dados de -pi a pi
        theta = np.linspace(-np.pi, np.pi, 500)
        y_sin = np.sin(theta)
        y_linear = theta

        fig = go.Figure()

        # Adicionando sen(theta)
        fig.add_trace(go.Scatter(x=theta, y=y_sin, name=r'sen(θ)', line=dict(color='cyan', width=3)))

        # Adicionando theta (linear)
        fig.add_trace(go.Scatter(x=theta, y=y_linear, name=r'θ (Aproximação)', line=dict(color='red', dash='dash')))

        fig.update_layout(
            title="Comparação: sen(θ) vs θ",
            xaxis_title="Ângulo θ (radianos)",
            yaxis_title="Valor da Função",
            template="plotly_dark",
            yaxis=dict(range=[-3, 3]), # Limitando para ver o desvio
            xaxis=dict(
                tickmode='array',
                tickvals=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                ticktext=['-π', '-π/2', '0', 'π/2', 'π']
            )
        )

        # Mostrando no Streamlit
        col1, col2, col3 = st.columns([1, 3, 1])
        col2.plotly_chart(fig, use_container_width=True)

def enunciado_questao4(type, L=None, d=None):
    if type == "Bola bastão":
        st.markdown("### Questão 4: Relação Cinemática e Limites de Operação")

        st.markdown("""
        O comportamento do sistema é regido pela inclinação do bastão $\\alpha$, 
        embora o controle direto seja exercido apenas sobre o ângulo do braço do motor $\\theta$.
        
        A relação entre esses dois ângulos é dada pela geometria do mecanismo:
        """)

        st.latex(r"sen(\alpha) \cdot L = sen(\theta) \cdot d")

        st.info("Utilize a linearização encontrada na questão anterior para $sen(\\alpha)$.")

        st.markdown(f"""
        **Pergunta:** \n
        Encontre quais são os maiores valores de $\\alpha$ e $\\theta$ em radianos para um sistema com:
        * $L = {L}$ metros (Comprimento do bastão).
        * $d = {d}$ metros (Offset do braço da alavanca).
        
        A resposta encontrada faz sentido com relação à condição de linearização?
        """)

    if type == "Pêndulo simples invertido":
        st.markdown(f"""
                    Tendo uma haste em formato cilíndrico com os seguintes parâmetros:\n

                    L= 20cm \n 
                    diâmetro= 2cm\n 
                    densidade_haste = 1.25g/cm³ \n 
                    gravidade = 9.81m/s²\n
                    Qual o torque mínimo em Nm necessário que o motor precisa ter para conseguir dar uma volta completa com essa haste estando acoplado em uma de suas extremidades se desconsiderarmos o amortecimento viscoso (atrito)?
                    """)
        
        st.info("💡 **Dica:** Considere que a força da gravidade é aplicada na metade do comprimento L. Ademais, não use a aproximação $sen(\\theta) \\approx \\theta$ neste caso.")

def enunciado_questao5(type):
    if type == "Bola bastão":
        st.markdown("### Questão 5: Resposta no tempo para o sistema em Malha fechada")

        st.markdown("a)\n")

        st.markdown("Sabendo que a frequência natural e o fator de amortecimento de um sistema de segunda ordem pode ser obtido através de:\n")

        st.latex(r"T(s) = \frac{ganho}{s^2 + 2\zeta\omega_ns + \omega_n^2}")

        st.markdown("Quais são esses valores para a função de transferência em malha fechada do pêndulo simples invertido? O que eles dizem sobre a resposta do sistema?")

        st.info(r"💡 *Dica:** Considere $G_{mf} = \frac{b}{s^2+b}$ e $b = \frac{mg}{m+\frac{J}{R^2}}$ pois $C(s) = -1$")

        st.markdown("b)\n")

        st.markdown("A partir das fórmulas descritas a seguir, calcule  o tempo de acomodação, o tempo de pico, o tempo de subida, a ultrapassagem percentual e a frequência natural amortecida. Ademais, explique o que cada um desses parâmetros significa.\n")
        st.latex(r"T_s(2\%) = \frac{4}{\zeta \cdot \omega_n}")
        st.latex(r"T_p = \frac{\pi}{\omega_d} = \frac{\pi}{\omega_n \sqrt{1-\zeta^2}}")
        st.latex(r"T_r \approx \frac{1.8}{\omega_n}")
        st.latex(r"\%UP = e^{-\left(\frac{\zeta\pi}{\sqrt{1-\zeta^2}}\right)} \times 100")
        st.latex(r"\omega_d = \omega_n \sqrt{1 - \zeta^2}")

    if type == "Pêndulo simples invertido":
        st.markdown("### Questão 5: Resposta no tempo para o sistema em Malha fechada")

        st.markdown("a)\n")

        st.markdown("Sabendo que a frequência natural e o fator de amortecimento de um sistema de segunda ordem pode ser obtido através de:\n")

        st.latex(r"T(s) = \frac{ganho}{s^2 + 2\zeta\omega_ns + \omega_n^2}")

        st.markdown("Quais são esses valores para a função de transferência em malha fechada do pêndulo simples invertido? O que eles dizem sobre a resposta do sistema?")

        st.markdown("b)\n")

        st.markdown("A partir das fórmulas descritas a seguir, calcule  o tempo de acomodação, o tempo de pico, o tempo de subida, a ultrapassagem percentual e a frequência natural amortecida. Ademais, explique o que cada um desses parâmetros significa.\n")
        st.latex(r"T_s(2\%) = \frac{4}{\zeta \cdot \omega_n}")
        st.latex(r"T_p = \frac{\pi}{\omega_d} = \frac{\pi}{\omega_n \sqrt{1-\zeta^2}}")
        st.latex(r"T_r \approx \frac{1.8}{\omega_n}")
        st.latex(r"\%UP = e^{-\left(\frac{\zeta\pi}{\sqrt{1-\zeta^2}}\right)} \times 100")
        st.latex(r"\omega_d = \omega_n \sqrt{1 - \zeta^2}")

def enunciado_questao6(type):
    if type == "Bola bastão":
        st.markdown("### Questão 6: Sistema em Malha Aberta")

        st.markdown("""##### Faça as simulações indicadas nas questões abaixo e cole os prints no relatório.\n
                    """)
        
        st.warning("Os prints devem conter os inputs da simulação juntamente com as informações pedidas abaixo.")

        st.markdown("""
                    a) Simule o sistema para um valor de entrada degrau diferente de 0.00 e demais inputs à sua escolha.\n
                    b) Plote a resposta ao degrau para o mesmo degrau escolhido na simulação.\n
                    b.1 - Qual o motivo da resposta ao degrau estar diferente (se o valor escolhido for positivo a resposta vai para o menos infinito e vice-versa)?\n
                    c) Plote o mapa de polos e zeros e o lugar das raízes e justifique o porquê do sistema ser instável para qualquer degrau colocado na entrada.\n
                    d) Plote os diagramas de Bode e Nyquist e explique o que eles dizem sobre esse sistema em Malha Aberta.
                    """)
    
    if type == "Pêndulo simples invertido":
        st.markdown(r"""
Como visto nas questões anteriores, a linearização do sistema utilizando a aproximação para pequenos ângulos ($\sin(\theta) \approx \theta$) possui uma limitação severa: ela só é válida quando o pêndulo já está muito próximo da posição de equilíbrio vertical. 
                    
Para contornar esse problema e conseguir levantar o pêndulo a partir da sua posição de repouso (pendurado para baixo), utiliza-se a estratégia não-linear de **Swing-up** (balanço). Nela, a lei de controle atua bombeando energia para o sistema com base na velocidade angular ($\omega$), onde o sinal de entrada é proporcional a $K_{SWING\_UP} \cdot \omega$.

**Com base nisso, realize as seguintes etapas:**
* **Simulação:** Simule o sistema com e sem a estratégia de Swing-up ativada e registre os resultados.
* **Análise:** Discorra sobre como esse método é útil para o controle do sistema no mundo real.
* **Controle Híbrido:** Explique o motivo de a simulação utilizar duas zonas de controle distintas: a fase de balanço (*Swing-up*) e a fase de estabilização (*Catching*, ou "captura").
""")

def enunciado_questao7(type):
    if type == "Bola bastão":
        st.markdown("### Questão 7: Sistema em Malha Fechada")

        st.markdown("""##### Faça as simulações indicadas nas questões abaixo e cole os prints no relatório.\n
                    """)
        
        st.warning("Os prints devem conter os inputs da simulação juntamente com as informações pedidas abaixo.")


        st.error("A resposta da questão 2 se resume ao valor de C(s) ser negativo pois isso faz com que a saída tenha resposta com sinal oposto da entrada e isso vai ser incrementado já nas simulações e plotes. Ou seja, nesta e nas próximas questões, se você colocar um valor de K positivo será computado internamente como negativo.")
        
        st.markdown("""
                    a) Simule o sistema para um valor de K_feedback diferente de 0.00 e demais inputs à sua escolha.\n
                    a.1- Já que essa simulação não considera o atrito na bolinha, quanto tempo demoraria para o sistema se estabilizar?\n
                    b) Plote a resposta do sistema em malha fechada para o mesmo K_feedback escolhido na simulação. Além disso, plote também a resposta no tempo.\n
                    c) Plote o mapa de polos e zeros e o lugar das raízes.\n
                    c.1- De acordo com esses plots, o sistema é estável ou instável para o ganho escolhido?\n
                    c.2- Seria possível usar apenas o feedback e escolher, com a ajuda do lugar das raízes, um ganho adequado como estratégia de controle para estabilizar esse sistema?
                    Justifique.\n
                    d) Plote os diagramas de Bode e Nyquist e explique o que eles dizem sobre esse sistema em Malha Fechada.
                    """)
        
    if type == "Pêndulo simples invertido":
        st.markdown("### Questão 7: Controle em Malha Fechada e Estabilização")

        st.markdown("##### Faça as simulações indicadas nas questões abaixo e cole os prints no relatório.\n")
        
        st.warning("Os prints devem conter os inputs da simulação juntamente com as informações pedidas abaixo.")

        st.markdown("""
a) Simule o sistema para um valor de K_feedback diferente de 0.00.

a.1- Como fica a função de transferência em malha fechada do sistema?

b) Plote a resposta do sistema em malha fechada para o mesmo K_feedback escolhido na simulação. Plote também a resposta no tempo.

c) Plote o mapa de polos e zeros e o lugar das raízes.

c.1- Seria possível, para a fase de catch (quando o pêndulo está na zona de linearização), usar apenas o feedback e escolher, com a ajuda do lugar das raízes, um ganho adequado como estratégia de controle para estabilizar esse sistema? Justifique.

c.2- Seria possível encontrar o menor valor de ganho K_feedback para fazer esse sistema ser estável?
""")
                
        st.info(r"""A partir da equação caraterística do sistema em malha fechada:
$$1 + K_{fb} G(s) = 0$$ 

o que pode ser escrito como:

$$1 + \frac{K_{fb} \cdot K_{ma} \cdot (s + z_1)(s + z_2)}{(s + p_1)(s + p_2)} = 0$$

Desenvolvendo essa fórmula, se chega em:

$K_{fb} \cdot K_{ma} \cdot (s + z_1)(s + z_2)+(s + p_1)(s + p_2) = 0$

Assim, percebe-se que com K_feedback = 0, os polos do sistema em malha fechada são os mesmos do sistema em malha aberta, ou seja, o sistema é instável. 
Já para K_feedback > 0, o sistema em malha fechada tem os polos deslocados assim como mostra o plote do lugar das raízes. Sendo assim, se em algum momento os polos cruzam o eixo imaginário, há um ganho K_fb que torna o sistema estável.
""")

        st.markdown("""
d) Plote os diagramas de Bode e Nyquist e explique o que eles dizem sobre esse sistema em Malha Fechada.
""")
        
def enunciado_questao8(type):
    if type == "Bola bastão":
        st.markdown("### Questão 8: Controle com PID")

        st.markdown("""##### Faça as simulações indicadas nas questões abaixo e cole os prints no relatório. Coloque os inputs da simulaçao no mesmo print dos gráficos de desempenho.\n
                    """)

        st.markdown("""
                    a) Simule o sistema com o controlador PID com a bolinha em qualquer posição (que não no centro e com velocidade inicial zero) e ganhos arbitrários.\n
                    a.1- Testando diferentes valores para os ganhos, foi possível achar uma resposta satisfatória? Se sim, quais são esses valores dos ganhos?\n
                    b) Plote o gráfico Resposta em função de Kp.\n
                    b.1 - Qual o impacto da variação de Kp na resposta do sistema?\n
                    c) Plote o gráfico Resposta em função de Ki.\n
                    c.1 - Qual o impacto da variação de Ki na resposta do sistema?\n
                    d) Plote o gráfico Resposta em função de Kd.\n
                    d.1 - Qual o impacto da variação de Kd na resposta do sistema?\n
                    e) Plote a respota no tempo do sistema.\n
                    f) Selecione os melhores valores dos ganhos que você encontrou e plote o mapa de polos e zeros e o lugar das raízes.\n
                    f.1- De acordo com esses plots, o sistema é estável ou instável para os ganhos escolhidos?\n
                    g) Plote os diagramas de Bode e Nyquist e explique o que eles dizem sobre esse sistema controlado por PID.
                    """)
        
    if type == "Pêndulo simples invertido":
        st.markdown("### Questão 8: Controle com PID")

        st.markdown("""##### Faça as simulações indicadas nas questões abaixo e cole os prints no relatório.\n
                    """)

        st.markdown("""
                    a) Simule o sistema com o controlador PID.\n
                    a.1- Testando diferentes valores para os ganhos, foi possível achar uma resposta satisfatória? Se sim, quais são esses valores dos ganhos?\n
                    b) Plote o gráfico Resposta em função de Kp.\n
                    b.1 - Qual o impacto da variação de Kp na resposta do sistema?\n
                    c) Plote o gráfico Resposta em função de Ki.\n
                    c.1 - Qual o impacto da variação de Ki na resposta do sistema?\n
                    d) Plote o gráfico Resposta em função de Kd.\n
                    d.1 - Qual o impacto da variação de Kd na resposta do sistema?\n
                    e) Plote a resposta no tempo do sistema.\n
                    f) Dentre os melhores valores encontrados dos gráficos citados acima, seria possível aplicar esses valores em um sistema físico real? Justifique.\n
                    g) De acordo com esses plots dos polos e zeros e o lugar das raízes, o sistema é estável ou instável para os ganhos escolhidos? Justifique.\n
                    """)

def enunciado_questao9(type):
    if type == "Bola bastão":
        st.markdown("### Questão 9: Simulação no Kit Real - Malha Aberta")

    if type == "Pêndulo simples invertido":
        st.markdown("### Questão 9: Simulação no Kit Real - Malha Aberta")

def enunciado_questao10(type):
    if type == "Bola bastão":
        st.markdown("### Questão 10: Simulação no Kit Real - Malha Fechada")

    if type == "Pêndulo simples invertido":
        st.markdown("### Questão 10: Simulação no Kit Real - Malha Fechada")

def enunciado_questao11(type):
    if type == "Bola bastão":
        st.markdown("### Questão 11: Simulação no Kit Real - Controle PID")

    if type == "Pêndulo simples invertido":
        st.markdown("### Questão 11: Simulação no Kit Real - Controle PID")   
    
def plote_resposta_MA_Bola_Bastao(m, g, j, R, q):

    Hs = TransferFunction([-m*g], [(m + j/R**2), 0, 0])

    u = q * np.ones(1001) 
    
    t = np.linspace(0, 5, 1001)

    t_out, yout = ctl.forced_response(Hs, T=t, U=u)
    
    fig = go.Figure()

    # Adicionando a Resposta do Sistema
    fig.add_trace(go.Scatter(
        x=t_out, 
        y=yout, 
        mode='lines',
        name='Resposta do Sistema (Saída)',
        line=dict(color='blue', width=2),
        hovertemplate='Tempo: %{x:.2f}s<br>Posição: %{y:.4f}m'
    ))

    # Adicionando a Entrada Degrau (Referência)
    fig.add_trace(go.Scatter(
        x=t, 
        y=u, 
        mode='lines',
        name=f'Entrada Degrau (q={q})',
        line=dict(color='green', width=2, dash='dash'),
        hovertemplate='Referência: %{y:.2f}m'
    ))

    # 3. Customização do Layout
    fig.update_layout(
        title=dict(
            text=f"Resposta em Malha Aberta (Degrau = {q})",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Tempo (s)",
        yaxis_title="Posição (m)",
        hovermode="x unified",  # Mostra os dois valores ao mesmo tempo no mouse
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.5)"
        ),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=450
    )

    # 4. Exibição no Streamlit
    # Centralizando o gráfico usando colunas
    col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
    with col2:
        st.plotly_chart(fig, use_container_width=True)

def plote_resposta_MF_Bola_Bastao(m, g, j, R, K_feedback):

    # 1. Definição do Sistema e Malha Fechada
    t = np.linspace(0, 10, 1000)
    Hs = ctl.TransferFunction([m*g], [(m + j/R**2), 0, 0])

    Gs = ctl.feedback(K_feedback*Hs, 1)
    
    tout_MF, yout_MF = ctl.step_response(Gs, t)

    # Vetor de Referência (Degrau de valor K_feedback)
    y_referencia = K_feedback * np.ones_like(t)

    # 3. Criação do Gráfico Interativo com Plotly
    fig = go.Figure()

    # Adicionando a Referência (Valor K_feedback)
    fig.add_trace(go.Scatter(
        x=t, 
        y=y_referencia, 
        mode='lines',
        name=f'Referência (Ganho_feedback = {K_feedback})',
        line=dict(color='green', width=2, dash='dot'),
        hovertemplate='Ref: %{y:.4f}'
    ))

    # Adicionando a Resposta em Malha Fechada
    fig.add_trace(go.Scatter(
        x=tout_MF, 
        y=yout_MF, 
        mode='lines',
        name='Resposta do Sistema (MF)',
        line=dict(color='blue', width=2.5),
        hovertemplate='Tempo: %{x:.2f}s<br>Posição: %{y:.4f}m'
    ))

    # 4. Customização do Layout
    fig.update_layout(
        title=dict(
            text=f"Malha Fechada: Resposta ao Degrau de Amplitude {K_feedback}",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Tempo (s)",
        yaxis_title="Posição (m)",
        hovermode="x unified",
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.5)"
        ),
        template="plotly_white",
        height=500
    )

    # 5. Exibição no Streamlit
    col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
    with col2:
        st.plotly_chart(fig, use_container_width=True)

def plote_resposta_PID_Bola_Bastao(m, g, j, R, Kp, Ki, Kd):
    """
    Plota a resposta em malha fechada para diferentes ganhos Kd, com Kp fixo.
    """

    t = np.linspace(0, 10, 1000)
    s = ctl.TransferFunction.s  # Operador de Laplace
    
    Hs = TransferFunction([m * g], [(m + j / R**2), 0, 0])

    C = Kp + Kd*s + Ki/s

    # Criando a figura do Plotly
    fig = go.Figure()

    Gs = ctl.feedback(C*Hs, 1)
        
    t_out, y_out = ctl.step_response(Gs, t)

    fig.add_trace(go.Scatter(
        x=t, 
        y=np.ones(len(t)), 
        mode='lines',
        name='Referência (Degrau)',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=t_out, 
        y=y_out, 
        mode='lines',
        name=f'Kp={Kp}, Ki={Ki}, Kd={Kd}',
        line=dict(color="purple", width=2),
        hovertemplate='Tempo: %{x:.2f}s<br>Saída: %{y:.4f}'
    ))

    # 4. Customização do Layout
    fig.update_layout(
        title=dict(
            text=f"Resposta ao PID: Kp={Kp}, Ki={Ki}, Kd={Kd}",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Tempo (s)",
        yaxis_title="Posição (m)",
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.5)"
        ),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=500
    )

    # 5. Exibição no Streamlit
    st.plotly_chart(fig, use_container_width=True)

def plote_resposta_MA_Pendulo_simples_invertido(m, g, L, b, q_input):

    Hs = TransferFunction([3/(m*L**2)], [1, 3*b/(m*L**2), -3*g/(2*L)])

    u = q_input * np.ones(1001) 
    
    t = np.linspace(0, 5, 1001)

    t_out, yout = ctl.forced_response(Hs, T=t, U=u)
    
    fig = go.Figure()

    # Adicionando a Resposta do Sistema
    fig.add_trace(go.Scatter(
        x=t_out, 
        y=yout, 
        mode='lines',
        name='Resposta do Sistema (Saída)',
        line=dict(color='blue', width=2),
        hovertemplate='Tempo: %{x:.2f}s<br>Posição: %{y:.4f}m'
    ))

    # Adicionando a Entrada Degrau (Referência)
    fig.add_trace(go.Scatter(
        x=t, 
        y=u, 
        mode='lines',
        name=f'Entrada Degrau (q={q_input})',
        line=dict(color='green', width=2, dash='dash'),
        hovertemplate='Referência: %{y:.2f}m'
    ))

    # 3. Customização do Layout
    fig.update_layout(
        title=dict(
            text=f"Resposta em Malha Aberta (Degrau = {q_input})",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Tempo (s)",
        yaxis_title="θ (rad)",
        hovermode="x unified",  # Mostra os dois valores ao mesmo tempo no mouse
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.5)"
        ),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=450
    )

    # 4. Exibição no Streamlit
    # Centralizando o gráfico usando colunas
    col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
    with col2:
        st.plotly_chart(fig, use_container_width=True)

def plote_resposta_MF_Pendulo_simples_invertido(m, g, L, b, K_feedback):

    Hs = TransferFunction([3/(m*L**2)], [1, 3*b/(m*L**2), -3*g/(2*L)])

    Hs_MF = ctl.feedback(K_feedback * Hs, 1)

    u = K_feedback * np.ones(1001)
    
    t = np.linspace(0, 5, 1001)

    t_out, yout = ctl.forced_response(Hs_MF, T=t, U=u)
    
    fig = go.Figure()

    # Adicionando a Resposta do Sistema
    fig.add_trace(go.Scatter(
        x=t_out, 
        y=yout, 
        mode='lines',
        name='Resposta do Sistema (Saída)',
        line=dict(color='blue', width=2),
        hovertemplate='Tempo: %{x:.2f}s<br>Posição: %{y:.4f}m'
    ))

    # Adicionando a Entrada Degrau (Referência)
    fig.add_trace(go.Scatter(
        x=t, 
        y=u, 
        mode='lines',
        name=f'Entrada do Feedback (Ganho_feedback={K_feedback})',
        line=dict(color='green', width=2, dash='dash'),
        hovertemplate='Referência: %{y:.2f}m'
    ))

    # 3. Customização do Layout
    fig.update_layout(
        title=dict(
            text=f"Resposta em Malha Fechada (Ganho_feedback={K_feedback})",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Tempo (s)",
        yaxis_title="θ (rad)",
        hovermode="x unified",  # Mostra os dois valores ao mesmo tempo no mouse
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.5)"
        ),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=450
    )

    # 4. Exibição no Streamlit
    # Centralizando o gráfico usando colunas
    col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
    with col2:
        st.plotly_chart(fig, use_container_width=True)

def plote_resposta_PID_Pendulo_simples_invertido(m, g, L, b, Kp, Ki, Kd):
    Hs = TransferFunction([3/(m*L**2)], [1, 3*b/(m*L**2), -3*g/(2*L)])

    s = ctl.TransferFunction.s  # Operador de Laplace

    C = Kp + Kd*s + Ki/s
    
    t = np.linspace(0, 5, 1001)

    Gs = ctl.feedback(C*Hs, 1)

    t_out, y_out = ctl.step_response(Gs, t)
    
    fig = go.Figure()

    # Adicionando a Resposta do Sistema
    fig.add_trace(go.Scatter(
        x=t_out, 
        y=y_out, 
        mode='lines',
        name='Resposta do Sistema (Saída)',
        line=dict(color='blue', width=2),
        hovertemplate='Tempo: %{x:.2f}s<br>Posição: %{y:.4f}m'
    ))

    # Adicionando a Entrada Degrau (Referência)
    fig.add_trace(go.Scatter(
        x=t, 
        y=0.1*np.ones(len(t)), 
        mode='lines',
        name='Referência (Degrau)',
        line=dict(color='green', width=2, dash='dash'),
        hovertemplate='Referência: %{y:.2f}m'
    ))

    # 3. Customização do Layout
    fig.update_layout(
        title=dict(
            text=f"Resposta com controlador (Kp={Kp}, Ki={Ki}, Kd={Kd})",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Tempo (s)",
        yaxis_title="θ (rad)",
        hovermode="x unified",  # Mostra os dois valores ao mesmo tempo no mouse
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.5)"
        ),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=450
    )

    # 4. Exibição no Streamlit
    # Centralizando o gráfico usando colunas
    col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
    with col2:
        st.plotly_chart(fig, use_container_width=True)

def resposta_em_funcao_de_Kp(m, g, j, R):
    """
    Plota a resposta em malha fechada para diferentes ganhos Kp.
    """

    Kp_vals = [5, 1, 0.5]
    # 1. Definição da Planta (Hs) e Malha Fechada Base (Gs)
    # Hs = (m*g) / ((m + j/R^2)s^2)
    Hs = TransferFunction([m * g], [(m + j / R**2), 0, 0])

    # 2. Configuração do Tempo
    t = np.linspace(0, 5, 1000)
    
    # Criando a figura do Plotly
    fig = go.Figure()

    # 3. Adicionando a Referência (Degrau Unitário)
    fig.add_trace(go.Scatter(
        x=t, 
        y=np.ones(len(t)), 
        mode='lines',
        name='Referência (Degrau)',
        line=dict(color='gray', width=2, dash='dash')
    ))

    # 4. Loop para testar diferentes Kp_vals
    colors = ['green', 'blue', 'red', 'orange', 'purple']
    
    for Kp, col in zip(Kp_vals, colors):
        
        sys_cl = ctl.feedback(Kp * Hs, 1)
        
        # Segundo: Simula a resposta ao degrau no tempo 't'
        t_out, y_out = ctl.step_response(sys_cl, t)
        
        fig.add_trace(go.Scatter(
            x=t_out, 
            y=y_out, 
            mode='lines',
            name=f'Kp = {Kp}',
            line=dict(color=col, width=2),
            hovertemplate='Tempo: %{x:.2f}s<br>Saída: %{y:.4f}'
        ))

    # 5. Customização do Layout
    fig.update_layout(
        title=dict(
            text="Respostas em Malha Fechada em função de Kp",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Tempo (s)",
        yaxis_title="Amplitude",
        hovermode="x unified",
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.5)"
        ),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=500
    )

    # 6. Exibição no Streamlit
    st.plotly_chart(fig, use_container_width=True)

def resposta_pendulo_em_funcao_de_Kp(m, L, b, g):
    # 1. Correção do Numerador (dividindo pela massa)
    Hs = TransferFunction([3 / (m * L**2)], [1, 3*b/(m*L**2), -3*g/(2*L)])

    # 2. Configuração do Tempo (Note que no gráfico 2 o tempo vai até 50)
    t = np.linspace(0, 50, 1000)
    
    # Criando a figura do Plotly
    fig = go.Figure()

    # Adicionando a Referência (Degrau Unitário)
    fig.add_trace(go.Scatter(
        x=t, 
        y=np.ones(len(t)), 
        mode='lines',
        name='Referência (Degrau)',
        line=dict(color='gray', width=2, dash='dash')
    ))

    Kp_vals = [5, 2, 1]
    colors = ['green', 'blue', 'red', 'orange', 'purple']
    
    # 3. Conserto do Loop
    for Kp, col in zip(Kp_vals, colors):
        
        sys_cl = ctl.feedback(Kp * Hs, 1)
        
        # Segundo: Simula a resposta ao degrau no tempo 't'
        t_out, y_out = ctl.step_response(sys_cl, t)
        
        fig.add_trace(go.Scatter(
            x=t_out, 
            y=y_out, 
            mode='lines',
            name=f'Kp = {Kp}',
            line=dict(color=col, width=2),
            hovertemplate='Tempo: %{x:.2f}s<br>Saída: %{y:.4f}'
        ))
    # 5. Customização do Layout
    fig.update_layout(
        title=dict(
            text="Respostas em Malha Fechada em função de Kp",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Tempo (s)",
        yaxis_title="Amplitude",
        hovermode="x unified",
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.5)"
        ),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=500
    )

    # 6. Exibição no Streamlit
    st.plotly_chart(fig, use_container_width=True)


def resposta_em_funcao_de_Ki(m, g, j, R):
    """
    Plota a resposta em malha fechada para diferentes ganhos Ki, com Kp fixo.
    """
    # Configurações iniciais
    Ki_vals = [1.2, 1, 0.5, 0]
    Kp_fixo = 1
    t = np.linspace(0, 5, 1000)
    s = ctl.TransferFunction.s  # Operador de Laplace
    
    # 1. Definição da Planta (Hs) e Malha de Feedback Base (Gs)
    # Hs = (m*g) / ((m + j/R^2)s^2)
    Hs = TransferFunction([m * g], [(m + j / R**2), 0, 0])

    # Criando a figura do Plotly
    fig = go.Figure()

    # 2. Adicionando a Referência (Degrau Unitário)
    fig.add_trace(go.Scatter(
        x=t, 
        y=np.ones(len(t)), 
        mode='lines',
        name='Referência (Degrau)',
        line=dict(color='gray', width=2, dash='dash')
    ))

    # 3. Loop para testar diferentes Ki_vals
    colors = ['green', 'blue', 'red', 'yellow']
    
    for Ki, col in zip(Ki_vals, colors):
        # Controlador PI: C = Kp + Ki/s
        # Se Ki for 0, o controlador é apenas Proporcional
        if Ki == 0:
            C = Kp_fixo
        else:
            C = Kp_fixo + (Ki / s)

        sys_cl = ctl.feedback(C * Hs, 1)
        
        # Segundo: Simula a resposta ao degrau no tempo 't'
        t_out, y_out = ctl.step_response(sys_cl, t)
        
        fig.add_trace(go.Scatter(
            x=t_out, 
            y=y_out, 
            mode='lines',
            name=f'Ki = {Ki} (Kp=1)',
            line=dict(color=col, width=2),
            hovertemplate='Tempo: %{x:.2f}s<br>Saída: %{y:.4f}'
        ))

    # 4. Customização do Layout
    fig.update_layout(
        title=dict(
            text=f"Resposta em Malha Fechada: Efeito do Ganho Integral (Ki)",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Tempo (s)",
        yaxis_title="Amplitude",
        hovermode="x unified",
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.5)"
        ),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=500
    )

    # 5. Exibição no Streamlit
    st.plotly_chart(fig, use_container_width=True)

def resposta_pendulo_em_funcao_de_Ki(m, L, b, g):
    # Definindo a variável de Laplace s
    Hs = TransferFunction([3/(m*L**2)], [1, 3*b/(m*L**2), -3*g/(2*L)])

    Kp_fixo = 1
    Ki_vals = [5, 1, 0.05, 0]

    t = np.linspace(0, 10, 10000)
    s = ctl.TransferFunction.s  # Operador de Laplace

    # Criando a figura do Plotly
    fig = go.Figure()

    # 2. Adicionando a Referência (Degrau Unitário)
    fig.add_trace(go.Scatter(
        x=t, 
        y=np.ones(len(t)), 
        mode='lines',
        name='Referência (Degrau)',
        line=dict(color='gray', width=2, dash='dash')
    ))

    # 3. Loop para testar diferentes Ki_vals
    colors = ['green', 'blue', 'red', 'yellow']
    
    for Ki, col in zip(Ki_vals, colors):
        # Controlador PI: C = Kp + Ki/s
        # Se Ki for 0, o controlador é apenas Proporcional
        if Ki == 0:
            C = Kp_fixo
        else:
            C = Kp_fixo + (Ki / s)
        
        sys_cl = ctl.feedback(C * Hs, 1)
        t_out, y_out = ctl.step_response(sys_cl, t)

        fig.add_trace(go.Scatter(
            x=t_out, 
            y=y_out, 
            mode='lines',
            name=f'Ki = {Ki} (Kp=1)',
            line=dict(color=col, width=2),
            hovertemplate='Tempo: %{x:.2f}s<br>Saída: %{y:.4f}'
        ))

    # 4. Customização do Layout
    fig.update_layout(
        title=dict(
            text=f"Resposta em Malha Fechada: Efeito do Ganho Integral (Ki)",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Tempo (s)",
        yaxis_title="Amplitude",
        hovermode="x unified",
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.5)"
        ),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=500
    )

    # 5. Exibição no Streamlit
    st.plotly_chart(fig, use_container_width=True)

def resposta_em_funcao_de_Kd(m, g, j, R):
    """
    Plota a resposta em malha fechada para diferentes ganhos Kd, com Kp fixo.
    """
    # Configurações de ganhos e tempo
    Kd_vals = [0, 0.5, 1, 5]
    Kp_fixo = 3  # Conforme seu código original
    t = np.linspace(0, 5, 1000)
    s = ctl.TransferFunction.s  # Operador de Laplace
    
    # 1. Definição da Planta (Hs)
    # Hs = (m*g) / ((m + j/R^2)s^2)
    Hs = TransferFunction([m * g], [(m + j / R**2), 0, 0])

    # Criando a figura do Plotly
    fig = go.Figure()

    # 2. Adicionando a Referência (Degrau Unitário)
    fig.add_trace(go.Scatter(
        x=t, 
        y=np.ones(len(t)), 
        mode='lines',
        name='Referência (Degrau)',
        line=dict(color='gray', width=2, dash='dash')
    ))

    # 3. Loop para testar diferentes Kd_vals
    colors = ['green', 'blue', 'red', 'yellow']
    
    for Kd, col in zip(Kd_vals, colors):
        # Controlador PD: C = Kp + Kd*s
        C = Kp_fixo + Kd * s
        
        sys_cl = ctl.feedback(C * Hs, 1)
        t_out, y_out = ctl.step_response(sys_cl, t)
        
        fig.add_trace(go.Scatter(
            x=t_out, 
            y=y_out, 
            mode='lines',
            name=f'Kd = {Kd} (Kp=3)',
            line=dict(color=col, width=2),
            hovertemplate='Tempo: %{x:.2f}s<br>Saída: %{y:.4f}'
        ))

    # 4. Customização do Layout
    fig.update_layout(
        title=dict(
            text="Resposta em Malha Fechada: Efeito do Ganho Derivativo (Kd)",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Tempo (s)",
        yaxis_title="Posição (m)",
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.5)"
        ),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=500
    )

    # 5. Exibição no Streamlit
    st.plotly_chart(fig, use_container_width=True)

def resposta_pendulo_em_funcao_de_Kd(m, L, b, g):
    """
    Plota a resposta em malha fechada para diferentes ganhos Kd, com Kp fixo.
    """
    # Configurações de ganhos e tempo
    Kd_vals = [5, 1, 0.01, 0]
    Kp_fixo = 1  # Conforme seu código original
    t = np.linspace(0, 5, 1000)
    s = ctl.TransferFunction.s  # Operador de Laplace
    
    Hs = TransferFunction([3/(m*L**2)], [1, 3*b/(m*L**2), -3*g/(2*L)])
    
    # Criando a figura do Plotly
    fig = go.Figure()

    # 2. Adicionando a Referência (Degrau Unitário)
    fig.add_trace(go.Scatter(
        x=t, 
        y=np.ones(len(t)), 
        mode='lines',
        name='Referência (Degrau)',
        line=dict(color='gray', width=2, dash='dash')
    ))

    # 3. Loop para testar diferentes Kd_vals
    colors = ['green', 'blue', 'red', 'yellow']
    
    for Kd, col in zip(Kd_vals, colors):
        # Controlador PD: C = Kp + Kd*s
        C = Kp_fixo + Kd * s

        sys_cl = ctl.feedback(C * Hs, 1)
        t_out, y_out = ctl.step_response(sys_cl, t)
        
        fig.add_trace(go.Scatter(
            x=t_out, 
            y=y_out, 
            mode='lines',
            name=f'Kd = {Kd} (Kp=1)',
            line=dict(color=col, width=2),
            hovertemplate='Tempo: %{x:.2f}s<br>Saída: %{y:.4f}'
        ))

    # 4. Customização do Layout
    fig.update_layout(
        title=dict(
            text="Resposta em Malha Fechada: Efeito do Ganho Derivativo (Kd)",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="Tempo (s)",
        yaxis_title="Posição (m)",
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.5)"
        ),
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        height=500
    )

    # 5. Exibição no Streamlit
    st.plotly_chart(fig, use_container_width=True)

def plote_resposta_no_tempo(type, m=None, g=None, j=None, R=None, L=None, b=None, k_MA = None, k_feedback = None, Kp=None, Ki=None, Kd=None):

    if type == "Bola bastão MF":
        Hs = TransferFunction([m*g], [(m+j/R**2), 0, 0])
        Gs = ctl.feedback(k_feedback*Hs, 1)
        sys = Gs

    if type == "Bola bastão PID":
        s = ctl.TransferFunction.s
        Hs = TransferFunction([m*g], [(m+j/R**2), 0, 0])
        C = Kp + Kd*s + Ki/s
        sys = ctl.feedback(C * Hs, 1)

    if type == "Pêndulo simples invertido MF":
        Hs = TransferFunction([3/(m*L**2)], [1, 3*b/(m*L**2), -3*g/(2*L)])
        Gs = ctl.feedback(k_feedback*Hs, 1)
        sys = Gs

    if type == "Pêndulo simples invertido PID":
        s = ctl.TransferFunction.s
        Hs = TransferFunction([3/(m*L**2)], [1, 3*b/(m*L**2), -3*g/(2*L)])
        C = Kp + Kd*s + Ki/s
        sys = ctl.feedback(C * Hs, 1)

    wn, zeta, polos = ctl.damp(sys, doprint=False)
    wn_sistema = wn[0]
    zeta_sistema = zeta[0]
        
    # --- 3. Simulação Numérica no Tempo ---
    tempo_simulacao = 10
    t = np.linspace(0, tempo_simulacao, 1001)  
    u = np.ones(1001)  
    t_out, y_out = ctl.forced_response(sys, U=u, T=t)

    # Valor de Regime Permanente Real simulado
    y_inf = ctl.dcgain(sys)
    if np.isinf(y_inf) or np.isnan(y_inf):
        y_inf = y_out[-1]

    # --- CÁLCULO NUMÉRICO REAL DOS PARÂMETROS (Baseado na curva real) ---
    # 1. Valor de Pico e Tempo de Pico extraídos direto do vetor simulado
    idx_pico = np.argmax(y_out)
    y_pico_real = y_out[idx_pico]
    tp_real = t_out[idx_pico]

    # Calcular Sobressinal Real baseado no comportamento prático da curva
    if y_pico_real > y_inf and y_inf > 0:
        up_porcentagem_real = ((y_pico_real - y_inf) / y_inf) * 100
    else:
        up_porcentagem_real = 0.0

    # Condição real: Se passou da linha de regime por mais de 0.5%, o gráfico deve mostrar os indicadores de pico
    sistema_oscilou_na_pratica = (up_porcentagem_real > 0.5)

    # 2. Tempo de Acomodação Numérico (Critério de 2%)
    limite_superior = y_inf * 1.02
    limite_inferior = y_inf * 0.98
    fora_da_faixa = np.where((y_out > limite_superior) | (y_out < limite_inferior))[0]
    
    if len(fora_da_faixa) > 0 and fora_da_faixa[-1] < len(t_out) - 1:
        ts_real = t_out[fora_da_faixa[-1]]
    else:
        ts_real = t_out[0] 

    # 3. Tempo de Subida Numérico (Instante em que cruza a linha de regime pela 1ª vez)
    idx_subida = np.where(y_out >= y_inf)[0]
    if len(idx_subida) > 0:
        tr_real = t_out[idx_subida[0]]
    else:
        tr_real = 1.8 / wn_sistema # Fallback teórico caso seja lento demais e não cruze nos 10s

    # --- 4. Construção do Gráfico no Matplotlib ---
    fig = plt.figure(figsize=(11, 6)) 

    plt.plot(t_out, y_out, "k", linewidth=2, label="Resposta ao Degrau")
    plt.axhline(y_inf, color="b", linestyle="--", alpha=0.5, label=f"Regime Permanente ($y_\\infty$ = {y_inf:.2f})")
    plt.axhline(1.0, color="g", linestyle=":", alpha=0.5, label="Referência (Degrau Unitário)")

    # Exibição inteligente dos indicadores baseada no comportamento prático do gráfico
    if sistema_oscilou_na_pratica:
        plt.axvline(tp_real, color="r", linestyle="--", alpha=0.7, label=f"Tempo de Pico ($T_p$ = {tp_real:.2f} s)")
        plt.plot(tp_real, y_pico_real, "ro", label=f"Pico Máximo ($y_{{max}}$ = {y_pico_real:.2f} | %UP = {up_porcentagem_real:.1f}%)")
        
    # Tempo de acomodação e subida sempre existem e são plotados
    plt.axvline(ts_real, color="purple", linestyle="--", alpha=0.7, label=f"Tempo de Acomodação ($T_s$ = {ts_real:.2f} s)")
    plt.plot(ts_real, y_inf, "purple", marker="X", markersize=8, linestyle="None")

    plt.axvline(tr_real, color="orange", linestyle="--", alpha=0.7, label=f"Tempo de Subida ($T_r \\approx$ {tr_real:.2f} s)")
    plt.title(f"Análise Dinâmica para t={tempo_simulacao}: {type}\n($\\omega_n$ = {wn_sistema:.2f} rad/s | $\\zeta$ = {zeta_sistema:.3f})", fontsize=12)
    plt.xlabel("Tempo (s)")
    plt.ylabel("Resposta do Sistema y(t)")
    plt.xlim(0, max(t_out))

    plt.legend(loc="lower right", shadow=True, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # --- 5. Renderização no Streamlit ---
    st.pyplot(fig)

def plote_mapa_polos_zeros(type, m=None, g=None, j=None, R=None, L=None, b=None, k_MA = None, k_feedback = None, Kp=None, Ki=None, Kd=None):

    if type == "Bola bastão MA":
        Hs = ctl.TransferFunction([-m*g], [(m + j/R**2), 0, 0])
        sys = k_MA * Hs

    if type == "Bola bastão MF":
        Hs = TransferFunction([m*g], [(m+j/R**2), 0, 0])
        Gs = ctl.feedback(k_feedback*Hs, 1)
        sys = Gs #só para encaixar nos demais

    if type == "Bola bastão PID":
        s = ctl.TransferFunction.s
        Hs = TransferFunction([m*g], [(m+j/R**2), 0, 0])
        C = Kp + Kd*s + Ki/s
        sys = ctl.feedback(C * Hs, 1)

    if type == "Pêndulo simples invertido MA":
        Hs = TransferFunction([3/(m*L**2)], [1, 3*b/(m*L**2), -3*g/(2*L)])
        sys = k_MA * Hs

    if type == "Pêndulo simples invertido MF":
        Hs = TransferFunction([3/(m*L**2)], [1, 3*b/(m*L**2), -3*g/(2*L)])
        Gs = ctl.feedback(k_feedback*Hs, 1)
        sys = Gs

    if type == "Pêndulo simples invertido PID":
        s = ctl.TransferFunction.s
        Hs = TransferFunction([3/(m*L**2)], [1, 3*b/(m*L**2), -3*g/(2*L)])
        C = Kp + Kd*s + Ki/s
        sys = ctl.feedback(C * Hs, 1)

    polo = ctl.poles(sys)
    st.write('Polos =', polo)

    zero = ctl.zeros(sys)
    st.write('Zeros =', zero)
            
    fig, ax = plt.subplots(figsize=(8, 4))

    # Passamos o 'ax=ax' para que o gráfico seja desenhado na figura do Streamlit
    poles, zeros = ctl.pzmap(sys, plot=True, grid=True, ax=ax)

    ax.set_title(f'Mapa de Polos e Zeros - {type}')
    ax.set_xlabel('Eixo Real (σ)')
    ax.set_ylabel('Eixo Imaginário (jω)')

    legend_elements = [
        Line2D([0], [0], marker='x', color='blue', label='Polos', linestyle='None', markersize=8, markeredgewidth=2),
        Line2D([0], [0], marker='o', color='blue', label='Zeros', linestyle='None', markerfacecolor='none', markersize=8)
    ]
    ax.legend(handles=legend_elements, loc='best', shadow=True)

    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    col2.pyplot(fig)
    
    plt.close(fig)

def plote_lugar_raizes(type, m=None, g=None, j=None, R=None, L=None, b=None, k_MA = None, k_feedback = None,Kp=None, Ki=None, Kd=None):

    if type == "Bola bastão MA":
        Hs = ctl.TransferFunction([-m*g], [(m + j/R**2), 0, 0])
        sys = k_MA * Hs

    if type == "Bola bastão MF":
        Hs = TransferFunction([m*g], [(m+j/R**2), 0, 0])
        Gs = ctl.feedback(k_feedback*Hs, 1)
        sys = Gs #só para encaixar nos demais

    if type == "Bola bastão PID":
        s = ctl.TransferFunction.s
        Hs = TransferFunction([m*g], [(m+j/R**2), 0, 0])
        C = Kp + Kd*s + Ki/s
        sys = ctl.feedback(C * Hs, 1)

    if type == "Pêndulo simples invertido MA":
        Hs = TransferFunction([3/(m*L**2)], [1, 3*b/(m*L**2), -3*g/(2*L)])
        sys = k_MA * Hs

    if type == "Pêndulo simples invertido MF":
        Hs = TransferFunction([3/(m*L**2)], [1, 3*b/(m*L**2), -3*g/(2*L)])
        Gs = ctl.feedback(k_feedback*Hs, 1)
        sys = Gs

    if type == "Pêndulo simples invertido PID":
        s = ctl.TransferFunction.s
        Hs = TransferFunction([3/(m*L**2)], [1, 3*b/(m*L**2), -3*g/(2*L)])
        C = Kp + Kd*s + Ki/s
        sys = ctl.feedback(C * Hs, 1)

    fig, ax = plt.subplots(figsize=(8, 4))

    # Executamos o rlocus (o ';' oculta a lista de valores)
    ctl.rlocus(sys, grid=True, ax=ax);

    ax.set_title(f'Lugar das Raízes - {type}')
    ax.set_xlabel('Real (σ)')
    ax.set_ylabel('Imaginário (jω)')

    legend_elements = [
        Line2D([0], [0], color='blue', lw=1.5, label='Trajetória dos Polos'),
        Line2D([0], [0], marker='x', color='blue', label='Polos (K=0)',
            linestyle='None', markersize=8, markeredgewidth=2),
        # Caso sua função G tenha zeros, descomente a linha abaixo:
        Line2D([0], [0], marker='o', color='blue', label='Zeros', linestyle='None', markerfacecolor='none')
    ]

    ax.legend(handles=legend_elements, loc='upper left', frameon=True, shadow=True)

    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    col1, col2, col3 = st.columns([1, 2, 1])

    col2.pyplot(fig)
    
    plt.close(fig)

def plote_bode(type, m=None, g=None, j=None, R=None, L=None, b=None, k_MA = None, k_feedback = None, Kp=None, Ki=None, Kd=None):

    if type == "Bola bastão MA":
        Hs = ctl.TransferFunction([-m*g], [(m + j/R**2), 0, 0])
        sys = k_MA * Hs

    if type == "Bola bastão MF":
        Hs = TransferFunction([m*g], [(m+j/R**2), 0, 0])
        Gs = ctl.feedback(k_feedback*Hs, 1)
        sys = Gs #só para encaixar nos demais

    if type == "Bola bastão PID":
        s = ctl.TransferFunction.s
        Hs = TransferFunction([m*g], [(m+j/R**2), 0, 0])
        C = Kp + Kd*s + Ki/s
        sys = ctl.feedback(C * Hs, 1)

    if type == "Pêndulo simples invertido MA":
        Hs = TransferFunction([3/(m*L**2)], [1, 3*b/(m*L**2), -3*g/(2*L)])
        sys = k_MA * Hs

    if type == "Pêndulo simples invertido MF":
        Hs = TransferFunction([3/(m*L**2)], [1, 3*b/(m*L**2), -3*g/(2*L)])
        Gs = ctl.feedback(k_feedback*Hs, 1)
        sys = Gs

    if type == "Pêndulo simples invertido PID":
        s = ctl.TransferFunction.s
        Hs = TransferFunction([3/(m*L**2)], [1, 3*b/(m*L**2), -3*g/(2*L)])
        C = Kp + Kd*s + Ki/s
        sys = ctl.feedback(C * Hs, 1)

    # O segredo: Não criamos plt.subplots() manualmente antes.
    # O bode_plot cria a figura. Nós apenas a capturamos com plt.gcf() (get current figure).
    
    plt.figure(figsize=(8, 6)) # Define o tamanho da nova figura que será usada
    
    # Executamos o bode_plot
    # dB=True garante a escala em Decibéis, Hz=False mantém Rad/s
    ctl.bode_plot(sys, dB=True, Hz=False, grid=True)
    
    # Capturamos a figura que o control acabou de gerar
    fig = plt.gcf()
    
    # Adicionamos um título geral para a figura (opcional)
    fig.suptitle(f'Diagrama de Bode - {type}', fontsize=12)

    # Exibição no Streamlit
    col1, col2, col3 = st.columns([0.6, 2, 0.6])
    with col2:
        st.pyplot(fig)
    
    # Importante limpar para não acumular na memória ou bugar o próximo gráfico
    plt.close(fig)

def plote_nyquist(type, m=None, g=None, j=None, R=None, L=None, b=None, k_MA = None, k_feedback = None, Kp=None, Ki=None, Kd=None):

    if type == "Bola bastão MA":
        Hs = ctl.TransferFunction([-m*g], [(m + j/R**2), 0, 0])
        sys = k_MA * Hs

    if type == "Bola bastão MF":
        Hs = TransferFunction([m*g], [(m+j/R**2), 0, 0])
        Gs = ctl.feedback(k_feedback*Hs, 1)
        sys = Gs #só para encaixar nos demais

    if type == "Bola bastão PID":
        s = ctl.TransferFunction.s
        Hs = TransferFunction([m*g], [(m+j/R**2), 0, 0])
        C = Kp + Kd*s + Ki/s
        sys = ctl.feedback(C * Hs, 1)

    if type == "Pêndulo simples invertido MA":
        Hs = TransferFunction([3/(m*L**2)], [1, 3*b/(m*L**2), -3*g/(2*L)])
        sys = k_MA * Hs

    if type == "Pêndulo simples invertido MF":
        Hs = TransferFunction([3/(m*L**2)], [1, 3*b/(m*L**2), -3*g/(2*L)])
        Gs = ctl.feedback(k_feedback*Hs, 1)
        sys = Gs

    if type == "Pêndulo simples invertido PID":
        s = ctl.TransferFunction.s
        Hs = TransferFunction([3/(m*L**2)], [1, 3*b/(m*L**2), -3*g/(2*L)])
        C = Kp + Kd*s + Ki/s
        sys = ctl.feedback(C * Hs, 1)

    fig, ax = plt.subplots(figsize=(8, 4))

    # Executamos o rlocus (o ';' oculta a lista de valores)
    ctl.nyquist_plot(sys, title=f'Nyquist for {sys}');

    col1, col2, col3 = st.columns([0.5, 4, 0.5])

    col2.pyplot(fig)
    
    plt.close(fig)

def get_ball_start_pos(L_beam, type, min_dist_pct=0.10):
                    if type == 'Aleatório':
                        min_dist = L_beam * min_dist_pct
                        side = np.random.choice([-1, 1])
                        # Garante que a bola não comece muito perto do centro ou das bordas extremas
                        pos = side * np.random.uniform(min_dist, L_beam / 2 - min_dist)

                    elif type == "Canto esquerdo":
                        side = -1
                        pos = 0.95 * side * L_beam/2

                    elif type == "Canto direito":
                        side = 1
                        pos = 0.95 * side * L_beam/2

                    elif type == "Centro":
                        pos = 0

                    return pos

def render_bola_bastao_frame(fig, ax, ball_x, beam_theta, L, passo_atual):
    """
    Desenha o estado atual do sistema Ball & Beam e retorna o frame como um array numpy.
    """
    ax.clear()
    
    # Desenhar o bastão (beam)
    beam_x_coords = [-(L/2)*np.cos(beam_theta), (L/2)*np.cos(beam_theta)]
    beam_y_coords = [-(L/2)*np.sin(beam_theta), (L/2)*np.sin(beam_theta)]
    ax.plot(beam_x_coords, beam_y_coords, 'black', linewidth=4)
    
    # Desenhar a bola
    ball_plot_x = ball_x * np.cos(beam_theta)
    ball_plot_y = ball_x * np.sin(beam_theta)
    ax.plot(ball_plot_x, ball_plot_y, 'ro', markersize=12)

    # Configurações do gráfico
    ax.set_xlim(-L/2 - 0.1, L/2 + 0.1)
    ax.set_ylim(-0.4, 0.4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"Passo: {passo_atual:03d} | Pos: {ball_x:+.2f}m")

    # Captura do frame
    fig.canvas.draw()
    frame = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    
    return frame

def plot_resultado_simulacao_bola_bastao(dt, L, max_ang_alpha, ball_pos_history, erro_history, 
                            alpha_history, action_history, ball_velocity_history, 
                            ball_aceleration_history):
    """
    Gera e plota os gráficos interativos da simulação usando Plotly.
    """
    # Eixo do tempo
    t = np.linspace(0, len(ball_pos_history) - 1, len(ball_pos_history)) * dt

    # --- Figura 1: Angulação, Ação e Erro ---
    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(x=t, y=erro_history, mode='lines', line=dict(color='red'), name='Erro em (m)'))
    fig1.add_trace(go.Scatter(x=t, y=alpha_history, mode='lines', line=dict(color='green'), name='Alpha - Ângulo Bastão (rad)'))
    fig1.add_trace(go.Scatter(x=t, y=action_history, mode='lines', line=dict(color='cyan', dash='dot'), name='Ação (rad)'))

    fig1.add_hline(y=max_ang_alpha, line_dash="dash", line_color="orange", annotation_text="Limite máximo (rad)", annotation_position="top right")
    fig1.add_hline(y=-max_ang_alpha, line_dash="dash", line_color="orange", annotation_text="Limite mínimo (rad)", annotation_position="bottom right")

    fig1.update_layout(
        title='Angulação do bastão, Ação e Erro do sistema',
        xaxis_title='Tempo',
        yaxis_title='Amplitude',
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40)
    )

    st.plotly_chart(fig1, use_container_width=True)

    # --- Figura 2: Posição, Velocidade e Aceleração ---
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(x=t, y=ball_pos_history, mode='lines', line=dict(color='blue'), name='Posição da Bola (m)'))
    fig2.add_trace(go.Scatter(x=t, y=ball_velocity_history, mode='lines', line=dict(color='green'), name='Velocidade da Bola (m/s)'))
    fig2.add_trace(go.Scatter(x=t, y=ball_aceleration_history, mode='lines', line=dict(color='#D4D000'), name='Aceleração da Bola (m/s²)'))

    fig2.add_hline(y=L/2, line_dash="dash", line_color="red", annotation_text="Limite máximo do bastão (m)", annotation_position="top right")
    fig2.add_hline(y=-L/2, line_dash="dash", line_color="red", annotation_text="Limite mínimo do bastão (m)", annotation_position="bottom right")

    fig2.update_layout(
        title="Posição, Velocidade e Aceleração da bola",
        xaxis_title='Tempo',
        yaxis_title='Amplitude',
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40)
    )

    st.plotly_chart(fig2, use_container_width=True)

def plot_resultado_simulacao_pendulo(dt, lim_motor, erro_history, external_action_history, theta_history, theta_dot_history, 
                                     theta_double_dot_history, control_type_history):
    """
    Gera e plota os gráficos interativos da simulação do pêndulo invertido usando Plotly.
    """
    # Eixo do tempo (garantindo que tenha o mesmo tamanho das listas)
    t = np.linspace(0, len(erro_history) - 1, len(erro_history)) * dt

    # --- Figura 1: Erro Angulatório ---
    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(x=t, y=erro_history, mode='lines', line=dict(color='red'), name='Erro do Ângulo (rad)'))

    fig1.update_layout(
        title='Erro de Angulação',
        xaxis_title='Tempo (s)',
        yaxis_title='θ (rad)',
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40)
    )

    st.plotly_chart(fig1, use_container_width=True)

    # --- Figura 2: Ação do motor ---
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=t, 
        y=[val * lim_motor for val in control_type_history], # Normalização aqui
        mode='lines', 
        line=dict(color='green', dash='solid', shape='hv'), 
        name='False:Swing-up\nTrue:Degrau'
    ))
    fig2.add_trace(go.Scatter(x=t, y=external_action_history, mode='lines', line=dict(color='cyan', dash='dot'), name='Ação / Torque (Nm)'))
    # Linhas de limite do motor
    fig2.add_hline(y=lim_motor, line_dash="dash", line_color="orange", annotation_text="Limite Máx Motor", annotation_position="top right")
    fig2.add_hline(y=-lim_motor, line_dash="dash", line_color="orange", annotation_text="Limite Mín Motor", annotation_position="bottom right")

    fig2.update_layout(
        title='Tipo de ação e Esforço de Controle (Torque)',
        xaxis_title='Tempo (s)',
        yaxis_title='Amplitude',
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40)
    )

    st.plotly_chart(fig2, use_container_width=True)

    # --- Figura 3: Posição e Velocidade Angular  ---
    fig3 = go.Figure()

    fig3.add_trace(go.Scatter(x=t, y=theta_history, mode='lines', line=dict(color="#08DF90"), name='Posição Angular (rad)'))
    fig3.add_trace(go.Scatter(x=t, y=theta_dot_history, mode='lines', line=dict(color="#A103C8"), name='Velocidade Angular (rad/s)'))

    fig3.update_layout(
        title="Posição e Velocidade Angular",
        xaxis_title='Tempo (s)',
        yaxis_title='Amplitude',
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40)
    )

    st.plotly_chart(fig3, use_container_width=True)

    # --- Figura 4: Aceleração Angular  ---
    fig4 = go.Figure()

    fig4.add_trace(go.Scatter(x=t, y=theta_double_dot_history, mode='lines', line=dict(color='#D4D000'), name='Aceleração Angular (rad/s²)')) 

    fig4.update_layout(
        title="Aceleração Angular",
        xaxis_title='Tempo (s)',
        yaxis_title='Amplitude',
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40)
    )

    st.plotly_chart(fig4, use_container_width=True)
