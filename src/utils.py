import streamlit as st
import plotly.graph_objects as go
import numpy as np
import control as ctl
from control import (TransferFunction)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def enunciado_questao2():
    st.markdown("### Questão 2: Modelagem e Função de Transferência")

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

def enunciado_questao3():
    st.markdown("### Questão 3: Linearização de Sistemas")
    
    st.markdown("""
    Sistemas de controle não lineares lidam com dinâmicas que não seguem o princípio da superposição, 
    onde a resposta não é proporcional à entrada. Apesar desses sistemas serem muito comuns e 
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
        \na) A melhor forma de se linearizar o seno é utilizar a função original $F(\text{original}) = \sin(\\alpha)$, pois ela representa com exatidão o comportamento do sistema ao longo do tempo.
        \nb) A melhor forma de se linearizar o seno é utilizar o polinômio de primeira ordem $F_1(x) = x$, pois ela respeita o princípio da superposição. Porém, ela se aproxima da função original só para ângulos próximos de zero.
        \nc) A melhor forma de se linearizar o seno é utilizar o polinômio de terceira ordem $F_3(x) = x - \\frac{x^3}{6}$ pois, por mais que ele não respeite o princípio da superposição, sua resposta se aproxima melhor do original.
        \nd) A melhor forma de se linearizar o seno é utilizar o polinômio de sétima ordem $F_7(x) = x - \\frac{x^3}{6} + \\frac{x^5}{120} - \\frac{x^7}{5040}$ pois ele é o que melhor representa o sinal original sem criar distorção.
        \ne) A melhor forma de se linearizar o seno é utilizar o polinômio de maior ordem que o computador permite calcular, pois quanto maior a ordem, mais próximo da resposta do seno o polinômio estará.
    """)

def enunciado_questao4(L, d):
    st.markdown("### Questão 4: Relação Cinemática e Limites de Operação")

    st.markdown("""
    O comportamento do sistema depende da angulação $\\alpha$ (inclinação do bastão), 
    mas geralmente temos controle direto apenas sobre o ângulo $\\theta$ (braço do motor).
    
    A relação entre esses dois ângulos é dada pela geometria do mecanismo:
    """)

    st.latex(r"sen(\alpha) \cdot L = sen(\theta) \cdot d")

    st.info("Utilize a linearização encontrada na questão anterior ($sen(\\alpha)$ é aproximadamente Fx, onde Fx é o polinômio de Taylor que melhor lineariza a função).")

    st.markdown(f"""
    **Pergunta:** \n
    Encontre quais são os maiores valores de $\\alpha$ e $\\theta$ em radianos para um sistema com:
    * $L = {L}$ metros (Comprimento do bastão).
    * $d = {d}$ metros (Offset do braço da alavanca).
    
    A resposta encontrada faz sentido com relação à condição de linearização?
    """)

def enunciado_questao5():
    st.markdown("### Questão 5: Sistema em Malha Aberta")

    st.markdown("""##### Faça as simulações indicadas nas questões abaixo e cole os prints no relatório. Coloque os inputs da simulaçao no mesmo print dos gráficos de desempenho.\n
                """)

    st.markdown("""
                a) Simule o sistema para um valor de entrada degrau diferente de 0.00 e demais inputs a sua escolha.\n
                b) Plote a resposta ao degrau para o mesmo degrau escolhido na simulação.\n
                c) Plote o mapa de polos e zeros e o lugar das raízes e justifique o porquê do sistema ser instável para qualquer degrau colocado na entrada.\n
                d) Plote os diagramas de Bode e Nyquist e explique o que eles dizem sobre esse sistema em Malha Aberta.
                """)
    
def enunciado_questao6():
    st.markdown("### Questão 6: Sistema em Malha Fechada")

    st.markdown("""##### Faça as simulações indicadas nas questões abaixo e cole os prints no relatório. Coloque os inputs da simulaçao no mesmo print dos gráficos de desempenho.\n
                """)

    st.markdown("""
                a) Simule o sistema para um valor de K_feedback diferente de 0.00 e demais inputs a sua escolha.\n
                a.1- Já que essa simulação não considera o atrito na bolinha, quanto tempo demoraria para o sistema se estabilizar?\n
                b) Plote a resposta do sistema em malha fechada para o mesmo K_feedback escolhido na simulação.\n
                c) Plote o mapa de polos e zeros e o lugar das raízes.\n
                c.1- De acordo com esses plotes, o sistema é estável ou instável para o ganho escolhido?\n
                c.2- Seria possível usar apenas o feedback e escolher, com a ajuda do lugar das raízes, um ganho adequado como estratégia de controle para estabilizar esse sistema?
                Justifique.\n
                d) Plote os diagramas de Bode e Nyquist e explique o que eles dizem sobre esse sistema em Malha Fechada.
                """)
    
def enunciado_questao7():
    st.markdown("### Questão 7: Controle com PID")

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
                e) Selecione os melhores valores dos ganhos que você encontrou e plote o mapa de polos e zeros e o lugar das raízes.\n
                c.1- De acordo com esses plotes, o sistema é estável ou instável para os ganhos escolhidos?\n
                f) Plote os diagramas de Bode e Nyquist e explique o que eles dizem sobre esse sistema controlado por PID.
                """)

def enunciado_questao8():
    st.markdown("### Questão 8: Simulação no Kit Real - Malha Aberta")

def enunciado_questao9():
    st.markdown("### Questão 9: Simulação no Kit Real - Malha Fechada")

def enunciado_questao10():
    st.markdown("### Questão 10: Simulação no Kit Real - Controle PID")
    
def plote_resposta_MA_Bola_Bastao(m, g, j, R, q):

    Hs = TransferFunction([m*g], [(m + j/R**2), 0, 0])

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

    st.warning("⚠️ Simulação usa K negativo e o referencial teórico é K positivo.")

    # 1. Definição do Sistema e Malha Fechada
    t = np.linspace(0, 10, 1000)
    Hs = ctl.TransferFunction([m*g], [(m + j/R**2), 0, 0])
    
    # Gs = (K * Hs) / (1 + K * Hs)
    Gs = ctl.feedback(K_feedback * Hs, 1)

    # 2. Simulação da Resposta ao Degrau com magnitude K_feedback
    # Multiplicamos a resposta por K_feedback para que o degrau não seja unitário
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
        name=f'Referência (Degrau = {K_feedback})',
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
    

    Gs = ctl.feedback(Hs, 1)

    C = Kp + Kd*s + Ki/s

    sys = ctl.feedback(C * Gs, 1)

    # Criando a figura do Plotly
    fig = go.Figure()
        
    t_out, y_out = ctl.step_response(sys, t)

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
        name=f'Kd = {Kd} (Kp=3)',
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

def resposta_em_funcao_de_Kp(m, g, j, R):
    """
    Plota a resposta em malha fechada para diferentes ganhos Kp.
    """

    Kp_vals = [5, 1, 0.5]
    # 1. Definição da Planta (Hs) e Malha Fechada Base (Gs)
    # Hs = (m*g) / ((m + j/R^2)s^2)
    Hs = TransferFunction([m * g], [(m + j / R**2), 0, 0])
    
    # K_feedback unitário conforme seu código original
    K_feedback = 1
    Gs = ctl.feedback(K_feedback * Hs, 1)

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
        # Controlador Proporcional C = Kp
        # Aplicado sobre a malha Gs conforme seu código original
        sys_cl = ctl.feedback(Kp * Gs, 1)
        
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
    
    # K_feedback unitário conforme lógica anterior
    K_feedback = 1
    Gs = ctl.feedback(K_feedback * Hs, 1)

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
        
        # Sistema em Malha Fechada: Feedback(C * Gs, 1)
        sys_cl = ctl.feedback(C * Gs, 1)
        
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
    
    # Gs base conforme sua lógica (Feedback unitário da planta pura)
    Gs = ctl.feedback(Hs, 1)

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
        
        # Sistema em Malha Fechada: Feedback(C * Gs, 1)
        # Nota: Se Gs já for malha fechada no seu código original, 
        # C * Gs aplica o controle sobre esse sistema realimentado.
        sys_cl = ctl.feedback(C * Gs, 1)
        
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

def plote_mapa_polos_zeros(m, g, j, R, type, Kp=None, Ki=None, Kd=None):

    if type == "Bola bastão MA":
        Hs = ctl.TransferFunction([m*g], [(m + j/R**2), 0, 0])
        sys = Hs

    if type == "Bola bastão MF":
        Hs = TransferFunction([m*g], [(m+j/R**2), 0, 0])
        Gs = ctl.feedback(1*Hs, 1)
        sys = Gs #só para encaixar nos demais

    if type == "Bola bastão PID":
        s = ctl.TransferFunction.s
        Hs = TransferFunction([m*g], [(m+j/R**2), 0, 0])
        Gs = ctl.feedback(1*Hs, 1)
        C = Kp + Kd*s + Ki/s
        sys = ctl.feedback(C * Gs, 1)


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

def plote_lugar_raizes(m, g, j, R, type, Kp=None, Ki=None, Kd=None):

    if type == "Bola bastão MA":
        Hs = ctl.TransferFunction([m*g], [(m + j/R**2), 0, 0])
        sys = Hs

    if type == "Bola bastão MF":
        Hs = TransferFunction([m*g], [(m+j/R**2), 0, 0])
        Gs = ctl.feedback(1*Hs, 1)
        sys = Gs #só para encaixar nos demais

    if type == "Bola bastão PID":
        s = ctl.TransferFunction.s
        Hs = TransferFunction([m*g], [(m+j/R**2), 0, 0])
        Gs = ctl.feedback(1*Hs, 1)
        C = Kp + Kd*s + Ki/s
        sys = ctl.feedback(C * Gs, 1)

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

def plote_bode(m, g, j, R, type, Kp=None, Ki=None, Kd=None):

    if type == "Bola bastão MA":
        Hs = ctl.TransferFunction([m*g], [(m + j/R**2), 0, 0])
        sys = Hs

    if type == "Bola bastão MF":
        Hs = TransferFunction([m*g], [(m+j/R**2), 0, 0])
        Gs = ctl.feedback(1*Hs, 1)
        sys = Gs #só para encaixar nos demais

    if type == "Bola bastão PID":
        s = ctl.TransferFunction.s
        Hs = TransferFunction([m*g], [(m+j/R**2), 0, 0])
        Gs = ctl.feedback(1*Hs, 1)
        C = Kp + Kd*s + Ki/s
        sys = ctl.feedback(C * Gs, 1)

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

def plote_nyquist(m, g, j, R, type, Kp=None, Ki=None, Kd=None):

    if type == "Bola bastão MA":
        Hs = ctl.TransferFunction([m*g], [(m + j/R**2), 0, 0])
        sys = Hs

    if type == "Bola bastão MF":
        Hs = TransferFunction([m*g], [(m+j/R**2), 0, 0])
        Gs = ctl.feedback(1*Hs, 1)
        sys = Gs #só para encaixar nos demais

    if type == "Bola bastão PID":
        s = ctl.TransferFunction.s
        Hs = TransferFunction([m*g], [(m+j/R**2), 0, 0])
        Gs = ctl.feedback(1*Hs, 1)
        C = Kp + Kd*s + Ki/s
        sys = ctl.feedback(C * Gs, 1)

    fig, ax = plt.subplots(figsize=(8, 4))

    # Executamos o rlocus (o ';' oculta a lista de valores)
    ctl.nyquist_plot(sys, title=f'Nyquist for {sys}');

    col1, col2, col3 = st.columns([0.5, 4, 0.5])

    col2.pyplot(fig)
    
    plt.close(fig)

def baixar_relatorio_bola_bastao():
    st.write("Arrumar lógica")

def baixar_relatorio_pendulo_simples():
    st.write("Arrumar lógica")

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