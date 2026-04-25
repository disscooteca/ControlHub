import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os
import imageio
from ballbeam_gym.envs.balance import BallBeamBalanceEnv
from IPython.display import HTML
import gymnasium as gym
from ballbeam_gym.envs.balance import BallBeamBalanceEnv
from src.utils import baixar_relatorio_pendulo_simples, get_ball_start_pos, render_bola_bastao_frame, plot_resultado_simulacao_bola_bastao, plot_resultado_simulacao_pendulo
from src.utils import enunciado_questao2, enunciado_questao3, enunciado_questao4, enunciado_questao5, enunciado_questao6, enunciado_questao7, enunciado_questao8, enunciado_questao9, enunciado_questao10
from src.utils import plote_resposta_MA_Bola_Bastao, plote_resposta_MF_Bola_Bastao, plote_resposta_PID_Bola_Bastao, plote_resposta_MA_Pendulo_simples_invertido, plote_resposta_MF_Pendulo_simples_invertido, plote_resposta_PID_Pendulo_simples_invertido 
from src.utils import plote_mapa_polos_zeros, plote_lugar_raizes, plote_bode, plote_nyquist
from src.utils import resposta_em_funcao_de_Kp, resposta_em_funcao_de_Ki, resposta_em_funcao_de_Kd
import plotly.graph_objects as go
import requests

ESP32_IP = "10.78.73.125"

def toggle_led():
    try:
        response = requests.get(f"http://{ESP32_IP}/toggle" )
        if response.status_code == 200:
            st.success(f"Comando enviado: {response.text}")
        else:
            st.error(f"Erro ao enviar comando: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Não foi possível conectar ao ESP32. Verifique o IP e a conexão Wi-Fi.")

st.set_page_config(
    page_title="Controlpy",
    page_icon="⚙️",
    layout="wide" 
)

sistema = st.sidebar.selectbox(
    "Escolha qual sistema deseja simular:",
    ("Bola bastão", "Pêndulo simples invertido", "Pêndulo furuta")
)

if sistema == "Bola bastão":
    
    st.title("Simulação do Sistema Bola e Bastão", text_alignment= 'center')

    with st.expander("Representação do sistema Bola e Bastão"):
        st.image("sistema-ball-and-bean.png")

    st.sidebar.write("---")
    with open("Roteiro Bola bastão.docx", "rb") as file:
        btn = st.sidebar.download_button(
            label="Baixar Roteiro",
            icon="🚨",
            data=file,
            file_name="Roteiro Bola bastão.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    st.sidebar.write("---")

    # --- Parâmetros do Sistema (fixos para esta simulação) ---
    m = 0.01  # Massa da bola em KG
    R = 0.02  # Raio da bola em metros
    d = 0.05  # Deslocamento inicial do braço (não usado diretamente na simulação, mas para max_ang_alpha)
    g = 9.8   # Aceleração da gravidade m/s^2
    L = 0.5   # Tamanho do bastão em metros
    j = 2 * m * R * R / 5  # Momento de inércia da bola kg.m^2

    max_ang_alpha = d / L  # Máximo ângulo do bastão (derivado de d/L)
    max_ang_theta = np.pi / 2 # Não usado diretamente na simulação do ballbeam-gym

    dt = 0.05
    tempo = 100
    amostras = int(tempo / dt)
    #num_iteracoes = amostras # Usar o número total de amostras como iterações
    num_iteracoes = 150

    parte_simulacao = st.sidebar.selectbox(
        "Selecione a questão da simulação",
        ("Questão 1", "Questão 2", "Questão 3", "Questão 4", "Questão 5", "Questão 6", "Questão 7", "Questão 8", "Questão 9", "Questão 10")
    )

    st.sidebar.write("---")

    if parte_simulacao == "Questão 1":
        st.warning("Conexão com a esp32")
        if st.button("Liga/Desliga Led"):
            toggle_led()
        
    if parte_simulacao == "Questão 2":

        enunciado_questao2(type="Bola bastão")

    if parte_simulacao == "Questão 3":

        enunciado_questao3(type="Bola bastão")

    if parte_simulacao == "Questão 4":

        enunciado_questao4(L=L, d=d, type="Bola bastão")

    if parte_simulacao == "Questão 5":

        with st.expander("Enunciado Questão 5"):
            enunciado_questao5(type="Bola bastão")

        # --- Inputs do Usuário ---
        st.sidebar.header("Inputs da Simulação")
        q_input = st.sidebar.slider("Valor da entrada degrau", min_value=-max_ang_alpha, max_value=max_ang_alpha, value=0.0, step=0.01, help=f"Valor de entrada é a angulação do bastão em radianos indo de -{max_ang_alpha} até {max_ang_alpha} ")
        init_velocity_input = st.sidebar.slider("Velocidade inicial da Bola", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)
        opcoes_escolha_posicao = ["Aleatório", "Canto esquerdo", "Canto direito", "Centro"]
        escolha_posicao = st.sidebar.radio("Posição inicial da bola no bastão", 
                                                  opcoes_escolha_posicao, help="Aleatório: A bola pode começar em qualquer ponto do bastão (mas não no centro e nem perto dele)." \
                                                  "\nCanto Esquerdo: A bola começa obrigatoriamente na extremidade esquerda.\nCanto Direito: A bola começa obrigatoriamente na extremidade direita.")

        if st.sidebar.button("Simular"):
            st.subheader("Simulação em Malha Aberta com entrada degrau")

            # Inicializar o ambiente
            env = BallBeamBalanceEnv(
                timestep=dt,
                beam_length=L,
                max_angle=max_ang_alpha,
                init_velocity=init_velocity_input,
                max_timesteps=500,
                action_mode='continuous'
            )

            # Configurações manuais
            env.bb.g = g
            env.bb.L = L

            # Resetar o ambiente
            state = env.reset()
            if isinstance(state, tuple):
                obs = state[0]
            else:
                obs = state

            # --- MODIFICAÇÃO: POSIÇÃO INICIAL ALEATÓRIA ---
            if escolha_posicao == "Aleatório":
                random_pos_val = get_ball_start_pos(L, type=escolha_posicao)
                env.bb.x = random_pos_val
                # Sincronizar a observação com a nova posição
                obs = list(obs)
                obs[1] = random_pos_val
                obs = np.array(obs)
                st.info(f"Bola iniciando na posição aleatória: {random_pos_val:.3f}m")

            elif escolha_posicao == "Canto esquerdo":
                corner_pos_val = get_ball_start_pos(L, type=escolha_posicao)
                env.bb.x = corner_pos_val
                # Sincronizar a observação com a nova posição
                obs = list(obs)
                obs[1] = corner_pos_val
                obs = np.array(obs)
                st.info(f"Bola iniciando na extremidade esquerda: {env.bb.x:.3f}m")

            elif escolha_posicao == "Canto direito":
                corner_pos_val = get_ball_start_pos(L, type=escolha_posicao)
                env.bb.x = corner_pos_val
                # Sincronizar a observação com a nova posição
                obs = list(obs)
                obs[1] = corner_pos_val
                obs = np.array(obs)
                st.info(f"Bola iniciando na extremidade direita: {env.bb.x:.3f}m")

            elif escolha_posicao == "Centro":
                posicao_central = get_ball_start_pos(L, type=escolha_posicao)
                env.bb.x = posicao_central
                # Sincronizar a observação com a nova posição
                obs = list(obs)
                obs[1] = posicao_central
                obs = np.array(obs)
                st.info(f"Bola iniciando no meio: {env.bb.x:.3f}m")

            frames = []
            alpha_history = []
            ball_pos_history = []
            ball_velocity_history = []
            ball_aceleration_history = []
            action_history = []
            erro_history = []

            st.write("Iniciando simulação...")

            # Criar figura para renderização manual
            fig_render, ax_render = plt.subplots(figsize=(6, 3))

            for i in range(num_iteracoes):
                # Pegar dados atuais do motor de física interno
                ball_x = env.bb.x
                beam_theta = env.bb.theta

                frame = render_bola_bastao_frame(fig_render, ax_render, ball_x, beam_theta, L, i)
                frames.append(frame)
                
                # Ação é a entrada 'q' fornecida pelo usuário
                action = q_input
                action_history.append(action)

                # Erro é a diferença entre a posição desejada (0) e a posição atual da bola
                erro_history.append(0 - ball_x)

                alpha_history.append(obs[0]) # Ângulo do bastão
                ball_pos_history.append(obs[1]) # Posição da bola
                ball_velocity_history.append(obs[2]) # Velocidade da bola

                aceleracao = - (m * g / (m + j / (R**2))) * obs[0]
                ball_aceleration_history.append(aceleracao)

                # Passo da simulação
                step_result = env.step(action)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                else:
                    obs, reward, done, info = step_result
                    terminated = done
                    truncated = False

                if terminated or truncated:
                    st.info(f"Simulação encerrada no passo {i}.")
                    break
            
            plt.close(fig_render)
            
            # --- Geração e Exibição do GIF ---
            if frames:
                fig_anim, ax_anim = plt.subplots(figsize=(6, 3))
                plt.axis("off")
                im = ax_anim.imshow(frames[0])

                def update(j):
                    im.set_data(frames[j])
                    return [im]
                
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    ani = animation.FuncAnimation(fig_anim, update, frames=len(frames), interval=int(dt*1000), blit=True)
                
                    # Converter a animação para JSHTML
                    html_player = ani.to_jshtml()
                    
                    # Injetar CSS para mudar apenas a cor da fonte
                    css_fix = """
                    <style>
                        .anim-state label {
                            color: #7F8C8D !important; /* Cinza que contrasta bem no modo Dark e Light */
                            font-weight: bold;
                            font-family: sans-serif;
                        }
                    </style>
                    """

                    html_com_estilo = css_fix + html_player

                    st.components.v1.html(html_com_estilo, height=400, scrolling=True)
                    plt.close(fig_anim)

            # --- Geração e Exibição dos Gráficos ---

            with st.expander("Gráficos de Desempenho"):

                plot_resultado_simulacao_bola_bastao(
                    dt=dt,
                    L=L,
                    max_ang_alpha=max_ang_alpha,
                    ball_pos_history=ball_pos_history,
                    erro_history=erro_history,
                    alpha_history=alpha_history,
                    action_history=action_history,
                    ball_velocity_history=ball_velocity_history,
                    ball_aceleration_history=ball_aceleration_history
                )

                # Fechar o ambiente (Gym) por segurança após a plotagem
                env.close()

        st.sidebar.write("---")
        st.sidebar.header("Plots")

        if st.sidebar.button("Plote resposta ao degrau"):
            
            plote_resposta_MA_Bola_Bastao(m, g, j, R, q_input)
        
        if st.sidebar.button("Plote mapa de polos e zeros"):
            
            plote_mapa_polos_zeros(m=m, g=g, j=j, R=R, type= "Bola bastão MA")

        if st.sidebar.button("Plote o lugar das raízes"):
            
            plote_lugar_raizes(m=m, g=g, j=j, R=R, type= "Bola bastão MA")

        if st.sidebar.button("Plote o diagrama de Bode"):
            
            plote_bode(m=m, g=g, j=j, R=R, type= "Bola bastão MA")

        if st.sidebar.button("Plote o diagrama de Nyquist"):
            
            plote_nyquist(m=m, g=g, j=j, R=R, type="Bola bastão MA")


    
    if parte_simulacao == "Questão 6":

        with st.expander("Enunciado Questão 6"):
            enunciado_questao6(type = "Bola bastão")


        st.sidebar.header("Inputs da Simulação")
        K_feedback = st.sidebar.slider("Valor do ganho de feedback", min_value=-5.0, max_value=5.0, value=0.0, step=0.01, help=f"Valor de entrada é o K_feedback*erro [erro = referência (0 m) - posição atual da bolinha (x m)] ")
        init_velocity_input = st.sidebar.slider("Velocidade inicial da Bola", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)
        opcoes_escolha_posicao = ["Aleatório", "Canto esquerdo", "Canto direito", "Centro"]
        escolha_posicao = st.sidebar.radio("Posição inicial da bola no bastão", 
                                                  opcoes_escolha_posicao, help="Aleatório: A bola pode começar em qualquer ponto do bastão (mas não no centro e nem perto dele)." \
                                                  "\n\nCanto Esquerdo: A bola começa obrigatoriamente na extremidade esquerda.\n\nCanto Direito: A bola começa obrigatoriamente na extremidade direita.")

        if st.sidebar.button("Simular"):
            st.subheader("Simulação em Malha Fechada (Feedback)")

            env = BallBeamBalanceEnv(
                timestep=dt,
                beam_length=L,
                max_angle=max_ang_alpha,
                init_velocity=init_velocity_input,
                max_timesteps=500,
                action_mode='continuous'
            )

            env.bb.g = g
            env.bb.L = L

            state = env.reset()
            if isinstance(state, tuple):
                obs = state[0]
            else:
                obs = state
                    

            # --- MODIFICAÇÃO: POSIÇÃO INICIAL ALEATÓRIA ---
            if escolha_posicao == "Aleatório":
                random_pos_val = get_ball_start_pos(L, type=escolha_posicao)
                env.bb.x = random_pos_val
                # Sincronizar a observação com a nova posição
                obs = list(obs)
                obs[1] = random_pos_val
                obs = np.array(obs)
                st.info(f"Bola iniciando na posição aleatória: {random_pos_val:.3f}m")

            elif escolha_posicao == "Canto esquerdo":
                corner_pos_val = get_ball_start_pos(L, type=escolha_posicao)
                env.bb.x = corner_pos_val
                # Sincronizar a observação com a nova posição
                obs = list(obs)
                obs[1] = corner_pos_val
                obs = np.array(obs)
                st.info(f"Bola iniciando na extremidade esquerda: {env.bb.x:.3f}m")

            elif escolha_posicao == "Canto direito":
                corner_pos_val = get_ball_start_pos(L, type=escolha_posicao)
                env.bb.x = corner_pos_val
                # Sincronizar a observação com a nova posição
                obs = list(obs)
                obs[1] = corner_pos_val
                obs = np.array(obs)
                st.info(f"Bola iniciando na extremidade direita: {env.bb.x:.3f}m")

            elif escolha_posicao == "Centro":
                posicao_central = get_ball_start_pos(L, type=escolha_posicao)
                env.bb.x = posicao_central
                # Sincronizar a observação com a nova posição
                obs = list(obs)
                obs[1] = posicao_central
                obs = np.array(obs)
                st.info(f"Bola iniciando no meio: {env.bb.x:.3f}m")

            frames = []
            alpha_history = []
            ball_pos_history = []
            ball_velocity_history = []
            ball_aceleration_history = []
            action_history = []
            erro_history = []

            st.write("Iniciando simulação...")

            fig_render, ax_render = plt.subplots(figsize=(6, 3))

            for i in range(num_iteracoes):
                ball_x = env.bb.x
                beam_theta = env.bb.theta

                frame = render_bola_bastao_frame(fig_render, ax_render, ball_x, beam_theta, L, i)
                frames.append(frame)

                erro = (0 - ball_x)
                action = K_feedback * erro

                action = np.clip(action, -max_ang_alpha, max_ang_alpha)

                action = -1 * action #aqui

                action_history.append(action)

                # Erro é a diferença entre a posição desejada (0) e a posição atual da bola
                erro_history.append(erro)

                alpha_history.append(obs[0]) # Ângulo do bastão
                ball_pos_history.append(obs[1]) # Posição da bola
                ball_velocity_history.append(obs[2]) # Velocidade da bola

                aceleracao = - (m * g / (m + j / (R**2))) * obs[0]
                ball_aceleration_history.append(aceleracao)

                # Passo da simulação
                step_result = env.step(action)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                else:
                    obs, reward, done, info = step_result
                    terminated = done
                    truncated = False

                if terminated or truncated:
                    st.warning(f"Simulação encerrada no passo {i} (Bola caiu ou limite de tempo atingido).")
                    break
            
            plt.close(fig_render)
            
            # --- Geração e Exibição do GIF ---
            if frames:
                fig_anim, ax_anim = plt.subplots(figsize=(6, 3))
                plt.axis("off")
                im = ax_anim.imshow(frames[0])

                def update(j):
                    im.set_data(frames[j])
                    return [im]
                
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    ani = animation.FuncAnimation(fig_anim, update, frames=len(frames), interval=int(dt*1000), blit=True)
                
                    # Converter a animação para JSHTML
                    html_player = ani.to_jshtml()
                    
                    # Injetar CSS para mudar apenas a cor da fonte
                    css_fix = """
                    <style>
                        .anim-state label {
                            color: #7F8C8D !important; /* Cinza que contrasta bem no modo Dark e Light */
                            font-weight: bold;
                            font-family: sans-serif;
                        }
                    </style>
                    """

                    html_com_estilo = css_fix + html_player

                    st.components.v1.html(html_com_estilo, height=400, scrolling=True)
                    plt.close(fig_anim)

            # --- Geração e Exibição dos Gráficos ---

            with st.expander("Gráficos de Desempenho"):

                plot_resultado_simulacao_bola_bastao(
                    dt=dt,
                    L=L,
                    max_ang_alpha=max_ang_alpha,
                    ball_pos_history=ball_pos_history,
                    erro_history=erro_history,
                    alpha_history=alpha_history,
                    action_history=action_history,
                    ball_velocity_history=ball_velocity_history,
                    ball_aceleration_history=ball_aceleration_history
                )

                # Fechar o ambiente (Gym) por segurança após a plotagem
                env.close()

        st.sidebar.write("---")
        st.sidebar.header("Plots")

        if st.sidebar.button("Plote resposta em MF"):
            
            plote_resposta_MF_Bola_Bastao(m, g, j, R, K_feedback)
        
        if st.sidebar.button("Plote mapa de polos e zeros"):
            
            plote_mapa_polos_zeros(m=m, g=g, j=j, R=R, type="Bola bastão MF")

        if st.sidebar.button("Plote o lugar das raízes"):
            
            plote_lugar_raizes(m=m, g=g, j=j, R=R, type="Bola bastão MF")

        if st.sidebar.button("Plote o diagrama de Bode"):
            
            plote_bode(m=m, g=g, j=j, R=R, type="Bola bastão MF")

        if st.sidebar.button("Plote o diagrama de Nyquist"):
            
            plote_nyquist(m=m, g=g, j=j, R=R, type="Bola bastão MF")

    if parte_simulacao == "Questão 7":

        with st.expander("Enunciado Questão 7"):
            enunciado_questao7(type="Bola bastão")


        st.sidebar.header("Inputs da Simulação")
        Kp = st.sidebar.slider("Escolha um valor de Kp", min_value = -5.0, max_value = 5.0, value = 0.0, step = 0.1)
        Ki = st.sidebar.slider("Escolha um valor de Ki", min_value = -5.0, max_value = 5.0, value = 0.0, step = 0.1)
        Kd = st.sidebar.slider("Escolha um valor de Kd", min_value = -5.0, max_value = 5.0, value = 0.0, step = 0.1)
        
        init_velocity_input = st.sidebar.slider("Velocidade inicial da Bola", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)
        opcoes_escolha_posicao = ["Aleatório", "Canto esquerdo", "Canto direito", "Centro"]
        escolha_posicao = st.sidebar.radio("Posição inicial da bola no bastão", 
                                                  opcoes_escolha_posicao, help="Aleatório: A bola pode começar em qualquer ponto do bastão (mas não no centro e nem perto dele)." \
                                                  "\n\nCanto Esquerdo: A bola começa obrigatoriamente na extremidade esquerda.\n\nCanto Direito: A bola começa obrigatoriamente na extremidade direita.")

        if st.sidebar.button("Simular"):
            st.subheader("Simulação controle PID")

            env = BallBeamBalanceEnv(
                timestep=dt,
                beam_length=L,
                max_angle=max_ang_alpha,
                init_velocity=init_velocity_input,
                max_timesteps=500,
                action_mode='continuous'
            )

            env.bb.g = g
            env.bb.L = L

            state = env.reset()
            if isinstance(state, tuple):
                obs = state[0]
            else:
                obs = state
                    

            # --- MODIFICAÇÃO: POSIÇÃO INICIAL ALEATÓRIA ---
            if escolha_posicao == "Aleatório":
                random_pos_val = get_ball_start_pos(L, type=escolha_posicao)
                env.bb.x = random_pos_val
                # Sincronizar a observação com a nova posição
                obs = list(obs)
                obs[1] = random_pos_val
                obs = np.array(obs)
                st.info(f"Bola iniciando na posição aleatória: {random_pos_val:.3f}m")

            elif escolha_posicao == "Canto esquerdo":
                corner_pos_val = get_ball_start_pos(L, type=escolha_posicao)
                env.bb.x = corner_pos_val
                # Sincronizar a observação com a nova posição
                obs = list(obs)
                obs[1] = corner_pos_val
                obs = np.array(obs)
                st.info(f"Bola iniciando na extremidade esquerda: {env.bb.x:.3f}m")

            elif escolha_posicao == "Canto direito":
                corner_pos_val = get_ball_start_pos(L, type=escolha_posicao)
                env.bb.x = corner_pos_val
                # Sincronizar a observação com a nova posição
                obs = list(obs)
                obs[1] = corner_pos_val
                obs = np.array(obs)
                st.info(f"Bola iniciando na extremidade direita: {env.bb.x:.3f}m")

            elif escolha_posicao == "Centro":
                posicao_central = get_ball_start_pos(L, type=escolha_posicao)
                env.bb.x = posicao_central
                # Sincronizar a observação com a nova posição
                obs = list(obs)
                obs[1] = posicao_central
                obs = np.array(obs)
                st.info(f"Bola iniciando no meio: {env.bb.x:.3f}m")

            frames = []
            alpha_history = []
            ball_pos_history = []
            ball_velocity_history = []
            ball_aceleration_history = []
            action_history = []
            erro_history = []

            erro_integral = 0

            st.write("Iniciando simulação...")

            fig_render, ax_render = plt.subplots(figsize=(6, 3))

            for i in range(num_iteracoes):
                ball_x = env.bb.x
                beam_theta = env.bb.theta

                frame = render_bola_bastao_frame(fig_render, ax_render, ball_x, beam_theta, L, i)
                frames.append(frame)

                erro = (0 - ball_x)
                
                # Erro é a diferença entre a posição desejada (0) e a posição atual da bola
                erro_history.append(erro)

                alpha_history.append(obs[0]) # Ângulo do bastão
                ball_pos_history.append(obs[1]) # Posição da bola
                ball_velocity_history.append(obs[2]) # Velocidade da bola

                aceleracao = - (m * g / (m + j / (R**2))) * obs[0]
                ball_aceleration_history.append(aceleracao)

                velocidade_bola = obs[2]

                erro_integral += erro
                erro_integral = np.clip(erro_integral, -10.0, 10.0)

                action = Kp * erro + Kd * velocidade_bola + Ki * erro_integral

                action = np.clip(action, -max_ang_alpha, max_ang_alpha)

                action_history.append(action)

                # Passo da simulação
                step_result = env.step(action)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                else:
                    obs, reward, done, info = step_result
                    terminated = done
                    truncated = False

                if terminated or truncated:
                    st.warning(f"Simulação encerrada no passo {i} (Bola caiu ou limite de tempo atingido).")
                    break
            
            plt.close(fig_render)
            
            # --- Geração e Exibição do GIF ---
            if frames:
                fig_anim, ax_anim = plt.subplots(figsize=(6, 3))
                plt.axis("off")
                im = ax_anim.imshow(frames[0])

                def update(j):
                    im.set_data(frames[j])
                    return [im]
                
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    ani = animation.FuncAnimation(fig_anim, update, frames=len(frames), interval=int(dt*1000), blit=True)
                
                    # Converter a animação para JSHTML
                    html_player = ani.to_jshtml()
                    
                    # Injetar CSS para mudar apenas a cor da fonte
                    css_fix = """
                    <style>
                        .anim-state label {
                            color: #7F8C8D !important; /* Cinza que contrasta bem no modo Dark e Light */
                            font-weight: bold;
                            font-family: sans-serif;
                        }
                    </style>
                    """

                    html_com_estilo = css_fix + html_player

                    st.components.v1.html(html_com_estilo, height=400, scrolling=True)
                    plt.close(fig_anim)

            # --- Geração e Exibição dos Gráficos ---

            with st.expander("Gráficos de Desempenho"):

                plot_resultado_simulacao_bola_bastao(
                    dt=dt,
                    L=L,
                    max_ang_alpha=max_ang_alpha,
                    ball_pos_history=ball_pos_history,
                    erro_history=erro_history,
                    alpha_history=alpha_history,
                    action_history=action_history,
                    ball_velocity_history=ball_velocity_history,
                    ball_aceleration_history=ball_aceleration_history
                )

                # Fechar o ambiente (Gym) por segurança após a plotagem
                env.close()

        st.sidebar.write("---")
        st.sidebar.header("Plots")

        if st.sidebar.button("Plote resposta em função de Kp"):
            
            resposta_em_funcao_de_Kp(m, g, j, R)

        if st.sidebar.button("Plote resposta em função de Ki"):
            
            resposta_em_funcao_de_Ki(m, g, j, R)

        if st.sidebar.button("Plote resposta em função de Kd"):
            
            resposta_em_funcao_de_Kd(m, g, j, R)
        
        if st.sidebar.button("Plote resposta PID"):
            
            plote_resposta_PID_Bola_Bastao(m, g, j, R, Kp, Ki, Kd)

        if st.sidebar.button("Plote mapa de polos e zeros"):
            
            plote_mapa_polos_zeros(m=m, g=g, j=j, R=R, type= "Bola bastão PID", Kp=Kp, Ki=Ki, Kd=Kd)

        if st.sidebar.button("Plote o lugar das raízes"):
            
            plote_lugar_raizes(m=m, g=g, j=j, R=R, type= "Bola bastão PID", Kp=Kp, Ki=Ki, Kd=Kd)

        if st.sidebar.button("Plote o diagrama de Bode"):

            plote_bode(m=m, g=g, j=j, R=R, type= "Bola bastão PID", Kp=Kp, Ki=Ki, Kd=Kd)

        if st.sidebar.button("Plote o diagrama de Nyquist"):

            plote_nyquist(m=m, g=g, j=j, R=R, type="Bola bastão PID", Kp=Kp, Ki=Ki, Kd=Kd)

        
    if parte_simulacao == "Questão 8":
        enunciado_questao8(type="Bola bastão")

    if parte_simulacao == "Questão 9":
        enunciado_questao9(type="Bola bastão")

    if parte_simulacao == "Questão 10":
        enunciado_questao10(type="Bola bastão")

if sistema == "Pêndulo simples invertido":
    st.title("Simulação do Sistema Pêndulo Simples Invertido")

    with st.expander("Representação do sistema Pêndulo Simples Invertido"):
        col1, col2, col3 = st.columns([2, 1, 2])
        col2.image("pendulum.png")

    st.sidebar.write("---")
    # with open("Roteiro Pêndulo.docx", "rb") as file:
    #     btn = st.sidebar.download_button(
    #         label="Baixar Roteiro",
    #         icon="🚨",
    #         data=file,
    #         file_name="Roteiro Pêndulo.docx",
    #         mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    #     )
    st.sidebar.write("---")

    parte_simulacao = st.sidebar.selectbox(
        "Selecione a questão da simulação",
        ("Questão 1", "Questão 2", "Questão 3", "Questão 4", "Questão 5", "Questão 6", "Questão 7", "Questão 8", "Questão 9", "Questão 10")
    )

    st.sidebar.write("---")

    # --- Parâmetros do Sistema (fixos para esta simulação) ---
    b = 0.01 #Coeficiente de atrito

    l_cm = 20 #cm
    d = 2 #cm
    v = np.pi * d**2/4 * l_cm #cm³
    m_g = 1.25 * v #1,25 g/cm³
    g = gravidade = 9.81

    lim_motor = 0.1

    m = m_g/1000
    peso = m * g # em N
    L = l_cm/100
    torque_min_N = peso * (L/2) #centro de massa na metade da haste
    torque_min_kgfcm = torque_min_N * 10.197

    if parte_simulacao == "Questão 1":
        st.warning("Conexão com a esp32")
        if st.button("Liga/Desliga Led"):
            toggle_led()
        
    if parte_simulacao == "Questão 2":
        enunciado_questao2(type="Pêndulo simples invertido")

    if parte_simulacao == "Questão 3":
        enunciado_questao3(type="Pêndulo simples invertido")

    if parte_simulacao == "Questão 4":
        enunciado_questao4(type="Pêndulo simples invertido")

    if parte_simulacao == "Questão 5":

        with st.expander("Enunciado Questão 5"):
            enunciado_questao5(type="Pêndulo simples invertido")

        # --- Inputs do Usuário ---
        st.sidebar.header("Inputs da Simulação")
        swing_up_true = st.sidebar.checkbox("Controle com Swing-up?", value=True, help="Defina se a ação vai ser dividida para duas condições ou não, conforme descrito no enunciado.")
        q_input = st.sidebar.slider("Valor da entrada degrau", min_value=-lim_motor, max_value=lim_motor, value=0.0, step=0.005, help=f"Valor de entrada é o torque do motor indo de -{lim_motor} até {lim_motor} Nm")
        #init_velocity_input = st.sidebar.slider("Velocidade inicial da Bola", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)
        #opcoes_escolha_posicao = ["Aleatório", "Canto esquerdo", "Canto direito", "Centro"]
        #escolha_posicao = st.sidebar.radio("Posição inicial da bola no bastão", 
        #                                          opcoes_escolha_posicao, help="Aleatório: A bola pode começar em qualquer ponto do bastão (mas não no centro e nem perto dele)." \
        #                                          "\nCanto Esquerdo: A bola começa obrigatoriamente na extremidade esquerda.\nCanto Direito: A bola começa obrigatoriamente na extremidade direita.")

        if st.sidebar.button("Simular"):

            env = gym.make("Pendulum-v1", render_mode="rgb_array", g=gravidade, max_episode_steps=500)
            env.unwrapped.m = m
            env.unwrapped.l = L
            env.unwrapped.g = gravidade

            state, _ = env.reset()

            frames = []
            control_type_history = [] # 0 para Swing-up, 1 para Degrau (Catch)
            theta_history = []
            theta_dot_history = []
            theta_double_dot_history = []
            external_action_history = []
            erro_history = []
            
            num_iteracoes = 250

            k_SWING_UP = 0.037
            q = q_input

            fig_render, ax_render = plt.subplots(figsize=(6, 3))

            for i in range(num_iteracoes):
                frames.append(env.render())

                theta = np.arctan2(state[1], state[0])
                valor_graus = np.degrees(theta)
                theta_history.append(theta)
                
                theta_dot = state[2]
                theta_dot_history.append(theta_dot)

                if swing_up_true:

                    # Lógica de Transição
                    if abs(valor_graus) <= 20:
                        # FASE DE CATCH - Degrau
                        erro = (0 - theta)
                        erro_history.append(erro)
                        torque = q 
                        torque = np.clip(torque, -lim_motor, lim_motor)
                        external_action_history.append(torque)
                        resistencia_torque = -b * theta_dot
                        action = [torque + resistencia_torque]
                        control_type_history.append(1) # Identificador para o gráfico
                    else:
                        # FASE DE SWING-UP (BALANÇO)
                        erro = (0 - theta)
                        erro_history.append(erro)
                        torque_swing = k_SWING_UP * theta_dot
                        torque_swing = np.clip(torque_swing, -lim_motor, lim_motor)
                        external_action_history.append(torque_swing)
                        resistencia_torque = -b * theta_dot
                        action = [torque_swing + resistencia_torque]
                        control_type_history.append(0) # Identificador para o gráfico

                else:
                    erro = (0 - theta)
                    erro_history.append(erro)
                    torque = q 
                    torque = np.clip(torque, -lim_motor, lim_motor)
                    external_action_history.append(torque)
                    resistencia_torque = -b * theta_dot
                    action = [torque + resistencia_torque]
                    control_type_history.append(1) # Identificador para o gráfico

                state, reward, terminated, truncated, info = env.step(action)

                theta_2dot = 3*action[0]/((m*L**2)) + 3*gravidade*np.sin(theta)/(2*L)

                theta_double_dot_history.append(theta_2dot)

                if terminated or truncated:
                    break

            env.close()

            # --- Geração e Exibição da Animação no Streamlit ---
            if frames:
                # Cria a figura para a animação
                fig_anim, ax_anim = plt.subplots(figsize=(5, 5)) 
                
                # Remove qualquer borda branca excedente do Matplotlib
                fig_anim.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
                ax_anim.axis("off") 
                im = ax_anim.imshow(frames[0])

                txt = ax_anim.text(0.5, 0.95, '', transform=ax_anim.transAxes, 
                   ha='center', va='center', fontsize=12, fontweight='bold')

                def update(j):
                    # Atualiza a imagem
                    im.set_data(frames[j])
                    
                    # Lógica do Label (ajuste 'control_type_history' para o nome da sua variável)
                    label = "DEGRAU (CATCH)" if control_type_history[j] == 1 else "SWING-UP (BALANÇO)"
                    color = "lime" if control_type_history[j] == 1 else "orange"
                    
                    txt.set_text(label)
                    txt.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor=color, boxstyle='round,pad=0.5'))
                    
                    return [im, txt]
                
                # Mudar para [1, 2, 1] costuma dar um resultado visualmente mais centralizado que [1, 3, 1]
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    intervalo_ms = int(dt * 1000) if 'dt' in locals() else 50
                    
                    ani = animation.FuncAnimation(fig_anim, update, frames=len(frames), interval=intervalo_ms, blit=True)
                
                    # Converter a animação para JSHTML
                    html_player = ani.to_jshtml()
                    
                    # CSS FIXO: Adicionadas as regras no 'body' e '.animation' para alinhar tudo ao centro
                    css_fix = """
                    <style>
                        /* Centraliza o conteúdo dentro do iframe gerado pelo Streamlit */
                        body {
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            flex-direction: column;
                            margin: 0;
                            padding: 0;
                            overflow: hidden; /* Previne barras de rolagem desnecessárias */
                        }
                        /* Alinha os controles (botões) e a imagem do Matplotlib */
                        .animation {
                            display: flex;
                            flex-direction: column;
                            align-items: center;
                            margin: 0 auto;
                        }
                        /* Conserta a cor da fonte para Dark/Light mode */
                        .anim-state label {
                            color: #7F8C8D !important;
                            font-weight: bold;
                            font-family: sans-serif;
                        }
                    </style>
                    """

                    html_com_estilo = css_fix + html_player

                    # Renderizar no Streamlit: scrolling=False remove as barras de rolagem cinzas laterais
                    st.components.v1.html(html_com_estilo, height=580, scrolling=False)
                    plt.close(fig_anim)

            with st.expander("Gráficos de Desempenho"):
            
                # Se a variável dt não estiver declarada globalmente no seu código,
                # você pode forçar o dt padrão do Gym: dt_sim = 0.05
                dt_sim = dt if 'dt' in locals() else 0.05 

                plot_resultado_simulacao_pendulo(
                    dt=dt_sim,
                    lim_motor=lim_motor,
                    erro_history=erro_history,
                    external_action_history=external_action_history,
                    theta_history=theta_history,
                    theta_dot_history=theta_dot_history,
                    theta_double_dot_history=theta_double_dot_history,
                    control_type_history=control_type_history
                )

        st.sidebar.write("---")
        st.sidebar.header("Plots")

        if st.sidebar.button("Plote resposta ao degrau"):
            
            plote_resposta_MA_Pendulo_simples_invertido(m, g, L, b, q_input)
        
        if st.sidebar.button("Plote mapa de polos e zeros"):
            
            plote_mapa_polos_zeros(m=m, g=g, L=L, b=b, type= "Pêndulo simples invertido MA")

        if st.sidebar.button("Plote o lugar das raízes"):
            
            plote_lugar_raizes(m=m, g=g, L=L, b=b, type= "Pêndulo simples invertido MA")

        if st.sidebar.button("Plote o diagrama de Bode"):
            
            plote_bode(m=m, g=g, L=L, b=b, type= "Pêndulo simples invertido MA")

        if st.sidebar.button("Plote o diagrama de Nyquist"):
            
            plote_nyquist(m=m, g=g, L=L, b=b, type= "Pêndulo simples invertido MA")

    if parte_simulacao == "Questão 6":

        with st.expander("Enunciado Questão 6"):
            enunciado_questao6(type="Pêndulo simples invertido")

        # --- Inputs do Usuário ---
        st.sidebar.header("Inputs da Simulação")
        swing_up_true = st.sidebar.checkbox("Controle com Swing-up?", value=True, help="Defina se a ação vai ser dividida para duas condições ou não, conforme descrito no enunciado.")
        K_feedback = st.sidebar.slider("Valor do ganho de feedback", min_value=-lim_motor, max_value=lim_motor, value=0.0, step=0.005, help=f"Valor de entrada é o ganho de feedback * erro (erro = 0 - theta) indo de -{lim_motor} até {lim_motor} Nm")

        if st.sidebar.button("Simular"):

            env = gym.make("Pendulum-v1", render_mode="rgb_array", g=gravidade, max_episode_steps=500)
            env.unwrapped.m = m
            env.unwrapped.l = L
            env.unwrapped.g = gravidade

            state, _ = env.reset()

            frames = []
            control_type_history = [] # 0 para Swing-up, 1 para Feedback (Catch)
            theta_history = []
            theta_dot_history = []
            theta_double_dot_history = []
            external_action_history = []
            erro_history = []
            
            num_iteracoes = 250

            k_SWING_UP = 0.037

            fig_render, ax_render = plt.subplots(figsize=(6, 3))

            for i in range(num_iteracoes):
                frames.append(env.render())

                theta = np.arctan2(state[1], state[0])
                valor_graus = np.degrees(theta)
                theta_history.append(theta)
                
                theta_dot = state[2]
                theta_dot_history.append(theta_dot)

                if swing_up_true:

                    # Lógica de Transição
                    if abs(valor_graus) <= 20:

                        # FASE DE CATCH - Feedback
                        erro = (0 - theta)
                        erro_history.append(erro)

                        torque = K_feedback * erro
                        torque = np.clip(torque, -lim_motor, lim_motor)
                        external_action_history.append(torque)
                        resistencia_torque = -b * theta_dot
                        action = [torque + resistencia_torque]
                        control_type_history.append(1) # Identificador para o gráfico
                    
                    else:
                        # FASE DE SWING-UP (BALANÇO)
                        erro = (0 - theta)
                        erro_history.append(erro)

                        torque_swing = k_SWING_UP * theta_dot
                        torque_swing = np.clip(torque_swing, -lim_motor, lim_motor)
                        external_action_history.append(torque_swing)
                        resistencia_torque = -b * theta_dot
                        action = [torque_swing + resistencia_torque]
                        control_type_history.append(0) # Identificador para o gráfico

                else:
                    erro = (0 - theta)
                    erro_history.append(erro)

                    torque = K_feedback * erro
                    torque = np.clip(torque, -lim_motor, lim_motor)
                    external_action_history.append(torque)
                    resistencia_torque = -b * theta_dot
                    action = [torque + resistencia_torque]
                    control_type_history.append(1) # Identificador para o gráfico
                    
                state, reward, terminated, truncated, info = env.step(action)

                theta_2dot = 3*action[0]/((m*L**2)) + 3*gravidade*np.sin(theta)/(2*L)

                theta_double_dot_history.append(theta_2dot)

                if terminated or truncated:
                    break

            env.close()

            # --- Geração e Exibição da Animação no Streamlit ---
            if frames:
                # Cria a figura para a animação
                fig_anim, ax_anim = plt.subplots(figsize=(5, 5)) 
                
                # Remove qualquer borda branca excedente do Matplotlib
                fig_anim.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
                ax_anim.axis("off") 
                im = ax_anim.imshow(frames[0])

                txt = ax_anim.text(0.5, 0.95, '', transform=ax_anim.transAxes, 
                   ha='center', va='center', fontsize=12, fontweight='bold')

                def update(j):
                    # Atualiza a imagem
                    im.set_data(frames[j])
                    
                    # Lógica do Label (ajuste 'control_type_history' para o nome da sua variável)
                    label = "FEEDBACK (CATCH)" if control_type_history[j] == 1 else "SWING-UP (BALANÇO)"
                    color = "lime" if control_type_history[j] == 1 else "orange"
                    
                    txt.set_text(label)
                    txt.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor=color, boxstyle='round,pad=0.5'))
                    
                    return [im, txt]
                
                # Mudar para [1, 2, 1] costuma dar um resultado visualmente mais centralizado que [1, 3, 1]
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    intervalo_ms = int(dt * 1000) if 'dt' in locals() else 50
                    
                    ani = animation.FuncAnimation(fig_anim, update, frames=len(frames), interval=intervalo_ms, blit=True)
                
                    # Converter a animação para JSHTML
                    html_player = ani.to_jshtml()
                    
                    # CSS FIXO: Adicionadas as regras no 'body' e '.animation' para alinhar tudo ao centro
                    css_fix = """
                    <style>
                        /* Centraliza o conteúdo dentro do iframe gerado pelo Streamlit */
                        body {
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            flex-direction: column;
                            margin: 0;
                            padding: 0;
                            overflow: hidden; /* Previne barras de rolagem desnecessárias */
                        }
                        /* Alinha os controles (botões) e a imagem do Matplotlib */
                        .animation {
                            display: flex;
                            flex-direction: column;
                            align-items: center;
                            margin: 0 auto;
                        }
                        /* Conserta a cor da fonte para Dark/Light mode */
                        .anim-state label {
                            color: #7F8C8D !important;
                            font-weight: bold;
                            font-family: sans-serif;
                        }
                    </style>
                    """

                    html_com_estilo = css_fix + html_player

                    # Renderizar no Streamlit: scrolling=False remove as barras de rolagem cinzas laterais
                    st.components.v1.html(html_com_estilo, height=580, scrolling=False)
                    plt.close(fig_anim)

            with st.expander("Gráficos de Desempenho"):
            
                # Se a variável dt não estiver declarada globalmente no seu código,
                # você pode forçar o dt padrão do Gym: dt_sim = 0.05
                dt_sim = dt if 'dt' in locals() else 0.05 

                plot_resultado_simulacao_pendulo(
                    dt=dt_sim,
                    lim_motor=lim_motor,
                    erro_history=erro_history,
                    external_action_history=external_action_history,
                    theta_history=theta_history,
                    theta_dot_history=theta_dot_history,
                    theta_double_dot_history=theta_double_dot_history,
                    control_type_history=control_type_history
                )

        st.sidebar.write("---")
        st.sidebar.header("Plots")

        if st.sidebar.button("Plote resposta ao degrau"):
            
            plote_resposta_MF_Pendulo_simples_invertido(m, g, L, b, K_feedback)
        
        if st.sidebar.button("Plote mapa de polos e zeros"):
            
            plote_mapa_polos_zeros(m=m, g=g, L=L, b=b, type= "Pêndulo simples invertido MF")

        if st.sidebar.button("Plote o lugar das raízes"):
            
            plote_lugar_raizes(m=m, g=g, L=L, b=b, type= "Pêndulo simples invertido MF")

        if st.sidebar.button("Plote o diagrama de Bode"):
            
            plote_bode(m=m, g=g, L=L, b=b, type= "Pêndulo simples invertido MF")

        if st.sidebar.button("Plote o diagrama de Nyquist"):
            
            plote_nyquist(m=m, g=g, L=L, b=b, type= "Pêndulo simples invertido MF")

    if parte_simulacao == "Questão 7":

        with st.expander("Enunciado Questão 7"):
            enunciado_questao7(type="Pêndulo simples invertido")

        # --- Inputs do Usuário ---
        st.sidebar.header("Inputs da Simulação")
        swing_up_true = st.sidebar.checkbox("Controle com Swing-up?", value=True, help="Defina se a ação vai ser dividida para duas condições ou não, conforme descrito no enunciado.")
        st.sidebar.header("Inputs da Simulação")
        Kp = st.sidebar.slider("Escolha um valor de Kp", min_value = -lim_motor, max_value = lim_motor, value = 0.0, step = 0.01)
        Ki = st.sidebar.slider("Escolha um valor de Ki", min_value = -lim_motor, max_value = lim_motor, value = 0.0, step = 0.01)
        Kd = st.sidebar.slider("Escolha um valor de Kd", min_value = -lim_motor, max_value = lim_motor, value = 0.0, step = 0.01)

        if st.sidebar.button("Simular"):

            env = gym.make("Pendulum-v1", render_mode="rgb_array", g=gravidade, max_episode_steps=500)
            env.unwrapped.m = m
            env.unwrapped.l = L
            env.unwrapped.g = gravidade

            state, _ = env.reset()

            frames = []
            control_type_history = [] # 0 para Swing-up, 1 para Feedback (Catch)
            theta_history = []
            theta_dot_history = []
            theta_double_dot_history = []
            external_action_history = []
            erro_history = []

            erro_integral = 0
            
            num_iteracoes = 250

            k_SWING_UP = 0.037

            fig_render, ax_render = plt.subplots(figsize=(6, 3))

            for i in range(num_iteracoes):
                frames.append(env.render())

                theta = np.arctan2(state[1], state[0])
                valor_graus = np.degrees(theta)
                theta_history.append(theta)
                
                theta_dot = state[2]
                theta_dot_history.append(theta_dot)

                if swing_up_true:

                    # Lógica de Transição
                    if abs(valor_graus) <= 20:

                        # FASE DE CATCH - Feedback
                        erro = (0 - theta)
                        erro_history.append(erro)

                        erro_integral += erro
                        erro_integral = np.clip(erro_integral, -10.0, 10.0) #Não permite que a integral cresça infinitamente

                        torque = +Kp * theta + Kd * theta_dot + Ki * erro_integral
                        torque = np.clip(torque, -lim_motor, lim_motor)
                        external_action_history.append(torque)
                        resistencia_torque = -b * theta_dot
                        action = [torque + resistencia_torque]
                        control_type_history.append(1) # Identificador para o gráfico
                    
                    else:
                        # FASE DE SWING-UP (BALANÇO)
                        erro = (0 - theta)
                        erro_history.append(erro)

                        erro_integral += erro
                        erro_integral = np.clip(erro_integral, -10.0, 10.0) #Não permite que a integral cresça infinitamente

                        torque_swing = k_SWING_UP * theta_dot
                        torque_swing = np.clip(torque_swing, -lim_motor, lim_motor)
                        external_action_history.append(torque_swing)
                        resistencia_torque = -b * theta_dot
                        action = [torque_swing + resistencia_torque]
                        control_type_history.append(0) # Identificador para o gráfico

                else:
                    erro = (0 - theta)
                    erro_history.append(erro)

                    erro_integral += erro
                    erro_integral = np.clip(erro_integral, -10.0, 10.0) #Não permite que a integral cresça infinitamente

                    torque = +Kp * theta + Kd * theta_dot + Ki * erro_integral
                    torque = np.clip(torque, -lim_motor, lim_motor)
                    external_action_history.append(torque)
                    resistencia_torque = -b * theta_dot
                    action = [torque + resistencia_torque]
                    control_type_history.append(1) # Identificador para o gráfico
                    
                state, reward, terminated, truncated, info = env.step(action)

                theta_2dot = 3*action[0]/((m*L**2)) + 3*gravidade*np.sin(theta)/(2*L)

                theta_double_dot_history.append(theta_2dot)

                if terminated or truncated:
                    break

            env.close()

            # --- Geração e Exibição da Animação no Streamlit ---
            if frames:
                # Cria a figura para a animação
                fig_anim, ax_anim = plt.subplots(figsize=(5, 5)) 
                
                # Remove qualquer borda branca excedente do Matplotlib
                fig_anim.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
                ax_anim.axis("off") 
                im = ax_anim.imshow(frames[0])

                txt = ax_anim.text(0.5, 0.95, '', transform=ax_anim.transAxes, 
                   ha='center', va='center', fontsize=12, fontweight='bold')

                def update(j):
                    # Atualiza a imagem
                    im.set_data(frames[j])
                    
                    # Lógica do Label (ajuste 'control_type_history' para o nome da sua variável)
                    label = "PID (CATCH)" if control_type_history[j] == 1 else "SWING-UP (BALANÇO)"
                    color = "lime" if control_type_history[j] == 1 else "orange"
                    
                    txt.set_text(label)
                    txt.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor=color, boxstyle='round,pad=0.5'))
                    
                    return [im, txt]
                
                # Mudar para [1, 2, 1] costuma dar um resultado visualmente mais centralizado que [1, 3, 1]
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    intervalo_ms = int(dt * 1000) if 'dt' in locals() else 50
                    
                    ani = animation.FuncAnimation(fig_anim, update, frames=len(frames), interval=intervalo_ms, blit=True)
                
                    # Converter a animação para JSHTML
                    html_player = ani.to_jshtml()
                    
                    # CSS FIXO: Adicionadas as regras no 'body' e '.animation' para alinhar tudo ao centro
                    css_fix = """
                    <style>
                        /* Centraliza o conteúdo dentro do iframe gerado pelo Streamlit */
                        body {
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            flex-direction: column;
                            margin: 0;
                            padding: 0;
                            overflow: hidden; /* Previne barras de rolagem desnecessárias */
                        }
                        /* Alinha os controles (botões) e a imagem do Matplotlib */
                        .animation {
                            display: flex;
                            flex-direction: column;
                            align-items: center;
                            margin: 0 auto;
                        }
                        /* Conserta a cor da fonte para Dark/Light mode */
                        .anim-state label {
                            color: #7F8C8D !important;
                            font-weight: bold;
                            font-family: sans-serif;
                        }
                    </style>
                    """

                    html_com_estilo = css_fix + html_player

                    # Renderizar no Streamlit: scrolling=False remove as barras de rolagem cinzas laterais
                    st.components.v1.html(html_com_estilo, height=580, scrolling=False)
                    plt.close(fig_anim)

            with st.expander("Gráficos de Desempenho"):
            
                # Se a variável dt não estiver declarada globalmente no seu código,
                # você pode forçar o dt padrão do Gym: dt_sim = 0.05
                dt_sim = dt if 'dt' in locals() else 0.05 

                plot_resultado_simulacao_pendulo(
                    dt=dt_sim,
                    lim_motor=lim_motor,
                    erro_history=erro_history,
                    external_action_history=external_action_history,
                    theta_history=theta_history,
                    theta_dot_history=theta_dot_history,
                    theta_double_dot_history=theta_double_dot_history,
                    control_type_history=control_type_history
                )

        st.sidebar.write("---")
        st.sidebar.header("Plots")

        if st.sidebar.button("Plote resposta ao degrau"):
    
            plote_resposta_PID_Pendulo_simples_invertido(m, g, L, b, Kp, Ki, Kd)
        
        if st.sidebar.button("Plote mapa de polos e zeros"):
            
            plote_mapa_polos_zeros(m=m, g=g, L=L, b=b, Kp=Kp, Ki=Ki, Kd=Kd, type= "Pêndulo simples invertido PID")

        if st.sidebar.button("Plote o lugar das raízes"):
            
            plote_lugar_raizes(m=m, g=g, L=L, b=b, Kp=Kp, Ki=Ki, Kd=Kd, type= "Pêndulo simples invertido PID")

        if st.sidebar.button("Plote o diagrama de Bode"):
            
            plote_bode(m=m, g=g, L=L, b=b, Kp=Kp, Ki=Ki, Kd=Kd, type= "Pêndulo simples invertido PID")

        if st.sidebar.button("Plote o diagrama de Nyquist"):
            
            plote_nyquist(m=m, g=g, L=L, b=b, Kp=Kp, Ki=Ki, Kd=Kd, type= "Pêndulo simples invertido PID")

    if parte_simulacao == "Questão 8":
        enunciado_questao8(type="Pêndulo simples invertido")

    if parte_simulacao == "Questão 9":
        enunciado_questao9(type="Pêndulo simples invertido")

    if parte_simulacao == "Questão 10":
        enunciado_questao10(type="Pêndulo simples invertido")
# else:
#     None
