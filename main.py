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
from src.utils import baixar_relatorio_bola_bastao, baixar_relatorio_pendulo_simples, get_ball_start_pos, render_bola_bastao_frame, plot_resultado_simulacao_bola_bastao
from src.utils import enunciado_questao2, enunciado_questao3, enunciado_questao4, enunciado_questao5, enunciado_questao6, enunciado_questao7, enunciado_questao8, enunciado_questao9, enunciado_questao10
from src.utils import plote_resposta_MA_Bola_Bastao, plote_resposta_MF_Bola_Bastao, plote_resposta_PID_Bola_Bastao, plote_mapa_polos_zeros, plote_lugar_raizes, plote_bode, plote_nyquist
from src.utils import resposta_em_funcao_de_Kp, resposta_em_funcao_de_Ki, resposta_em_funcao_de_Kd
import plotly.graph_objects as go

# --- Monkey Patch para corrigir o erro do matplotlib --- #
# Isso é necessário porque o ballbeam-gym tenta usar set_window_title
# em um backend não interativo, o que causa um AttributeError.
# Comentamos a linha diretamente no arquivo ballbeam.py anteriormente.
# Se o erro persistir, uma solução alternativa seria:
# import matplotlib
# matplotlib.figure.FigureCanvasAgg.set_window_title = lambda self, title: None

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
    st.sidebar.button("Baixar Relatório", on_click=baixar_relatorio_bola_bastao, icon="🚨")
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
    num_iteracoes = amostras # Usar o número total de amostras como iterações
    num_iteracoes = 150

    parte_simulacao = st.sidebar.selectbox(
        "Selecione a questão da simulação",
        ("Questão 1", "Questão 2", "Questão 3", "Questão 4", "Questão 5", "Questão 6", "Questão 7", "Questão 8", "Questão 9", "Questão 10")
    )

    st.sidebar.write("---")

    if parte_simulacao == "Questão 1":
        st.warning("Conexão com a esp32")

    if parte_simulacao == "Questão 2":

        enunciado_questao2()

    if parte_simulacao == "Questão 3":

        enunciado_questao3()

    if parte_simulacao == "Questão 4":

        enunciado_questao4(L, d)

    if parte_simulacao == "Questão 5":

        with st.expander("Enunciado Questão 5"):
            enunciado_questao5()

        if 'exibir_mensagem' not in st.session_state:
            st.session_state.exibir_mensagem = True

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

            

            st.session_state.exibir_mensagem = False

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
            st.session_state.exibir_mensagem = False
            plote_resposta_MA_Bola_Bastao(m, g, j, R, q_input)
        
        if st.sidebar.button("Plote mapa de polos e zeros"):
            st.session_state.exibir_mensagem = False
            plote_mapa_polos_zeros(m, g, j, R, type= "Bola bastão MA")

        if st.sidebar.button("Plote o lugar das raízes"):
            st.session_state.exibir_mensagem = False
            plote_lugar_raizes(m, g, j, R, type= "Bola bastão MA")

        if st.sidebar.button("Plote o diagrama de Bode"):
            st.session_state.exibir_mensagem = False
            plote_bode(m, g, j, R, type= "Bola bastão MA")

        if st.sidebar.button("Plote o diagrama de Nyquist"):
            st.session_state.exibir_mensagem = False
            plote_nyquist(m, g, j, R, type="Bola bastão MA")

        if st.session_state.exibir_mensagem:
            st.info("💡 Simule o sistema clicando no botão SIMULE na barra lateral.")
    
    if parte_simulacao == "Questão 6":

        with st.expander("Enunciado Questão 6"):
            enunciado_questao6()

        if 'exibir_mensagem' not in st.session_state:
            st.session_state.exibir_mensagem = True

        st.sidebar.header("Inputs da Simulação")
        K_feedback = st.sidebar.slider("Valor do ganho de feedback", min_value=-5.0, max_value=5.0, value=0.0, step=0.01, help=f"Valor de entrada é o K_feedback*erro [erro = referência (0 m) - posição atual da bolinha (x m)] ")
        init_velocity_input = st.sidebar.slider("Velocidade inicial da Bola", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)
        opcoes_escolha_posicao = ["Aleatório", "Canto esquerdo", "Canto direito", "Centro"]
        escolha_posicao = st.sidebar.radio("Posição inicial da bola no bastão", 
                                                  opcoes_escolha_posicao, help="Aleatório: A bola pode começar em qualquer ponto do bastão (mas não no centro e nem perto dele)." \
                                                  "\n\nCanto Esquerdo: A bola começa obrigatoriamente na extremidade esquerda.\n\nCanto Direito: A bola começa obrigatoriamente na extremidade direita.")

        if st.sidebar.button("Simular"):
            st.subheader("Simulação em Malha Fechada (Feedback)")

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

                erro = (0 - ball_x)
                action = K_feedback * erro

                action = np.clip(action, -max_ang_alpha, max_ang_alpha)

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
            st.session_state.exibir_mensagem = False
            plote_resposta_MF_Bola_Bastao(m, g, j, R, K_feedback)
        
        if st.sidebar.button("Plote mapa de polos e zeros"):
            st.session_state.exibir_mensagem = False
            plote_mapa_polos_zeros(m, g, j, R, type="Bola bastão MF")

        if st.sidebar.button("Plote o lugar das raízes"):
            st.session_state.exibir_mensagem = False
            plote_lugar_raizes(m, g, j, R, type="Bola bastão MF")

        if st.sidebar.button("Plote o diagrama de Bode"):
            st.session_state.exibir_mensagem = False
            plote_bode(m, g, j, R, type="Bola bastão MF")

        if st.sidebar.button("Plote o diagrama de Nyquist"):
            st.session_state.exibir_mensagem = False
            plote_nyquist(m, g, j, R, type="Bola bastão MF")

        if st.session_state.exibir_mensagem:
            st.info("💡 Simule o sistema clicando no botão SIMULE na barra lateral.")

    if parte_simulacao == "Questão 7":

        with st.expander("Enunciado Questão 7"):
            enunciado_questao7()

        if 'exibir_mensagem' not in st.session_state:
            st.session_state.exibir_mensagem = True

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

            erro_integral = 0

            st.write("Iniciando simulação...")

            # Criar figura para renderização manual
            fig_render, ax_render = plt.subplots(figsize=(6, 3))

            for i in range(num_iteracoes):
                # Pegar dados atuais do motor de física interno
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
            st.session_state.exibir_mensagem = False
            resposta_em_funcao_de_Kp(m, g, j, R)

        if st.sidebar.button("Plote resposta em função de Ki"):
            st.session_state.exibir_mensagem = False
            resposta_em_funcao_de_Ki(m, g, j, R)

        if st.sidebar.button("Plote resposta em função de Kd"):
            st.session_state.exibir_mensagem = False
            resposta_em_funcao_de_Kd(m, g, j, R)
        
        if st.sidebar.button("Plote resposta PID"):
            st.session_state.exibir_mensagem = False
            plote_resposta_PID_Bola_Bastao(m, g, j, R, Kp, Ki, Kd)

        if st.sidebar.button("Plote mapa de polos e zeros"):
            st.session_state.exibir_mensagem = False
            plote_mapa_polos_zeros(m, g, j, R, type= "Bola bastão PID", Kp=Kp, Ki=Ki, Kd=Kd)

        if st.sidebar.button("Plote o lugar das raízes"):
            st.session_state.exibir_mensagem = False
            plote_lugar_raizes(m, g, j, R, type= "Bola bastão PID", Kp=Kp, Ki=Ki, Kd=Kd)

        if st.sidebar.button("Plote o diagrama de Bode"):
            st.session_state.exibir_mensagem = False
            plote_bode(m, g, j, R, type= "Bola bastão PID", Kp=Kp, Ki=Ki, Kd=Kd)

        if st.sidebar.button("Plote o diagrama de Nyquist"):
            st.session_state.exibir_mensagem = False
            plote_nyquist(m, g, j, R, type="Bola bastão PID", Kp=Kp, Ki=Ki, Kd=Kd)

        if st.session_state.exibir_mensagem:
            st.info("💡 Simule o sistema clicando no botão SIMULE na barra lateral.")

    if parte_simulacao == "Questão 8":
        enunciado_questao8()

    if parte_simulacao == "Questão 9":
        enunciado_questao9()

    if parte_simulacao == "Questão 10":
        enunciado_questao10()

if sistema == "Pêndulo simples invertido":
    st.title("Simulação do Sistema Bola e Bastão")

    st.sidebar.write("---")
    st.sidebar.button("Baixar Relatório", on_click=baixar_relatorio_pendulo_simples, icon="🚨")
    st.sidebar.write("---")


# else:
#     None
