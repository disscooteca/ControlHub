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
import plotly.graph_objects as go

# --- Monkey Patch para corrigir o erro do matplotlib --- #
# Isso é necessário porque o ballbeam-gym tenta usar set_window_title
# em um backend não interativo, o que causa um AttributeError.
# Comentamos a linha diretamente no arquivo ballbeam.py anteriormente.
# Se o erro persistir, uma solução alternativa seria:
# import matplotlib
# matplotlib.figure.FigureCanvasAgg.set_window_title = lambda self, title: None

st.set_page_config(
    page_title="Simulações Controle",
    page_icon="⚙️",
    layout="wide" 
)

sistema = st.sidebar.selectbox(
    "Escolha qual sistema deseja simular:",
    ("Bola bastão", "Pêndulo simples invertido", "Pêndulo furuta")
)

if sistema == "Bola bastão":
    
    st.title("Simulação do Sistema Bola e Bastão")

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
        ("Questão 1", "Questão 2", "Questão 3")
    )
    st.sidebar.write("---")

    if parte_simulacao == "Questão 1":

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
    
    if parte_simulacao == "Questão 2":
        st.sidebar.header("Inputs da Simulação")
        sinal_negativo_feedback = st.sidebar.checkbox("feedback negativo", value=True, help=f"Marcado: a entrada do sistema será (0 - posição da bola)."\
                                             " Isso pois u = (meio do bastão) - (distância da bola ao meio).\n"
                                             "\nDesmarcado: a entrada do sistema será (posição da bola - 0).")
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

            if sinal_negativo_feedback == True:
                st.write("Erro = - posição da bola")
                    

            elif sinal_negativo_feedback == False:
                st.write("Erro = + posição da bola")
                    

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
                
                if sinal_negativo_feedback == True:
                    erro = (0 - ball_x)
                    action = erro

                elif sinal_negativo_feedback == False:
                    erro = (ball_x)
                    action = erro

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

    if parte_simulacao == "Questão 3":
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

if sistema == "Pêndulo simples invertido":
    st.title("Simulação do Sistema Bola e Bastão")

    st.sidebar.write("---")
    st.sidebar.button("Baixar Relatório", on_click=baixar_relatorio_pendulo_simples, icon="🚨")
    st.sidebar.write("---")


else:
    None
