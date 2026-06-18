import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ballbeam_gym.envs.balance import BallBeamBalanceEnv
import gymnasium as gym
from ballbeam_gym.envs.balance import BallBeamBalanceEnv
from src.utils import get_ball_start_pos, plote_resposta_no_tempo, render_bola_bastao_frame, plot_resultado_simulacao_bola_bastao, plot_resultado_simulacao_pendulo
from src.utils import enunciado_questao1, enunciado_questao2, enunciado_questao3, enunciado_questao4, enunciado_questao5, enunciado_questao6, enunciado_questao7, enunciado_questao8, enunciado_questao9, enunciado_questao10
from src.utils import plote_resposta_MA_Bola_Bastao, plote_resposta_MF_Bola_Bastao, plote_resposta_PID_Bola_Bastao, plote_resposta_MA_Pendulo_simples_invertido, plote_resposta_MF_Pendulo_simples_invertido, plote_resposta_PID_Pendulo_simples_invertido
from src.utils import resposta_pendulo_em_funcao_de_Kp, resposta_pendulo_em_funcao_de_Ki, resposta_pendulo_em_funcao_de_Kd
from src.connect import collect_error_data, send_command, read_status, scan_and_connect
import asyncio
import pygame
import sys 
from src.utils import obter_caminho_arquivo, plote_mapa_polos_zeros, plote_lugar_raizes, plote_bode, plote_nyquist
from src.utils import resposta_em_funcao_de_Kp, resposta_em_funcao_de_Ki, resposta_em_funcao_de_Kd
import requests
import plotly.graph_objects as go

def pendulo_game(m, g, L, b, lim_motor):
    pygame.init()
    screen = pygame.display.set_mode((400, 400))
    pygame.display.set_caption("Pêndulo Invertido - Controle Manual")
    clock = pygame.time.Clock()
    
    # Fontes em tamanhos diferentes para o título e as instruções
    font_title = pygame.font.SysFont(None, 48)
    font_inst = pygame.font.SysFont(None, 36)
    
    # --- Tela inicial (Menu) ---
    running = True
    game_started = False
    
    while running and not game_started:
        screen.fill((255, 255, 255))
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:  # Inicia ao apertar S
                    game_started = True
        
        # Renderizar Título
        title_text = font_title.render("Pêndulo - Pygame", True, (0, 0, 0))
        title_rect = title_text.get_rect(center=(200, 100))
        screen.blit(title_text, title_rect)
        
        # Renderizar Instruções (linhas separadas)
        inst_1 = font_inst.render("A - horário", True, (50, 50, 50))
        inst_2 = font_inst.render("S - start", True, (0, 150, 0)) # S em verde para destacar
        inst_3 = font_inst.render("D - anti-horário", True, (50, 50, 50))
        
        # Centralizando as instruções
        screen.blit(inst_1, inst_1.get_rect(center=(200, 180)))
        screen.blit(inst_2, inst_2.get_rect(center=(200, 220)))
        screen.blit(inst_3, inst_3.get_rect(center=(200, 260)))
        
        pygame.display.flip()
        clock.tick(60)
    
    if not running:
        pygame.quit()
        return 0, [], []
    
    # --- Inicializar ambiente Gym para física ---
    env = gym.make("Pendulum-v1", render_mode=None, g=g)
    env.unwrapped.m = m
    env.unwrapped.l = L
    
    state, _ = env.reset()  # Estado aleatório inicial

    env.unwrapped.state = np.array([np.pi, 0.0]) 
    
    # Atualiza a variável 'state' para refletir a nova observação [cos, sin, dot]
    state = env.unwrapped._get_obs()
    
    erro_history = []
    torque_history = []
    
    frame = 0
    max_frames = 150
    length_pixels = 150  # Comprimento fixo do bastão em pixels
    
    # Configurações da barra de progresso
    bar_x = 20
    bar_y = 20
    bar_max_width = 360
    bar_height = 20
    
    # --- Loop do Jogo ---
    while running and frame < max_frames:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        torque = 0
        
        # Controles atualizados para A e D
        if keys[pygame.K_d]:
            torque = -lim_motor
        elif keys[pygame.K_a]:
            torque = lim_motor
        
        # Aplicar resistência (amortecimento)
        theta_dot = state[2]
        resistencia_torque = -b * theta_dot
        action = [torque + resistencia_torque]
        
        state, reward, terminated, truncated, info = env.step(action)
        
        # O estado do Gymnasium é [cos(theta), sin(theta), theta_dot]
        theta = np.arctan2(state[1], state[0])
        erro = theta  # erro em relação a 0 (posição vertical para cima)
        erro_history.append(erro)
        torque_history.append(torque)
        
        # --- Desenhar na tela ---
        screen.fill((255, 255, 255))
        
        # 1. Desenhar a barra de progresso
        progress_ratio = frame / max_frames
        current_bar_width = int(bar_max_width * progress_ratio)
        
        pygame.draw.rect(screen, (220, 220, 220), (bar_x, bar_y, bar_max_width, bar_height))
        pygame.draw.rect(screen, (0, 0, 255), (bar_x, bar_y, current_bar_width, bar_height))
        pygame.draw.rect(screen, (0, 0, 0), (bar_x, bar_y, bar_max_width, bar_height), 2)
        
        # 2. Desenhar o pêndulo
        center = (200, 200)
        end_x = center[0] + length_pixels * np.sin(theta)
        end_y = center[1] - length_pixels * np.cos(theta) 
        
        # Bastão
        pygame.draw.line(screen, (0, 0, 0), center, (int(end_x), int(end_y)), 10) 
        # Pino central
        pygame.draw.circle(screen, (100, 100, 100), center, 5) 
        
        pygame.display.flip()
        clock.tick(20)
        frame += 1
    
    env.close()
    pygame.quit()
    
    erro_medio = np.mean(np.abs(erro_history)) if erro_history else 0
    return erro_medio, erro_history, torque_history

def bola_bastao_game(m, g, j, R, L, max_ang_alpha):
    pygame.init()
    screen = pygame.display.set_mode((400, 400))
    pygame.display.set_caption("Bola Bastão - Controle Manual")
    clock = pygame.time.Clock()
    
    # Fontes
    font_title = pygame.font.SysFont(None, 48)
    font_inst = pygame.font.SysFont(None, 36)
    
    # --- Tela inicial (Menu) ---
    running = True
    game_started = False
    
    while running and not game_started:
        screen.fill((255, 255, 255))
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:  # Inicia ao apertar S
                    game_started = True
        
        # Renderizar Título
        title_text = font_title.render("Bola Bastão Pygame", True, (0, 0, 0))
        title_rect = title_text.get_rect(center=(200, 100))
        screen.blit(title_text, title_rect)
        
        # Renderizar Instruções
        inst_1 = font_inst.render("A - anti-horário", True, (50, 50, 50))
        inst_2 = font_inst.render("S - start", True, (0, 150, 0))
        inst_3 = font_inst.render("D - horário", True, (50, 50, 50))
        
        screen.blit(inst_1, inst_1.get_rect(center=(200, 180)))
        screen.blit(inst_2, inst_2.get_rect(center=(200, 220)))
        screen.blit(inst_3, inst_3.get_rect(center=(200, 260)))
        
        pygame.display.flip()
        clock.tick(60)
    
    if not running:
        pygame.quit()
        return 0, [], []
    
    # --- Inicializar ambiente ---
    env = BallBeamBalanceEnv(
        timestep=0.05,
        beam_length=L,
        max_angle=max_ang_alpha,
        init_velocity=0.0,
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
    
    # Posição inicial aleatória da bola
    from src.utils import get_ball_start_pos
    random_pos_val = get_ball_start_pos(L, type='Aleatório')
    env.bb.x = random_pos_val
    obs = list(obs)
    obs[1] = random_pos_val
    obs = np.array(obs)
    
    erro_history = []
    torque_history = []
    
    frame = 0
    max_frames = 150
    length_pixels = 300  # Comprimento fixo do bastão em pixels
    ball_radius_pixels = 15 # Tamanho fixo da bola, independente do parâmetro R
    
    # Barra de progresso
    bar_x = 20
    bar_y = 20
    bar_max_width = 360
    bar_height = 20
    
    # --- Loop do Jogo ---
    while running and frame < max_frames:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        torque = 0
        
        if keys[pygame.K_d]:
            torque = -max_ang_alpha
        elif keys[pygame.K_a]:
            torque = max_ang_alpha
        
        torque = np.clip(torque, -max_ang_alpha, max_ang_alpha)
        
        step_result = env.step(torque)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            obs, reward, done, info = step_result
            terminated = done
            truncated = False
        
        # Assumindo que obs[0] é o ângulo do bastão e obs[1] é a posição da bola
        alpha = obs[0] 
        ball_x = obs[1]
        
        erro = ball_x  # erro em relação a 0 (centro do bastão)
        erro_history.append(erro)
        torque_history.append(torque)
        
        # --- Desenhar ---
        screen.fill((255, 255, 255))
        
        # 1. Barra de progresso
        progress_ratio = frame / max_frames
        current_bar_width = int(bar_max_width * progress_ratio)
        pygame.draw.rect(screen, (220, 220, 220), (bar_x, bar_y, bar_max_width, bar_height))
        pygame.draw.rect(screen, (0, 0, 255), (bar_x, bar_y, current_bar_width, bar_height))
        pygame.draw.rect(screen, (0, 0, 0), (bar_x, bar_y, bar_max_width, bar_height), 2)
        
        # 2. Bastão Inclinado
        center_x, center_y = 200, 250
        half_L = length_pixels / 2
        
        # Coordenadas das pontas do bastão usando seno e cosseno
        # O eixo Y no Pygame é invertido (cresce para baixo), por isso os sinais diferem
        beam_start_x = center_x - half_L * np.cos(alpha)
        beam_start_y = center_y + half_L * np.sin(alpha)
        beam_end_x = center_x + half_L * np.cos(alpha)
        beam_end_y = center_y - half_L * np.sin(alpha)
        
        pygame.draw.line(screen, (0, 0, 0), (beam_start_x, beam_start_y), (beam_end_x, beam_end_y), 8)
        
        # 3. Bola rotacionada
        scale = length_pixels / L  # Escala para mapear posição física para pixels
        d_pixels = ball_x * scale  # Distância da bola do centro do bastão em pixels
        
        # Calcula a posição da bola para que ela fique "apoiada" na linha, acompanhando a inclinação
        ball_screen_x = center_x + d_pixels * np.cos(alpha) - ball_radius_pixels * np.sin(alpha)
        ball_screen_y = center_y - d_pixels * np.sin(alpha) - ball_radius_pixels * np.cos(alpha)
        
        pygame.draw.circle(screen, (255, 0, 0), (int(ball_screen_x), int(ball_screen_y)), ball_radius_pixels)
        
        # Pino central do bastão (Opcional, ajuda a ver o eixo de rotação)
        pygame.draw.circle(screen, (100, 100, 100), (center_x, center_y), 5)
        
        pygame.display.flip()
        clock.tick(20)
        frame += 1
        
        if terminated or truncated:
            break
    
    env.close()
    pygame.quit()
    
    erro_medio = np.mean(np.abs(erro_history)) if erro_history else 0
    return erro_medio, erro_history, torque_history

st.set_page_config(
    page_title="ControlHub",
    page_icon="⚙️",
    layout="wide" 
)

sistema = st.sidebar.selectbox(
    "Escolha qual sistema deseja simular:",
    ("Menu Inicial", "Bola bastão", "Pêndulo simples invertido")
)

if sistema == "Menu Inicial":

    st.title("Bem-vindo ao simulador interativo ControlHub! 🚀")
    st.subheader("Aqui você pode explorar a dinâmica e o controle de sistemas clássicos.")

    st.markdown("---")

    st.subheader("Dentre eles, você pode escolher entre:")

    # Criação de colunas para as imagens
    col1, col2 = st.columns([2, 3], gap="large")  

    with col1:
        st.write("🎛️ **Controle:** Equilíbrio de posição")
        st.image("sistema-ball-and-bean.png", caption="Sistema Bola-Bastão (Ball & Beam)")
        

    with col2:
        st.write("⚖️ **Controle:** Estabilização de ângulo")
        st.image("pendulum.png", caption="Pêndulo Simples Invertido")
        

    st.markdown("---")
    
    # Seção Como Usar
    with st.expander("📖 Como usar o sistema (Clique para expandir)"):
        st.markdown("""
        ### Passo a passo para dominar a automação:
        1. **Selecione o sistema** no menu da barra lateral esquerda 👈.
        2. **Baixe o Roteiro:** Baixe o roteiro ao apertar o botão na barra lateral. Cada sistema terá um roteiro.
        3. **Jogue e Simule:** De acordo com o que for pedido no roteiro, navegue entre as perguntas e execute as tarefas descritas 🕹️.

        """)  

if sistema == "Bola bastão":
    
    st.title("Simulação do Sistema Bola e Bastão", text_alignment= 'center')

    with st.expander("Representação do sistema Bola e Bastão"):
        col1, col2, col3 = st.columns([1, 4, 1])
        caminho_imagem = obter_caminho_arquivo("sistema-ball-and-bean.png")
        col2.image(caminho_imagem)

    st.sidebar.write("---")
    caminho_roteiro = obter_caminho_arquivo("Roteiro Bola Bastão.docx")

    with open(caminho_roteiro, "rb") as file:
        btn = st.sidebar.download_button(
            label="Baixar Roteiro",
            icon="🚨",
            data=file,
            file_name="Roteiro Bola Bastão.docx", # Você pode mudar o nome final do arquivo aqui se quiser tirar o "Arrumar -"
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

        with st.expander("Enunciado Questão 1"):
            enunciado_questao1(type="Bola bastão")

        if st.button("Jogar Bola Bastão"):
            erro_medio, erro_history, torque_history = bola_bastao_game(m, g, j, R, L, max_ang_alpha)
            
            fig = go.Figure()
            
            # Adicionando a linha do Erro
            fig.add_trace(go.Scatter(
                y=erro_history, 
                mode='lines', 
                name='Erro (m)',
                line=dict(color='#2ca02c', width=2)
            ))
            
            # Adicionando a linha do Torque (neste caso, é o ângulo de controle alpha)
            fig.add_trace(go.Scatter(
                y=torque_history, 
                mode='lines', 
                name='Torque (rad)',
                line=dict(color='#d62728', width=2)
            ))
            
            # Formatando o layout
            fig.update_layout(
                title=f"Desempenho da Bola Bastão | Erro médio: {erro_medio:.4f} m",
                xaxis_title="Frames",
                yaxis_title="Valores",
                template="plotly_white",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        
    if parte_simulacao == "Questão 2":

        enunciado_questao2(type="Bola bastão")

    if parte_simulacao == "Questão 3":

        enunciado_questao3(type="Bola bastão")

    if parte_simulacao == "Questão 4":

        enunciado_questao4(L=L, d=d, type="Bola bastão")

    if parte_simulacao == "Questão 5":

        enunciado_questao5(type="Bola bastão")

    if parte_simulacao == "Questão 6":

        with st.expander("Enunciado Questão 6"):
            enunciado_questao6(type="Bola bastão")

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
            
            plote_mapa_polos_zeros(m=m, g=g, j=j, R=R, k_MA=q_input, type= "Bola bastão MA")

        if st.sidebar.button("Plote o lugar das raízes"):
            
            plote_lugar_raizes(m=m, g=g, j=j, R=R, k_MA=q_input, type= "Bola bastão MA")

        if st.sidebar.button("Plote o diagrama de Bode"):
            
            plote_bode(m=m, g=g, j=j, R=R, k_MA=q_input, type= "Bola bastão MA")

        if st.sidebar.button("Plote o diagrama de Nyquist"):
            
            plote_nyquist(m=m, g=g, j=j, R=R, k_MA=q_input, type="Bola bastão MA")
    
    if parte_simulacao == "Questão 7":

        with st.expander("Enunciado Questão 7"):
            enunciado_questao7(type = "Bola bastão")


        st.sidebar.header("Inputs da Simulação")
        K_feedback = st.sidebar.slider("Valor do ganho de feedback", min_value=0.0, max_value=5.0, value=0.0, step=0.01, help=f"A entrada será -K_feedback * erro, onde erro é a diferença entre a posição desejada (0) e a posição atual da bola. K_feedback é o ganho que amplifica essa diferença para gerar a ação de controle. O menos se dá pois se o erro é negativo (bola a direita), a inclinação deve ser positiva e vice-versa.")
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

                action = -1 * action #aqui foi aplicado a inversão pois se o erro é negativo (bola a direita), a inclinação deve ser positiva e vice-versa, e a conta de K_feedback * erro não leva isso em consideração, então o sinal é invertido para corrigir isso.

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

        if st.sidebar.button("Plote resposta em MF"):
            
            plote_resposta_MF_Bola_Bastao(m, g, j, R, K_feedback)

        if st.sidebar.button("Plote resposta no tempo em função de K_feedback"):

            plote_resposta_no_tempo(m=m, g=g, j=j, R=R, k_feedback=K_feedback, type="Bola bastão MF")
        
        if st.sidebar.button("Plote mapa de polos e zeros"):

            plote_mapa_polos_zeros(m=m, g=g, j=j, R=R, k_feedback=K_feedback, type="Bola bastão MF")

        if st.sidebar.button("Plote o lugar das raízes"):
            
            plote_lugar_raizes(m=m, g=g, j=j, R=R, k_feedback=K_feedback, type="Bola bastão MF")

        if st.sidebar.button("Plote o diagrama de Bode"):
            
            plote_bode(m=m, g=g, j=j, R=R, k_feedback=K_feedback, type="Bola bastão MF")

        if st.sidebar.button("Plote o diagrama de Nyquist"):
            
            plote_nyquist(m=m, g=g, j=j, R=R, k_feedback=K_feedback, type="Bola bastão MF")

    if parte_simulacao == "Questão 8":

        with st.expander("Enunciado Questão 8"):
            enunciado_questao8(type="Bola bastão")


        st.sidebar.header("Inputs da Simulação")
        Kp = st.sidebar.slider("Escolha um valor de Kp", min_value = 0.0, max_value = 5.0, value = 0.0, step = 0.1, help="Define a força de retorno imediata. Como Erro = -Posição, se a bola está na direita (Pos > 0), o erro é negativo. O controle aplica -Kp * Erro para gerar uma inclinação positiva e empurrar a bola de volta ao centro.")
        Ki = st.sidebar.slider("Escolha um valor de Ki", min_value = 0.0, max_value = 5.0, value = 0.0, step = 0.1, help="Elimina o erro residual acumulado no tempo. Se a bola permanece à direita, o erro integral acumula valores negativos. A ação -Ki * Integral força uma inclinação positiva crescente para garantir que a bola chegue exatamente ao zero.")
        Kd = st.sidebar.slider("Escolha um valor de Kd", min_value = 0.0, max_value = 5.0, value = 0.0, step = 0.1, help="Atua como um freio para evitar oscilações. Ele observa a velocidade da bola: se ela volta rápido para o centro, a derivada do erro é positiva. A ação -Kd * Derivada aplica uma inclinação oposta para amortecer o movimento e evitar o overshoot.")

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

                erro_integral += erro * 0.01 #colocando uma lógica de dt para o erro integral não subir muito rápido
                erro_integral = np.clip(erro_integral, -10.0, 10.0)

                action = -Kp * erro + Kd * velocidade_bola - Ki * erro_integral #necessário inverter o sinal de Kp, Ki e Kd para que o controle atue na direção correta, pois o erro é calculado como 0 - posição da bola, então se a bola está à direita (posição positiva), o erro é negativo, e para empurrar a bola de volta ao centro, a inclinação deve ser positiva. O mesmo raciocínio se aplica para a velocidade da bola e o erro integral.

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

        if st.sidebar.button("Plote resposta em função de Kp"):
            
            resposta_em_funcao_de_Kp(m, g, j, R)

        if st.sidebar.button("Plote resposta em função de Ki"):
            
            resposta_em_funcao_de_Ki(m, g, j, R)

        if st.sidebar.button("Plote resposta em função de Kd"):
            
            resposta_em_funcao_de_Kd(m, g, j, R)
        
        if st.sidebar.button("Plote resposta PID"):
            
            plote_resposta_PID_Bola_Bastao(m, g, j, R, Kp, Ki, Kd)

        if st.sidebar.button("Plote resposta no tempo em função de Kp, Ki e Kd"):

            plote_resposta_no_tempo(m=m, g=g, j=j, R=R, type= "Bola bastão PID", Kp=Kp, Ki=Ki, Kd=Kd)


        if st.sidebar.button("Plote mapa de polos e zeros"):
            
            plote_mapa_polos_zeros(m=m, g=g, j=j, R=R, type= "Bola bastão PID", Kp=Kp, Ki=Ki, Kd=Kd)

        if st.sidebar.button("Plote o lugar das raízes"):
            
            plote_lugar_raizes(m=m, g=g, j=j, R=R, type= "Bola bastão PID", Kp=Kp, Ki=Ki, Kd=Kd)

        if st.sidebar.button("Plote o diagrama de Bode"):

            plote_bode(m=m, g=g, j=j, R=R, type= "Bola bastão PID", Kp=Kp, Ki=Ki, Kd=Kd)

        if st.sidebar.button("Plote o diagrama de Nyquist"):

            plote_nyquist(m=m, g=g, j=j, R=R, type="Bola bastão PID", Kp=Kp, Ki=Ki, Kd=Kd)
       
    if parte_simulacao == "Questão 9":
        enunciado_questao9(type="Bola bastão")
        
        if st.button("Iniciar", type="primary"):
        
            # Congela a interface enquanto a ESP32 trabalha
            with st.spinner("Aguarde... A ESP32 está coletando dados e calculando o erro..."):
                
                # Chama a função nova
                sucesso, mensagem_ou_dado = asyncio.run(collect_error_data("Gamificacao Bola Bastao"))
                
                if sucesso:
                    st.success("Coleta finalizada com sucesso!")
                    
                    try:
                        # Converte o texto recebido para número decimal
                        erro_total_num = float(mensagem_ou_dado)
                        
                        st.divider() # Linha divisória para dar um visual legal
                        
                        # Exibe o número de forma BEM GRANDE e centralizada (usando HTML seguro no Streamlit)
                        st.markdown("<h3 style='text-align: center; color: gray;'>Erro Total Acumulado</h3>", unsafe_allow_html=True)
                        st.markdown(
                            f"<h1 style='text-align: center; color: #FF4B4B; font-size: 85px; margin-top: -20px;'>{erro_total_num:.2f}</h1>", 
                            unsafe_allow_html=True
                        )
                        
                        st.divider()
                        
                    except ValueError:
                        st.error("Erro ao processar os dados. O valor recebido não é um número válido.")
                        st.write(f"Dado bruto: {mensagem_ou_dado}")
                else:
                    st.error(mensagem_ou_dado)

    if parte_simulacao == "Questão 10":
        enunciado_questao10(type="Bola bastão")

        Kp = st.slider("Escolha um valor de Kp para o controle PID", min_value = 0.0, max_value = 5.0, value = 1.0, step = 0.1)
        Ki = st.slider("Escolha um valor de Ki para o controle PID", min_value = 0.0, max_value = 5.0, value = 1.0, step = 0.1)
        Kd = st.slider("Escolha um valor de Kd para o controle PID", min_value = 0.0, max_value = 5.0, value = 1.0, step = 0.1)

        if st.button("Iniciar", type="primary"):
        
            # Congela a interface enquanto a ESP32 trabalha
            with st.spinner("Aguarde... A ESP32 está coletando dados e calculando o erro..."):
                
                # Chama a função nova
                sucesso, mensagem_ou_dado = asyncio.run(collect_error_data(f"PID Bola Bastao (Kp={Kp}, Ki={Ki}, Kd={Kd})"))
                
                if sucesso:
                    st.success("Coleta finalizada com sucesso!")
                    
                    try:
                        # Converte o texto recebido para número decimal
                        erro_total_num = float(mensagem_ou_dado)
                        
                        st.divider() # Linha divisória para dar um visual legal
                        
                        # Exibe o número de forma BEM GRANDE e centralizada (usando HTML seguro no Streamlit)
                        st.markdown("<h3 style='text-align: center; color: gray;'>Erro Total Acumulado</h3>", unsafe_allow_html=True)
                        st.markdown(
                            f"<h1 style='text-align: center; color: #FF4B4B; font-size: 85px; margin-top: -20px;'>{erro_total_num:.2f}</h1>", 
                            unsafe_allow_html=True
                        )
                        
                        st.divider()
                        
                    except ValueError:
                        st.error("Erro ao processar os dados. O valor recebido não é um número válido.")
                        st.write(f"Dado bruto: {mensagem_ou_dado}")
                else:
                    st.error(mensagem_ou_dado)


if sistema == "Pêndulo simples invertido":
    st.title("Simulação do Sistema Pêndulo Simples Invertido")

    with st.expander("Representação do sistema Pêndulo Simples Invertido"):
        col1, col2, col3 = st.columns([2, 1, 2])
        caminho_imagem = obter_caminho_arquivo("pendulum.png")
        col2.image(caminho_imagem)

    st.sidebar.write("---")
    caminho_roteiro = obter_caminho_arquivo("Roteiro Pêndulo Simples Invertido.docx")

    with open(caminho_roteiro, "rb") as file:
        btn = st.sidebar.download_button(
            label="Baixar Roteiro",
            icon="🚨",
            data=file,
            file_name="Roteiro Pêndulo Simples Invertido.docx", # Você pode mudar o nome final do arquivo aqui se quiser tirar o "Arrumar -"
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
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
        with st.expander("Enunciado Questão 1"):
            enunciado_questao1(type="Pêndulo simples invertido")

        if st.button("Jogar Pêndulo"):
            erro_medio, erro_history, torque_history = pendulo_game(m, g, L, b, lim_motor)
            
            fig = go.Figure()
            
            # Adicionando a linha do Erro
            fig.add_trace(go.Scatter(
                y=erro_history, 
                mode='lines', 
                name='Erro (rad)',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Adicionando a linha do Torque
            fig.add_trace(go.Scatter(
                y=torque_history, 
                mode='lines', 
                name='Torque (Nm)',
                line=dict(color='#ff7f0e', width=2)
            ))
            
            # Formatando o layout
            fig.update_layout(
                title=f"Desempenho do Pêndulo | Erro médio: {erro_medio:.4f} rad",
                xaxis_title="Frames",
                yaxis_title="Valores",
                template="plotly_white",
                hovermode="x unified" # Cria uma linha interativa ao passar o mouse
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    if parte_simulacao == "Questão 2":
        enunciado_questao2(type="Pêndulo simples invertido")

    if parte_simulacao == "Questão 3":
        enunciado_questao3(type="Pêndulo simples invertido")

    if parte_simulacao == "Questão 4":
        enunciado_questao4(type="Pêndulo simples invertido")

    if parte_simulacao == "Questão 5":
        enunciado_questao5(type="Pêndulo simples invertido")

    if parte_simulacao == "Questão 6":

        with st.expander("Enunciado Questão 6"):
            enunciado_questao6(type="Pêndulo simples invertido")

        # --- Inputs do Usuário ---
        st.sidebar.header("Inputs da Simulação")
        swing_up_true = st.sidebar.checkbox("Controle com Swing-up?", value=True, help="Defina se a ação vai ser dividida para duas condições ou não, conforme descrito no enunciado.")
        q_input = st.sidebar.slider("Valor da entrada degrau", min_value=-lim_motor, max_value=lim_motor, value=0.0, step=0.005, help=f"Valor de entrada é o torque do motor indo de -{lim_motor} até {lim_motor} Nm")

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
                    st.info(f"Simulação encerrada no passo {i}.")
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

        if st.sidebar.button("Plote resposta em MA"):
            
            plote_resposta_MA_Pendulo_simples_invertido(m, g, L, b, q_input)
        
        if st.sidebar.button("Plote mapa de polos e zeros"):
            
            plote_mapa_polos_zeros(m=m, g=g, L=L, b=b, k_MA=q_input, type= "Pêndulo simples invertido MA")

        if st.sidebar.button("Plote o lugar das raízes"):
            
            plote_lugar_raizes(m=m, g=g, L=L, b=b, k_MA=q_input, type= "Pêndulo simples invertido MA")

        if st.sidebar.button("Plote o diagrama de Bode"):
            
            plote_bode(m=m, g=g, L=L, b=b, k_MA=q_input, type= "Pêndulo simples invertido MA")

        if st.sidebar.button("Plote o diagrama de Nyquist"):
            
            plote_nyquist(m=m, g=g, L=L, b=b, k_MA=q_input, type= "Pêndulo simples invertido MA")

    if parte_simulacao == "Questão 7":

        with st.expander("Enunciado Questão 7"):
            enunciado_questao7(type="Pêndulo simples invertido")

        # --- Inputs do Usuário ---
        st.sidebar.header("Inputs da Simulação")
        swing_up_true = st.sidebar.checkbox("Controle com Swing-up?", value=True, help="Defina se a ação vai ser dividida para duas condições ou não, conforme descrito no enunciado da questão 5.")
        K_feedback = st.sidebar.slider("Valor do ganho de feedback", min_value=0.0, max_value=lim_motor, value=0.0, step=0.005, help=f"Valor de entrada é o ganho de feedback * erro (erro = 0 - theta) indo de -{lim_motor} até {lim_motor} Nm")

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
                    if abs(valor_graus) <= 20:
                        erro = (0 - theta)
                        erro_history.append(erro)

                        torque = K_feedback * erro
                        torque = np.clip(torque, -lim_motor, lim_motor)
                        external_action_history.append(torque)
                        resistencia_torque = -b * theta_dot
                        action = [torque + resistencia_torque]
                        control_type_history.append(1) # Identificador para o gráfico
                    else:
                        # Entrada constante
                        erro = (0 - theta)
                        erro_history.append(erro)

                        torque = lim_motor
                        external_action_history.append(torque)
                        resistencia_torque = -b * theta_dot
                        action = [torque + resistencia_torque]
                        control_type_history.append(0) # Identificador para o gráfico
                    
                state, reward, terminated, truncated, info = env.step(action)

                theta_2dot = 3*action[0]/((m*L**2)) + 3*gravidade*np.sin(theta)/(2*L)

                theta_double_dot_history.append(theta_2dot)

                if terminated or truncated:
                    st.info(f"Simulação encerrada no passo {i}.")
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

        if st.sidebar.button("Plote resposta em MF"):
            
            plote_resposta_MF_Pendulo_simples_invertido(m, g, L, b, K_feedback)

        if st.sidebar.button("Plote resposta no tempo em função de K_feedback"):

            plote_resposta_no_tempo(type= "Pêndulo simples invertido MF",m=m, g=g, L=L, b=b, k_feedback=K_feedback,)
        
        if st.sidebar.button("Plote mapa de polos e zeros"):
            
            plote_mapa_polos_zeros(m=m, g=g, L=L, b=b, k_feedback=K_feedback, type= "Pêndulo simples invertido MF")

        if st.sidebar.button("Plote o lugar das raízes"):
            
            plote_lugar_raizes(m=m, g=g, L=L, b=b, k_feedback=K_feedback, type= "Pêndulo simples invertido MF")

        if st.sidebar.button("Plote o diagrama de Bode"):
            
            plote_bode(m=m, g=g, L=L, b=b, k_feedback=K_feedback, type= "Pêndulo simples invertido MF")

        if st.sidebar.button("Plote o diagrama de Nyquist"):
            
            plote_nyquist(m=m, g=g, L=L, b=b, k_feedback=K_feedback, type= "Pêndulo simples invertido MF")

    if parte_simulacao == "Questão 8":
        st.warning("Questão em Avaliação para trabalhos futuros.")
        enunciado_questao8(type="Pêndulo simples invertido")

        # --- Inputs do Usuário ---
        st.sidebar.header("Inputs da Simulação")
        swing_up_true = st.sidebar.checkbox("Controle com Swing-up?", value=True, help="Defina se a ação vai ser dividida para duas condições ou não, conforme descrito no enunciado.")
        st.sidebar.header("Inputs da Simulação")
        Kp = st.sidebar.slider("Escolha um valor de Kp", min_value = 0.0, max_value = lim_motor, value = 0.0, step = 0.01)
        Ki = st.sidebar.slider("Escolha um valor de Ki", min_value = 0.0, max_value = lim_motor, value = 0.0, step = 0.01)
        Kd = st.sidebar.slider("Escolha um valor de Kd", min_value = 0.0, max_value = lim_motor, value = 0.0, step = 0.01)

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

                        dt_sim = env.unwrapped.dt 

                        erro_integral += erro * dt_sim 
                        erro_integral = np.clip(erro_integral, -10.0, 10.0)

                        erro_derivativo = (0 - theta_dot)

                        torque = (Kp * erro) + (Kd * erro_derivativo) + (Ki * erro_integral)
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
                    # Lógica de Transição
                    if abs(valor_graus) <= 20:

                        # FASE DE CATCH - Feedback
                        erro = (0 - theta)
                        erro_history.append(erro)

                        dt_sim = env.unwrapped.dt 

                        erro_integral += erro * dt_sim 
                        erro_integral = np.clip(erro_integral, -10.0, 10.0)

                        erro_derivativo = (0 - theta_dot)

                        torque = (Kp * erro) + (Kd * erro_derivativo) + (Ki * erro_integral)
                        torque = np.clip(torque, -lim_motor, lim_motor)
                        external_action_history.append(torque)
                        resistencia_torque = -b * theta_dot
                        action = [torque + resistencia_torque]
                        control_type_history.append(1) # Identificador para o gráfico
                    
                    else:
                        # Entrada constante
                        erro = (0 - theta)
                        erro_history.append(erro)

                        erro_integral += erro
                        erro_integral = np.clip(erro_integral, -10.0, 10.0) #Não permite que a integral cresça infinitamente

                        torque = lim_motor
                        external_action_history.append(torque)
                        resistencia_torque = -b * theta_dot
                        action = [torque + resistencia_torque]
                        control_type_history.append(0) # Identificador para o gráfico

                    
                state, reward, terminated, truncated, info = env.step(action)

                theta_2dot = 3*action[0]/((m*L**2)) + 3*gravidade*np.sin(theta)/(2*L)

                theta_double_dot_history.append(theta_2dot)

                if terminated or truncated:
                    st.info(f"Simulação encerrada no passo {i}.")
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

        if st.sidebar.button("Plote resposta em função de Kp"):
            
            resposta_pendulo_em_funcao_de_Kp(m, L, b, g)

        if st.sidebar.button("Plote resposta em função de Ki"):
            
            resposta_pendulo_em_funcao_de_Ki(m, L, b, g)

        if st.sidebar.button("Plote resposta em função de Kd"):
            
            resposta_pendulo_em_funcao_de_Kd(m, L, b, g)

        if st.sidebar.button("Plote resposta no tempo em função de Kp, Ki e Kd"):

            plote_resposta_no_tempo(m=m, g=g, L=L, b=b, Kp=Kp, Ki=Ki, Kd=Kd, type= "Pêndulo simples invertido PID")
        
        if st.sidebar.button("Plote mapa de polos e zeros"):
            
            plote_mapa_polos_zeros(m=m, g=g, L=L, b=b, Kp=Kp, Ki=Ki, Kd=Kd, type= "Pêndulo simples invertido PID")

        if st.sidebar.button("Plote o lugar das raízes"):
            
            plote_lugar_raizes(m=m, g=g, L=L, b=b, Kp=Kp, Ki=Ki, Kd=Kd, type= "Pêndulo simples invertido PID")

    if parte_simulacao == "Questão 9":
        enunciado_questao9(type="Pêndulo simples invertido")

        if st.button("Iniciar", type="primary"):
        
            # Congela a interface enquanto a ESP32 trabalha
            with st.spinner("Aguarde... A ESP32 está coletando dados e calculando o erro..."):
                
                # Chama a função nova
                sucesso, mensagem_ou_dado = asyncio.run(collect_error_data("Gamificacao Pendulo Invertido"))
                
                if sucesso:
                    st.success("Coleta finalizada com sucesso!")
                    
                    try:
                        # Converte o texto recebido para número decimal
                        erro_total_num = float(mensagem_ou_dado)
                        
                        st.divider() # Linha divisória para dar um visual legal
                        
                        # Exibe o número de forma BEM GRANDE e centralizada (usando HTML seguro no Streamlit)
                        st.markdown("<h3 style='text-align: center; color: gray;'>Erro Total Acumulado</h3>", unsafe_allow_html=True)
                        st.markdown(
                            f"<h1 style='text-align: center; color: #FF4B4B; font-size: 85px; margin-top: -20px;'>{erro_total_num:.2f}</h1>", 
                            unsafe_allow_html=True
                        )
                        
                        st.divider()
                        
                    except ValueError:
                        st.error("Erro ao processar os dados. O valor recebido não é um número válido.")
                        st.write(f"Dado bruto: {mensagem_ou_dado}")
                else:
                    st.error(mensagem_ou_dado)
    


    if parte_simulacao == "Questão 10":
        enunciado_questao10(type="Pêndulo simples invertido")
        Kf = st.slider("Escolha um valor de Kf para o controle por lqr", min_value = 0.0, max_value = 0.10, value = 0.00, step = 0.001)

        if st.button("Iniciar", type="primary"):
        
            # Congela a interface enquanto a ESP32 trabalha
            with st.spinner("Aguarde... A ESP32 está coletando dados e calculando o erro..."):
                
                # Chama a função nova
                sucesso, mensagem_ou_dado = asyncio.run(collect_error_data(f"Feedback Pendulo Invertido (Kf={Kf})"))
                
                if sucesso:
                    st.success("Coleta finalizada com sucesso!")
                    
                    try:
                        # Converte o texto recebido para número decimal
                        erro_total_num = float(mensagem_ou_dado)
                        
                        st.divider() # Linha divisória para dar um visual legal
                        
                        # Exibe o número de forma BEM GRANDE e centralizada (usando HTML seguro no Streamlit)
                        st.markdown("<h3 style='text-align: center; color: gray;'>Erro Total Acumulado</h3>", unsafe_allow_html=True)
                        st.markdown(
                            f"<h1 style='text-align: center; color: #FF4B4B; font-size: 85px; margin-top: -20px;'>{erro_total_num:.2f}</h1>", 
                            unsafe_allow_html=True
                        )
                        
                        st.divider()
                        
                    except ValueError:
                        st.error("Erro ao processar os dados. O valor recebido não é um número válido.")
                        st.write(f"Dado bruto: {mensagem_ou_dado}")
                else:
                    st.error(mensagem_ou_dado)
