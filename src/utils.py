import streamlit as st
import plotly.graph_objects as go
import numpy as np

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