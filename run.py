import sys
import os
from streamlit.web import cli as stcli
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ballbeam_gym.envs.balance import BallBeamBalanceEnv 
import gymnasium as gym
from ballbeam_gym.envs.balance import BallBeamBalanceEnv
import plotly.graph_objects as go
import requests
from control import (TransferFunction)
from matplotlib.lines import Line2D
import pygame

def resolve_path(path):
    # Esta função garante que o arquivo seja encontrado tanto rodando
    # como script quanto rodando como .exe (congelado)
    if getattr(sys, "frozen", False):
        basedir = sys._MEIPASS
    else:
        basedir = os.path.dirname(__file__)
    return os.path.join(basedir, path)

if __name__ == "__main__":
    # Aponta para o seu arquivo principal
    app_path = resolve_path("main.py")
    
    # Simula o comando "streamlit run main.py" via código
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--global.developmentMode=false",
    ]
    
    sys.exit(stcli.main())