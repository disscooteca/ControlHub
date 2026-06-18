import streamlit as st
import asyncio
from bleak import BleakScanner, BleakClient

# Os mesmos UUIDs e Nome usados no código da ESP32
DEVICE_NAME = "ESP32_BLE_Control"
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
CHAR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

# --- FUNÇÕES ASSÍNCRONAS PARA O BLE ---

async def scan_and_connect():
    """Procura a ESP32 pelo nome e retorna o endereço MAC dela."""
    st.info("Procurando ESP32...")
    devices = await BleakScanner.discover(timeout=3.0)
    for d in devices:
        if d.name == DEVICE_NAME:
            return d
    return None

async def send_command(command: str):
    """Conecta, envia o texto e desconecta."""
    device = await scan_and_connect()
    if device:
        try:
            async with BleakClient(device) as client:
                await client.write_gatt_char(CHAR_UUID, command.encode('utf-8'), response=True)
                return True, f"Comando '{command}' enviado com sucesso!"
        except Exception as e:
            return False, f"Erro ao enviar: {e}"
    else:
        return False, "ESP32 não encontrada. Ela está ligada?"

async def read_status():
    """Conecta, lê o texto atual da ESP32 e desconecta."""
    device = await scan_and_connect()
    if device:
        try:
            async with BleakClient(device) as client:
                valor_bytes = await client.read_gatt_char(CHAR_UUID)
                return True, valor_bytes.decode('utf-8')
        except Exception as e:
            return False, f"Erro ao ler: {e}"
    else:
        return False, "ESP32 não encontrada."
    
async def collect_error_data(command: str):
    """Envia o comando, aguarda os 20s da ESP32 e retorna o Erro Total acumulado."""
    device = await scan_and_connect()
    if not device:
        return False, "ESP32 não encontrada."

    erro_recebido = ""
    coleta_finalizada = asyncio.Event()

    # Como a ESP32 agora manda apenas 1 notificação com o valor do erro,
    # pegamos esse valor e avisamos o código para parar de esperar.
    def notification_handler(sender, data):
        nonlocal erro_recebido
        erro_recebido = data.decode('utf-8')
        coleta_finalizada.set() # Destrava o wait_for

    try:
        async with BleakClient(device) as client:
            # 1. Ativa a escuta
            await client.start_notify(CHAR_UUID, notification_handler)

            # 2. Envia o comando para iniciar a rotina na placa
            await client.write_gatt_char(CHAR_UUID, command.encode('utf-8'), response=True)

            # 3. Fica esperando a placa fazer a matemática por 20s e devolver a notificação
            # Mantemos os 30s de timeout para dar margem de segurança
            await asyncio.wait_for(coleta_finalizada.wait(), timeout=40.0)

            # 4. Desativa a escuta
            await client.stop_notify(CHAR_UUID)

            return True, erro_recebido
            
    except asyncio.TimeoutError:
        return False, "A ESP32 demorou demais para responder ou o tempo acabou."
    except Exception as e:
        return False, f"Erro na conexão BLE: {e}"