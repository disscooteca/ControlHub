#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>
#include <BLE2902.h>
#include <ESP32Servo.h>
#include <HCSR04.h>

//Motor JGA (pêndulo)
#define velmotor 4
#define mla 18
#define mlb 19

const int pinoEncoderA = 22; 
const int pinoEncoderB = 21;

volatile long posicao_motor = 0;

const float PULSOS_POR_VOLTA = 1320.0;

//Sensor HCSR04
#define trig 27
#define echo 33

//Servo (BB)
#define pin_servo 13

//Console
#define LED_PIN 23
#define chave_console 17

const int pinoPotenciometro = 34;

//BLE
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"


float kf, desejado, dt, erro_total;
float val, j, t, desejado, erro, u, kp, ki, kd, soma, ultimo, x;
float leitura, limitex;

//Interrupção para leitura do encoder (pêndulo)
void IRAM_ATTR lerEncoder() {
  // Lógica de quadratura rápida usando leitura dos dois pinos
  if (digitalRead(pinoEncoderA) == digitalRead(pinoEncoderB)) {
    posicao_motor--;
  } else {
    posicao_motor++;
  }
}

String comandoAtual = "";

bool coletaConcluida = false;
const int maxLeituras = 666; 

UltraSonicDistanceSensor distanceSensor(trig, echo);

Servo myservo;

BLECharacteristic *pCharacteristic;

class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      Serial.println("PC Conectado!");
    };

    void onDisconnect(BLEServer* pServer) {
      Serial.println("PC Desconectado! Reiniciando anúncio (Advertising)...");
      BLEDevice::startAdvertising(); 
      comandoAtual = "";
    }
};

class MyCallbacks: public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *pCharacteristic) {
    String value = pCharacteristic->getValue().c_str();

    if (value.length() > 0) {
      Serial.println("*********");
      Serial.print("Novo comando do PC: ");
      Serial.println(value);
      Serial.println("*********");

      if (value.startsWith("Feedback Pendulo Invertido")) {
        comandoAtual = "Feedback Pendulo Invertido";
        coletaConcluida = false; 

        int posKf = value.indexOf("Kf=");

        if (posKf != -1){
          String strKf = value.substring(posKf + 3, value.indexOf(")", posKf));

          kf = strKf.toFloat();
        }        
      }

      if (value.startsWith("Gamificacao Pendulo Invertido")) {
        comandoAtual = "Gamificacao Pendulo Invertido";
        coletaConcluida = false; 
      }

      if (value.startsWith("Gamificacao Bola Bastao")) {
        comandoAtual = "Gamificacao Bola Bastao";
        coletaConcluida = false; 
      }

      if (value.startsWith("PID Bola Bastao")) {
        comandoAtual = "PID Bola Bastao"; // Mantém a string limpa para o void loop()
        coletaConcluida = false; 

        // 2. Encontra a posição onde começa cada parâmetro
        int posKp = value.indexOf("Kp=");
        int posKi = value.indexOf("Ki=");
        int posKd = value.indexOf("Kd=");

        // 3. Se encontrou todos os marcadores, extrai e converte para float
        if (posKp != -1 && posKi != -1 && posKd != -1) {
          
          // Recorta do "Kp=" até a próxima vírgula
          String strKp = value.substring(posKp + 3, value.indexOf(",", posKp));
          // Recorta do "Ki=" até a próxima vírgula
          String strKi = value.substring(posKi + 3, value.indexOf(",", posKi));
          // Recorta do "Kd=" até o parêntese de fechamento ')'
          String strKd = value.substring(posKd + 3, value.indexOf(")", posKd));

          // Converte os textos recortados em números decimais (float)
          kp = strKp.toFloat();
          ki = strKi.toFloat();
          kd = strKd.toFloat();

          // Mostra no Serial os novos valores aplicando o controle
          Serial.print("-> PID Atualizado com Sucesso! ");
          Serial.printf("Kp: %.2f | Ki: %.2f | Kd: %.2f\n", kp, ki, kd);
        }
      }
    }
  }
};

void gamePenduloInvertido() {

  if (digitalRead(chave_console) == LOW){
    digitalWrite(mla, LOW);
    digitalWrite(mlb, LOW);
    digitalWrite(LED_PIN, LOW);
    analogWrite(velmotor, 0);
    delay(500);
    digitalWrite(LED_PIN, LOW);
    delay(500);
    digitalWrite(LED_PIN, HIGH);

    coletaConcluida = false;
  }

  if (digitalRead(chave_console) == HIGH && !coletaConcluida) {

    delay(1000);

    digitalWrite(LED_PIN, HIGH);
    Serial.println("Iniciando coleta ...");

    erro_total = 0;

    for (int i = 0; i < maxLeituras; i++) {

      int valorBruto = analogRead(pinoPotenciometro);
      int valorMapeado = map(valorBruto, 0, 4095, -250, 250);

      // --- LÓGICA DE POSIÇÃO E ERRO ---
      
      long pulsosAtuais = posicao_motor; 

      // 2. Converte os pulsos brutos para Graus
      float anguloAtual = (pulsosAtuais * 360.0) / PULSOS_POR_VOLTA;

      // 3. NORMALIZAÇÃO: Garante que o ângulo fique sempre entre 0° e 360°
      // (Mesmo que o pêndulo dê 10 voltas, ele entenderá a posição real de 0 a 360)
      anguloAtual = fmod(anguloAtual, 360.0);
      if (anguloAtual < 0) {
        anguloAtual += 360.0;
      }

      // 4. Define o alvo. Como ele inicia em 0° (para baixo), o "para cima" é exatamente 180°
      float anguloDesejado = 180.0; 

      // 5. Cálculo exato do Erro: Desejado - Atual
      float erro = anguloDesejado - anguloAtual;

      erro_total += erro;

      // --- PRINTS DE MONITORAMENTO ---
      Serial.print("Pulsos Brutos: "); Serial.print(pulsosAtuais);
      Serial.print(" | Angulo Atual: "); Serial.print(anguloAtual);
      Serial.print(" | Erro: "); Serial.println(erro);

      // --- SUA LÓGICA ORIGINAL DO MOTOR ---
      if (valorMapeado >= 0){
        digitalWrite(mla, HIGH);
        digitalWrite(mlb, LOW);
        digitalWrite(LED_PIN, HIGH);
        analogWrite(velmotor, valorMapeado);
      }
      else{
        digitalWrite(mla, LOW);
        digitalWrite(mlb, HIGH);
        digitalWrite(LED_PIN, HIGH);
        analogWrite(velmotor, -valorMapeado);
      }

      delay(dt*1000);    
    }

    analogWrite(velmotor, 0);

    Serial.println("Coleta finalizada! Enviando via Bluetooth...");

    String payload = String(erro_total, 2);

    pCharacteristic->setValue(payload.c_str());
    pCharacteristic->notify();
    
    Serial.print("Erro total enviado: ");
    Serial.println(payload);

    coletaConcluida = true;
  }
}

void feedbackPenduloInvertido() {

  desejado = 180.0;

  if (digitalRead(chave_console) == LOW){
    digitalWrite(mla, LOW);
    digitalWrite(mlb, LOW);
    digitalWrite(LED_PIN, LOW);
    analogWrite(velmotor, 0);

    coletaConcluida = false;
  }

  if (digitalRead(chave_console) == HIGH && !coletaConcluida) {

    delay(1000);

    digitalWrite(LED_PIN, HIGH);
    Serial.println("Iniciando coleta ...");

    erro_total = 0;

    for (int i = 0; i < maxLeituras; i++) {

      // --- LÓGICA DE POSIÇÃO E ERRO ---
      
      long pulsosAtuais = posicao_motor; 

      // 2. Converte os pulsos brutos para Graus
      float anguloAtual = (pulsosAtuais * 360.0) / PULSOS_POR_VOLTA;

      // 3. NORMALIZAÇÃO: Garante que o ângulo fique sempre entre 0° e 360°
      // (Mesmo que o pêndulo dê 10 voltas, ele entenderá a posição real de 0 a 360)
      anguloAtual = fmod(anguloAtual, 360.0);
      if (anguloAtual < 0) {
        anguloAtual += 360.0;
      }

      // 4. Define o alvo. Como ele inicia em 0° (para baixo), o "para cima" é exatamente 180°
      float anguloDesejado = 180.0; 

      // 5. Cálculo exato do Erro: Desejado - Atual
      float erro = anguloDesejado - anguloAtual;

      erro_total += erro;

      float acao = kf * erro;

      if (acao >= 250){
        acao = 250;
      }

      else if (acao <= -250){
        acao = -250;
      }

      // --- PRINTS DE MONITORAMENTO ---
      Serial.print("Pulsos Brutos: "); Serial.print(pulsosAtuais);
      Serial.print(" | Angulo Atual: "); Serial.print(anguloAtual);
      Serial.print(" | Erro: "); Serial.println(erro);

      // --- SUA LÓGICA ORIGINAL DO MOTOR ---
      if (acao >= 0){
        digitalWrite(mla, HIGH);
        digitalWrite(mlb, LOW);
        digitalWrite(LED_PIN, HIGH);
        analogWrite(velmotor, acao);
      }
      else{
        digitalWrite(mla, LOW);
        digitalWrite(mlb, HIGH);
        digitalWrite(LED_PIN, HIGH);
        analogWrite(velmotor, -acao);
      }

      delay(dt*1000);    
    }

    analogWrite(velmotor, 0);

    Serial.println("Coleta finalizada! Enviando via Bluetooth...");

    String payload = String(erro_total, 2);

    pCharacteristic->setValue(payload.c_str());
    pCharacteristic->notify();
    
    Serial.print("Erro total enviado: ");
    Serial.println(payload);

    coletaConcluida = true;
  }
}

void gameBolaBastao() {

  desejado = 25;
  erro_total = 0;
  leitura = 1;

  myservo.write(0);
  digitalWrite(LED_PIN, LOW);
  
  if (digitalRead(chave_console) == LOW) {
    
    coletaConcluida = false; // Prepara para a próxima vez que a chave subir
  }

  // Se a chave for HIGH e ainda não fizemos a coleta desta vez
  if (digitalRead(chave_console) == HIGH && !coletaConcluida) {

    delay(1000);

    digitalWrite(LED_PIN, HIGH);
    Serial.println("Iniciando coleta ...");

    erro_total = 0.0;

    // 1. FASE DE COLETA (Dura exatos 20 segundos)
    for (int i = 0; i < maxLeituras; i++) {
      // Pega a distância
      leitura = distanceSensor.measureDistanceCm();

      erro_total += (desejado - leitura);

      // Mantém o motor funcionando pelo potenciômetro durante a coleta
      int valorBruto = analogRead(pinoPotenciometro);
      int valorMapeado = map(valorBruto, 0, 4095, 160, 0);
      myservo.write(valorMapeado);

      delay(dt * 1000); // Pausa de 30ms
    }

    myservo.write(90);

    Serial.println("Coleta finalizada! Enviando via Bluetooth...");

    String payload = String(erro_total, 2);

    pCharacteristic->setValue(payload.c_str());
    pCharacteristic->notify();
    
    Serial.print("Erro total enviado: ");
    Serial.println(payload);

    coletaConcluida = true;

    for (int i = 1; i <= 18; i++) {
      myservo.write(90-5*i);
      delay(100);
    }
  }
}

void pidBolaBastao() {

  desejado = 25;
  soma = 0;
  ultimo = 0;
  t = 0;
  val = 0;
  erro_total = 0;
  limitex = 50;
  leitura = 1;


  myservo.write(0);
  digitalWrite(LED_PIN, LOW);

  if (digitalRead(chave_console) == LOW) {
    
    coletaConcluida = false; // Prepara para a próxima vez que a chave subir
  }

  if (digitalRead(chave_console) == HIGH && !coletaConcluida) {

    delay(1000);

    erro_total = 0;

    digitalWrite(LED_PIN, HIGH);
    Serial.println("Iniciando coleta de 20 segundos...");

    for (int i = 0; i < maxLeituras; i++) {
      if (t >= leitura) {
        erro = (desejado - (val / leitura));
        t = 0;
        val = 0;

        erro_total += fabs(erro);
        soma += erro * dt; // Multiplicar por dt para integrar corretamente

        if (soma > 30) {
          soma = 30;
        }
        if (soma < -30) {
          soma = -30;
        }

        x = kp * erro + ki * soma + kd * (erro - ultimo) / dt; // Ajustado para dt

        ultimo = erro;

        if (x > limitex) {
          x = limitex;
        }
        if (x < -limitex) {
          x = -limitex;
        }

        u = map(x, -limitex, limitex, 0, 180); // Mapeia a saída do PID para o ângulo do servo
        myservo.write(u);
        Serial.println(u);
      }

      t += 1;

      j = distanceSensor.measureDistanceCm();

      if (j < 0) {
        j = 0;
      } else if (j > 50) {
        j = 50;
      }

      // Serial.print("Leitura:");
      // Serial.println(j);

      val += j;

      delay(dt * 1000);

    }

    myservo.write(90);

    Serial.println("Coleta finalizada! Enviando via Bluetooth...");

    String payload = String(erro_total, 2);

    pCharacteristic->setValue(payload.c_str());
    pCharacteristic->notify();
    
    Serial.print("Erro total enviado: ");
    Serial.println(payload);

    coletaConcluida = true;

    for (int i = 1; i <= 18; i++) {
      myservo.write(90-5*i);
      delay(100);
    }
  }
}

void setup() {
  Serial.begin(115200);
  Serial.println("Iniciando Servidor BLE...");

  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);
  myservo.setPeriodHertz(50);

  myservo.attach(pin_servo);

  pinMode(velmotor, OUTPUT);
  pinMode(mla, OUTPUT);
  pinMode(mlb, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  pinMode(chave_console, INPUT_PULLDOWN);
  pinMode(pinoEncoderA, INPUT_PULLUP);
  pinMode(pinoEncoderB, INPUT_PULLUP);
  pinMode(pinoPotenciometro, INPUT);

  dt = 0.03;
  erro_total = 0.0;

  digitalWrite(LED_PIN, LOW);
  digitalWrite(mla, LOW);
  digitalWrite(mlb, LOW);
  analogWrite(velmotor, 0);

  attachInterrupt(digitalPinToInterrupt(pinoEncoderA), lerEncoder, CHANGE);

  BLEDevice::init("ESP32_BLE_Control");
  BLEServer *pServer = BLEDevice::createServer();
  
  pServer->setCallbacks(new MyServerCallbacks());
  BLEService *pService = pServer->createService(SERVICE_UUID);

  pCharacteristic = pService->createCharacteristic(
                      CHARACTERISTIC_UUID,
                      BLECharacteristic::PROPERTY_READ   |
                      BLECharacteristic::PROPERTY_WRITE |
                      BLECharacteristic::PROPERTY_NOTIFY
                    );

  pCharacteristic->addDescriptor(new BLE2902());
  
  pCharacteristic->setCallbacks(new MyCallbacks());
  pCharacteristic->setValue("Status: Tudo OK na ESP32!");
  pService->start();
  
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06);  
  pAdvertising->setMinPreferred(0x12);
  BLEDevice::startAdvertising();
  
  Serial.println("Aguardando conexão...");
}

void loop() {
  if (comandoAtual == "Gamificacao Pendulo Invertido"){

    gamePenduloInvertido();
    
  } else if (comandoAtual == "Feedback Pendulo Invertido"){

    feedbackPenduloInvertido();

  }else if (comandoAtual == "Gamificacao Bola Bastao"){

    gameBolaBastao();

  }else if (comandoAtual == "PID Bola Bastao"){

    pidBolaBastao();
    
  }
}